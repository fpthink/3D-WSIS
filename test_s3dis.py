
import argparse
import collections
from math import sqrt
from re import S
import numpy as np
import os
import sys
import importlib
from sklearn.cluster import DBSCAN
import collections

import torch
import spconv
import scipy.stats as stats
from plyfile import PlyData, PlyElement
import pointgroup_ops
from torch_scatter import scatter_min, scatter_mean, scatter_max, scatter
# import sstnet

import evaluation
import utils
from utils.planeSegment import get_room_walls
from utils.eval_s3dis import S3DIS_Instance_evaluator
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'modules'))              # !!!! for importlib
sys.path.append(os.path.join(BASE_DIR, 'modules/model'))        # !!!! for importlib
sys.path.append(os.path.join(BASE_DIR, 'modules/datasets'))     # !!!! for importlib

def get_parser():
    parser = argparse.ArgumentParser(description="Point Cloud Instance Segmentation")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="path to config file")
    ### pretrain
    parser.add_argument("--pretrain", type=str, default="", help="path to pretrain model")
    ### split
    parser.add_argument("--split", type=str, default="val", help="dataset split to test")
    ### semantic only
    parser.add_argument("--semantic", action="store_true", help="only evaluate semantic segmentation")
    ### log file path
    parser.add_argument("--log-file", type=str, default=None, help="log_file path")
    ### test srcipt operation
    parser.add_argument("--eval", action="store_true", help="evaluate or not")
    parser.add_argument("--save", action="store_true", help="save results or not")
    parser.add_argument("--visual", type=str, default=None, help="visual path, give to save visualization results")

    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    cfg = utils.Config.fromfile(args.config)
    cfg.pretrain = args.pretrain
    cfg.semantic = args.semantic
    cfg.dataset.task = args.split  # change tasks ！！！！！！！！！！！！！

    cfg.data.visual = args.visual
    cfg.data.eval = args.eval
    cfg.data.save = args.save

    utils.set_random_seed(cfg.data.test_seed)

    #### get logger file
    params_dict = dict(
        epoch=cfg.data.test_epoch,
        optim=cfg.optimizer.type,
        lr=cfg.optimizer.lr,
        scheduler=cfg.lr_scheduler.type
    )
    if "test" in args.split:
        params_dict["suffix"] = "test"

    log_dir, logger = utils.collect_logger(
        prefix=os.path.splitext(args.config.split("/")[-1])[0],     # the name of the yaml file
        log_name="test_{}".format(time.time()),
        log_file=args.log_file,
        # **params_dict
    )

    logger.info("************************ Start Logging ************************")

    # log the config
    logger.info(cfg)

    # global result_dir
    # result_dir = os.path.join(
    #     log_dir,
    #     "result",
    #     "epoch_{}".format(cfg.data.test_epoch),
    #     args.split)
    # os.makedirs(os.path.join(result_dir, "predicted_masks"), exist_ok=True)

    global semantic_label_idx
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # semantic_label_idx = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]).cuda() # ScanNet
    semantic_label_idx = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) # S3DIS

    return logger, cfg


def test(model, cfg, logger):
    
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

    epoch = cfg.data.test_epoch
    semantic = cfg.semantic

    cfg.dataset.test_mode = True
    cfg.dataloader.batch_size = 1 
    cfg.dataloader.num_workers = 2

    datasets = importlib.import_module(cfg.dataset.type)
    test_dataset = datasets.S3DIS_Inst_spg(cfg.model, cfg.dataset, test_mode=True)
    
    test_dataset.aug_flag = False
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=cfg.dataloader.batch_size, 
                                                  shuffle=False,
                                                  num_workers=cfg.dataloader.num_workers,
                                                  collate_fn=test_dataset.collate_fn,
                                                  pin_memory=True,
                                                  drop_last=False)
    
    with torch.no_grad():
        model = model.eval() ##########

        # init timer to calculate time
        timer = utils.Timer()

        #####  semantic test
        point_sem_evaluator = evaluation.S3DISSemanticEvaluator(logger=logger)
        mid_sem_evaluator = evaluation.S3DISSemanticEvaluator(logger=logger)
        sem_evaluator = evaluation.S3DISSemanticEvaluator(logger=logger) 

        ##### instance test
        inst_evaluator = S3DIS_Instance_evaluator(logger=logger)


        #### instance ap

        data_root = os.path.join(os.path.dirname(__file__), cfg.dataset.data_root)
        data_root = os.path.split(data_root)[0]
        label_root = os.path.join(data_root, 'labels')
        s3dis_ap_inst_evaluator = evaluation.S3DISInstanceEvaluator(label_root, logger=logger)
        
        for i, batch in enumerate(test_dataloader):

            torch.cuda.empty_cache() # (empty cuda cache, Optional)

            ##### prepare input and forward
            coords = batch["locs"].cuda()                           # [N, 1 + 3], long, cuda, dimension 0 for batch_idx

            voxel_coords = batch["voxel_locs"].cuda()               # [M, 1 + 3], long, cuda
            p2v_map = batch["p2v_map"].cuda()                       # [N], int, cuda
            v2p_map = batch["v2p_map"].cuda()                       # [M, 1 + maxActive], int, cuda

            coords_float = batch["locs_float"].cuda()               # [N, 3], float32, cuda
            feats = batch["feats"].cuda()                           # [N, C], float32, cuda

            superpoint = batch["superpoint"].cuda()                 # [N], long, cuda
            GIs = batch["GIs"]                       # igraph


            edge_u_list = batch["edge_u_list"].cuda()
            edge_v_list = batch["edge_v_list"].cuda()

            spatial_shape = batch["spatial_shape"]

            superpoint_cenetr_xyz = scatter(coords_float, superpoint, dim=0, reduce='mean')
            extra_data = {
                "superpoint": superpoint, 
                "GIs": GIs,
                "edge_u_list": edge_u_list,
                "edge_v_list": edge_v_list,
                "superpoint_cenetr_xyz": superpoint_cenetr_xyz,
            }

            if cfg.model.use_coords:
                feats = torch.cat((feats, coords_float), 1)

            voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.data.mode)    # [M, C]

            input_ = spconv.SparseConvTensor(voxel_feats,
                                             voxel_coords.int(),
                                             spatial_shape,
                                             cfg.dataloader.batch_size)

            ret = model(input_,             # SparseConvTensor
                        p2v_map,            # [N], int, cuda
                        extra_data)         # dict

            semantic_scores = ret["semantic_scores"]    # [N, nClass] float32, cuda

            sp_semantic_scores = ret['sp_semantic_scores']

            pred_sp_offset_vectors = ret['pred_sp_offset_vectors'] 

            pred_sp_occupancy = ret['pred_sp_occupancy']

            pred_sp_instance_size = ret['pred_sp_ins_size'] 


            ################################# test #########################################
            

            test_scene_name = batch["scene_list"][0]

            scene_sem_gt = test_dataset.get_scene_sem_gt(test_scene_name)
            superpoint = superpoint.cpu().detach().numpy()  


            ##### point-level semantic segmentation evaluation
            semantic_pred = semantic_scores.max(1)[1]  # [N]
            outputs = [{"semantic_pred": semantic_pred, "semantic_gt":scene_sem_gt}]
            point_sem_evaluator.process([{}], outputs) 

            ##### middle-level semantic segmentation evaluation
            middle_level_semantic_pred = np.zeros(len(superpoint))
            point_semantic_pred = semantic_pred.cpu().detach().numpy()
            for spID in np.unique(superpoint):
                spMask = np.where(superpoint == spID)[0]
                sp_sem_label = stats.mode(point_semantic_pred[spMask])[0][0]
                middle_level_semantic_pred[spMask] = sp_sem_label
            
            middle_level_semantic_pred = middle_level_semantic_pred.astype('int')
            middle_level_semantic_pred = torch.from_numpy(middle_level_semantic_pred)

            outputs = [{"semantic_pred": middle_level_semantic_pred, "semantic_gt":scene_sem_gt}]
            mid_sem_evaluator.process([{}], outputs)

            ##### superpoint-level semantic segmentation evaluation
            sp_semantic_pred = sp_semantic_scores.max(1)[1]
            sp_semantic_pred = sp_semantic_pred.cpu().detach().numpy()
            
            assert len(sp_semantic_pred) == (superpoint.max() + 1)
            assert len(coords) == len(superpoint)
            point_level_semantic_pred = np.zeros(len(superpoint))

            for spID in np.unique(superpoint):
                spMask = np.where(superpoint == spID)[0]
                point_level_semantic_pred[spMask] = sp_semantic_pred[spID]
            
            point_level_semantic_pred = point_level_semantic_pred.astype('int')
            point_level_semantic_pred = torch.from_numpy(point_level_semantic_pred)

            outputs = [{"semantic_pred": point_level_semantic_pred, "semantic_gt":scene_sem_gt}]
            sem_evaluator.process([{}], outputs) 
            
            ##### instance segmentation evaluation
            xyz_origin = coords_float.cpu().detach().numpy()
            graph = test_dataset.get_scene_graph(test_scene_name)
            pred_sp_offset_vectors = pred_sp_offset_vectors.cpu().detach().numpy()
            pred_sp_occupancy = pred_sp_occupancy.cpu().detach().numpy()
            pred_sp_instance_size = pred_sp_instance_size.cpu().detach().numpy()

            pred_info = {}
            pred_info["scene_name"] = test_scene_name
   
            pred_info["conf"], pred_info["sem_label"], pred_info["mask"] = clustering_in_graph(test_scene_name, xyz_origin, superpoint, graph, 
                                                                            sp_semantic_pred, pred_sp_offset_vectors, pred_sp_occupancy, pred_sp_instance_size)

            gt_info = {}
            gt_info['sem_gt'] = scene_sem_gt.numpy()
            gt_info['ins_gt'] = test_dataset.get_scene_ins_gt(test_scene_name).numpy()

            inst_evaluator.process(pred_info, gt_info)

            # s3dis ap
            pred_info["label_id"] = pred_info["sem_label"]
            inputs = [{"scene_name": test_scene_name}]
            s3dis_ap_inst_evaluator.process(inputs, [pred_info])

        
        ### semantic
        logger.info("point semantic evaluation")
        point_sem_evaluator.evaluate()
        logger.info("middle-level semantic evalution")
        mid_sem_evaluator.evaluate()
        logger.info("superpoint semantic evaluation")
        sem_evaluator.evaluate()

        ### instance 
        logger.info("instance evaluation")
        inst_evaluator.evaluate()

        ###
        logger.info("instance evaluation")
        s3dis_ap_inst_evaluator.evaluate(prec_rec=False)




def clustering_in_graph(scene_name, xyz_origin, superpoint, graph, sp_semnatic_pred, pred_sp_offset_vectors, pred_sp_occupancy, pred_sp_ins_size):

    print(scene_name)
    assert len(xyz_origin) == len(superpoint)
    assert len(np.unique(superpoint)) == (superpoint.max() + 1) == len(sp_semnatic_pred) == len(pred_sp_offset_vectors)

    # show_superpoint_instance_center(scene_name, xyz_origin, superpoint, sp_semnatic_pred, pred_sp_offset_vectors)
    ######################################
    superpoint_feat = dict()
    def get_superpoint_feature(spID):
        
        if spID in superpoint_feat:
            return superpoint_feat[spID]

        superpoint_mask = (superpoint == spID)

        sp_semantic_label = sp_semnatic_pred[spID]

        superpoint_center = xyz_origin[superpoint_mask].mean(0)
        sp_instance_center = superpoint_center + pred_sp_offset_vectors[spID]

        superpoint_feat[spID] = {'superpoint_pred_label':sp_semantic_label, 'sp_instance_center':sp_instance_center, 'superpoint_mask':superpoint_mask}
        return superpoint_feat[spID]

    spID_list = np.unique(superpoint)
    superpoint_visited = { spID:False for spID in spID_list }

    def BFS(spID):
        nonlocal superpoint_visited
        superpoint_visited[spID] = True
        queue = collections.deque()
        queue.append(spID)
        group_superpoint = set()
        group_superpoint.add(spID)

        group_mask = np.zeros(len(xyz_origin)).astype(bool)

        initial_spID_feat = get_superpoint_feature(spID)
        semantic_label = initial_spID_feat['superpoint_pred_label']
        group_mask = group_mask | initial_spID_feat['superpoint_mask']

        while queue:
            cur_spID = queue.popleft()

            cur_spID_feat = get_superpoint_feature(cur_spID)
            
            # for neighbor_spID in graph[cur_spID]:
            for neighbor_spID in graph.neighbors(vertex=cur_spID, mode='all'): # igraph https://igraph.org/python/api/latest/igraph._igraph.GraphBase.html#neighbors
                neighbor_spID_feat = get_superpoint_feature(neighbor_spID)

                if (neighbor_spID_feat['superpoint_pred_label'] == semantic_label) and (superpoint_visited[neighbor_spID] == False):

                    if np.linalg.norm((cur_spID_feat['sp_instance_center']-neighbor_spID_feat['sp_instance_center']), ord=2) < 0.8 * pred_sp_ins_size[spID]: ### key threshold
                        group_superpoint.add(neighbor_spID)
                        group_mask = group_mask | neighbor_spID_feat['superpoint_mask'] ####
                        superpoint_visited[neighbor_spID] = True
                        queue.append(neighbor_spID)

        return list(group_superpoint), group_mask

    
    def get_ceiling_floor_wall():
        """
        0 : ceiling
        1 : floor
        """
        ceiling_sp = set()
        floor_sp = set()
        wall_sp = set()
        

        ceiling_mask = np.zeros(len(xyz_origin)).astype(bool)
        floor_mask = np.zeros(len(xyz_origin)).astype(bool)
        wall_mask = np.zeros(len(xyz_origin)).astype(bool)
        

        for v in graph.vs:
            spID = v['v']
            spID_feat = get_superpoint_feature(spID)

            if (spID_feat['superpoint_pred_label'] == 0):
                ceiling_mask = ceiling_mask | spID_feat['superpoint_mask']
                ceiling_sp.add(spID)
            
            if (spID_feat['superpoint_pred_label'] == 1):
                floor_mask = floor_mask | spID_feat['superpoint_mask']
                floor_sp.add(spID)
            
            if (spID_feat['superpoint_pred_label'] == 2):
                wall_mask = wall_mask | spID_feat['superpoint_mask']
                wall_sp.add(spID)
        
        ceiling_n = ceiling_mask.sum()
        floor_n = floor_mask.sum()
        wall_n = wall_mask.sum()

        ceiling = {'mask':ceiling_mask, 'classLabel':0,  'group_sp_list': list(ceiling_sp), 'group_n': ceiling_n}
        floor = {'mask':floor_mask, 'classLabel':1,  'group_sp_list': list(floor_sp), 'group_n': floor_n}
        wall = {'mask':wall_mask, 'classLabel':2,  'group_sp_list': list(wall_sp), 'group_n': wall_n}
        
        
        return ceiling, floor, wall



    #################################################################

    primary_instance_list = []
    fragment_list = []

    def get_group_pred_occupancy(group_sp_list):
        group_sp_list = np.array(group_sp_list)
        group_pred_occupancy = pred_sp_occupancy[group_sp_list]
        group_pred_occupancy = np.exp(group_pred_occupancy).mean() 
        return group_pred_occupancy

    
    def get_group_instance_center(group_list):
        instance_center = np.zeros(3)
        group_point_n = 0
        for spID in group_list:
            spID_feat = get_superpoint_feature(spID)
            instance_center += spID_feat['sp_instance_center'] * spID_feat['superpoint_mask'].sum()
            group_point_n += spID_feat['superpoint_mask'].sum()
        
        return instance_center / group_point_n
    
    def get_group_instance_size(group_sp_list):
        group_sp_list = np.array(group_sp_list)
        return np.mean(pred_sp_ins_size[group_sp_list])

    bfs_mask_list = []

    for spID in spID_list:
        spID_feat = get_superpoint_feature(spID)
        superpoint_pred_label = spID_feat['superpoint_pred_label']
        if (superpoint_visited[spID] == True): 
            continue

        if (superpoint_pred_label==0 or superpoint_pred_label==1 or superpoint_pred_label == 2):
            continue

        group_sp_list, group_mask = BFS(spID)
        ############
        bfs_mask_list.append(group_mask.astype(int))

        group_pred_occupancy = get_group_pred_occupancy(group_sp_list)
        low_thre = 0.05 * group_pred_occupancy
        high_thre = 0.3 * group_pred_occupancy

        group_xyz = xyz_origin[group_mask]
        xyz = group_xyz * 50 # 0.02 cm
        xyz = torch.from_numpy(xyz).long()
        xyz = torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(0), xyz], 1)
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(xyz, 1, 4)
        group_voxel_num = voxel_locs.shape[0] 

        group_n = group_mask.sum()

        if group_voxel_num < high_thre:
            fragment_center = get_group_instance_center(group_sp_list)
            fragment = {'mask':group_mask, 'classLabel':superpoint_pred_label, 'instance_center':fragment_center, 
                        'absorbed':False, 'group_sp_list':group_sp_list, 'group_n': group_n}
            fragment_list.append(fragment)
        else:
            r_voxel = 0.02 * sqrt(group_pred_occupancy) # 2cm = 0.02m
            r_size = 0.01 * sqrt(group_n)
            r_ins_size = get_group_instance_size(group_sp_list)
            r_set = max(r_size, r_voxel, r_ins_size)

            # r_set = get_group_instance_size(group_sp_list)

            primary_instance_center = get_group_instance_center(group_sp_list)
            primary_instance = {'mask':group_mask, 'classLabel':superpoint_pred_label, 'instance_center':primary_instance_center, 
                                'r_set':r_set, 'group_sp_list':group_sp_list, 'group_n': group_n}
            primary_instance_list.append(primary_instance)

    for fi, fragment in enumerate(fragment_list):
        index, dis_min = -1, float('inf')

        for i, primary_instance in enumerate(primary_instance_list):
            dis = fragment['instance_center'] - primary_instance['instance_center']
            assert dis.shape==(3,)
            dis = np.linalg.norm(dis, ord=2)
            if fragment['classLabel']==primary_instance['classLabel'] and dis < dis_min:
                index = i
                dis_min = dis

        if index == -1:
            continue
        
        closest_primary = primary_instance_list[index]
        if dis_min < closest_primary['r_set']:#

            _ins_mask = fragment['mask'] | closest_primary['mask']
            _center = get_group_instance_center(fragment['group_sp_list'] + closest_primary['group_sp_list'])

            _r_voxel = 0.02 * sqrt(get_group_pred_occupancy(fragment['group_sp_list'] + closest_primary['group_sp_list']))
            _r_size = 0.01 * sqrt(_ins_mask.sum())
            _r_ins_size = get_group_instance_size(fragment['group_sp_list'] + closest_primary['group_sp_list'])
            _r_set = max(_r_voxel, _r_size, closest_primary['r_set'], _r_ins_size)

            closest_primary['mask'] = _ins_mask
            closest_primary['instance_center'] = _center
            closest_primary['r_set'] = _r_set
            closest_primary['group_n'] = _ins_mask.sum()
            closest_primary['group_sp_list'] += fragment['group_sp_list']
            fragment['absorbed'] = True
    
    ############################ return result ##########################

    semantic_label_idx_s3dis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    conf, label_id, ins_mask_list = [], [], []
    for primary_instance in primary_instance_list:

        _conf = primary_instance['group_n'] / get_group_pred_occupancy(primary_instance['group_sp_list'])
        _conf = min(_conf, 1)
        conf.append(_conf)
        label_id.append(semantic_label_idx_s3dis[primary_instance['classLabel']])
        ins_mask_list.append(primary_instance['mask'].astype(int))  
    

    # ceiling floor
    ceiling, floor, wall = get_ceiling_floor_wall()

    if ceiling['group_n'] > 100:
        conf.append(1)
        label_id.append(semantic_label_idx_s3dis[ceiling['classLabel']])
        ins_mask_list.append(ceiling['mask'].astype(int)) 

    if floor['group_n'] > 100:
        conf.append(1)
        label_id.append(semantic_label_idx_s3dis[floor['classLabel']])
        ins_mask_list.append(floor['mask'].astype(int)) 
    
    walls = get_room_walls(xyz_origin, wall['mask'], max_num=10)

    for _wall in walls:
        conf.append(1)
        label_id.append(semantic_label_idx_s3dis[2])
        ins_mask_list.append(_wall.astype(int))  
    
    # show_instance_segmentation_result(scene_name, xyz_origin, ins_mask_list, name='final')
    
    return np.array(conf), np.array(label_id), np.array(ins_mask_list)



def show_instance_segmentation_result(scene_name, xyz, instance_mask_list:list, name=''):
    instance_num = len(instance_mask_list)
    instance_color_table = np.random.randint(low=0, high=255, size=(instance_num, 3))
    color = np.zeros((len(xyz), 3))

    for i, instance_mask in enumerate(instance_mask_list):
        instance_mask = instance_mask.astype(bool)
        color[instance_mask] = instance_color_table[i]

    xyz_rgb = np.concatenate((xyz, color), axis=1)
    vertex = np.array([tuple(i) for i in xyz_rgb], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    d = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([d])
    save_path = r''
    plydata.write(os.path.join(save_path, scene_name+f'_{name}.ply'))


def show_superpoint_instance_center(scene_name, xyz, superpoint, sp_semnatic_pred, pred_sp_offset_vectors):
    superpoint_num = len(np.unique(superpoint))
    color_table = np.random.randint(low=0, high=255, size=(20, 3))
    color = np.zeros((superpoint_num, 3))
    superpoint_instance_center = np.zeros((superpoint_num, 3))

    for spID in np.unique(superpoint):
        superpoint_mask = (superpoint == spID)

        sp_semantic_label = sp_semnatic_pred[spID]

        superpoint_center = xyz[superpoint_mask].mean(0)
        sp_instance_center = superpoint_center + pred_sp_offset_vectors[spID]

        superpoint_instance_center[spID] = sp_instance_center
        color[spID] = color_table[sp_semantic_label]
    
    xyz_rgb = np.concatenate((superpoint_instance_center, color), axis=1)
    vertex = np.array([tuple(i) for i in xyz_rgb], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    d = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([d])
    save_path = r''
    plydata.write(os.path.join(save_path, scene_name+'_ins_center.ply'))


if __name__ == "__main__":
    logger, cfg = init()

    ##### model
    logger.info("=> creating model ...")
    logger.info(f"Classes: {cfg.model.classes}")

    # model = xxx.build_model(cfg.model)
    # create model
    model = importlib.import_module(cfg.model.type)
    model = model.Network(cfg.model)

    use_cuda = torch.cuda.is_available()
    logger.info(f"cuda available: {use_cuda}")
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info(f"#classifier parameters (model): {sum([x.nelement() for x in model.parameters()])}")

    ##### load model
    utils.load_checkpoint(
        model, 
        cfg.pretrain,
        strict=False,
    )  # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, cfg, logger)
