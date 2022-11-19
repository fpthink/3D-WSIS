import glob
from math import sqrt
import os
import sys
import importlib
import argparse
import ast
from scipy import stats

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch_scatter import scatter_min, scatter_mean, scatter_max, scatter

import spconv

import pointgroup_ops

import evaluation
import utils
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'modules'))              # !!!! for importlib
sys.path.append(os.path.join(BASE_DIR, 'modules/model'))        # !!!! for importlib
sys.path.append(os.path.join(BASE_DIR, 'modules/datasets'))     # !!!! for importlib

def get_parser():
    # the default argument parser contains some essential parameters for distributed    
    parser = argparse.ArgumentParser(description="Point Cloud Instance Segmentation")
    parser.add_argument("--resume", action="store_true", help="whether to attempt to resume from the checkpoint directory")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--autoscale-lr", action="store_true", help="automatically scale lr with the number of gpus")
    
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid()) % 2**14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    parser.add_argument("--config", type=str, default="config/default.yaml", help="path to config file")
    args_cfg = parser.parse_args()

    return args_cfg


def initialize_dataset(cfg):

    # ----------------------------------------------------------------------------------------
    # initialize train dataset
    
    datasets = importlib.import_module(cfg.dataset.type)
    train_dataset = datasets.S3DIS_Inst_spg(cfg.model, cfg.dataset, test_mode=False)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=cfg.dataloader.batch_size,
                                                   shuffle=True,
                                                   num_workers=cfg.dataloader.num_workers,
                                                   collate_fn=train_dataset.collate_fn,
                                                   pin_memory=True,
                                                   drop_last=True)

    # initialize val dataset ------------------------------------------
    cfg.dataset.task = 'val'

    datasets = importlib.import_module(cfg.dataset.type)
    val_dataset = datasets.S3DIS_Inst_spg(cfg.model, cfg.dataset, test_mode=True)
    
    val_dataset.aug_flag = False
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=1, # single scene for test 
                                                  shuffle=False,
                                                  num_workers=2,
                                                  collate_fn=val_dataset.collate_fn,
                                                  pin_memory=True,
                                                  drop_last=False)
    cfg.dataset.task = 'train'

    return train_dataset, val_dataset, train_dataloader, val_dataloader


def do_train(model, cfg, logger, train_dataloader, val_dataloader, iteration_ind):
    model = model.train()
    
    # ----------------------------------------------------------------------------------------
    # initilize optimizer and scheduler (scheduler is optional-adjust learning rate manually)
    if cfg.optimizer.type == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    else:
        print("optimizer error!")
        exit()

    
    if cfg.lr_scheduler.type == "PolyLR":
        from utils.lr_scheduler import PolyLR
        lr_scheduler = PolyLR(optimizer,
                              max_iters=cfg.lr_scheduler.max_iters,
                              last_epoch=-1,
                              power=cfg.lr_scheduler.power,
                              constant_ending=cfg.lr_scheduler.constant_ending)
    else:
        print("schelder error!")
        exit()

    # ----------------------------------------------------------------------------------------
    # initialize criterion (Optional, can calculate in model forward)
    
    losses = importlib.import_module(cfg.loss.type)
    criterion = losses.MultiTaskLoss(logger, cfg.loss, cfg.model)


    iter, epoch = 1, 1
    


    # ----------------------------------------------------------------------------------------
    # initialize tensorboard (Optional) TODO: integrating the tensorborad manager
    writer = utils.TensorBoardWriter(cfg.log_dir)

    # ----------------------------------------------------------------------------------------
    # initialize timers (Optional)
    iter_timer = utils.Timer()
    epoch_timer = utils.Timer()

    # ----------------------------------------------------------------------------------------
    # loss/time buffer for epoch record (Optional)
    loss_buffer = utils.HistoryBuffer()
    iter_time = utils.HistoryBuffer()
    data_time = utils.HistoryBuffer()

    # ----------------------------------------------------------------------------------------
    # training

    
    while epoch <= cfg.data.epochs:

        for i, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache() # (empty cuda cache, Optional)
            # calculate data loading time
            data_time.update(iter_timer.since_last())
            ##### prepare input and forward

            voxel_coords = batch["voxel_locs"].cuda()               # [M, 1 + 3], long, cuda
            p2v_map = batch["p2v_map"].cuda()                       # [N], int, cuda
            v2p_map = batch["v2p_map"].cuda()                       # [M, 1 + maxActive], int, cuda

            coords_float = batch["locs_float"].cuda()               # [N, 3], float32, cuda
            feats = batch["feats"].cuda()                           # [N, C], float32, cuda
            semantic_labels = batch["semantic_labels"].cuda()       # [N], long, cuda
            instance_labels = batch["instance_labels"].cuda()       # [N], long, cuda, 0~total_num_inst, -100

            superpoint = batch["superpoint"].cuda()                 # [N], long, cuda
            GIs = batch["GIs"]                       # igraph
            is1ins_labels = batch["is1ins_labels"].float().cuda()           # [NE], float32, cuda

            superpoint_semantic_labels = batch["superpoint_semantic_labels"].cuda()
            superpoint_instance_labels = batch["superpoint_instance_labels"].cuda()
            sp_batch_offsets = batch["sp_batch_offsets"].cuda()
            
            superpoint_offset_vector = batch["superpoint_offset_vector"].cuda()
            superpoint_instance_voxel_num = batch["superpoint_instance_voxel_num"].cuda() 

            superpoint_instance_size = batch['superpoint_instance_size'].cuda()

            edge_u_list = batch["edge_u_list"].cuda()
            edge_v_list = batch["edge_v_list"].cuda()

            scene_list = batch["scene_list"]
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

            edge_affinity = ret['edge_affinity']

            sp_discriminative_feats = ret['sp_discriminative_feats']

            pred_sp_occupancy = ret['pred_sp_occupancy'] 

            pred_sp_instance_size = ret['pred_sp_ins_size'] 

            loss_inp = {}

            ############ point-level ##############
            loss_inp['point_labels'] = (semantic_labels, instance_labels)
            loss_inp["semantic_scores"] = (semantic_scores)

            ############ superpoint-level ##############
            loss_inp['superpoint_labels'] = (superpoint_semantic_labels, superpoint_instance_labels)
            loss_inp['sp_semantic'] = (sp_semantic_scores)
            loss_inp['sp_offset_vector'] = (pred_sp_offset_vectors, superpoint_offset_vector)

            ############ superpoint instance size info ###############
            loss_inp['sp_occupancy'] = (pred_sp_occupancy, superpoint_instance_voxel_num)
            loss_inp['sp_instance_size'] = (pred_sp_instance_size, superpoint_instance_size)
            
            ############ affinity matrix ##############
            loss_inp['sp_discriminative_features'] = (sp_discriminative_feats, sp_batch_offsets)

            loss, loss_out = criterion(loss_inp, epoch)
            loss_buffer.update(loss.data) 

            # sample the learning rate(Optional)
            lr = optimizer.param_groups[0]["lr"]
            # write tensorboard
            loss_out.update({"loss": loss, "lr": lr})
            writer.update(loss_out, iter)


            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient clamp
            for p in model.ecc.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-1, 1)
            ########

            optimizer.step()
            iter += 1

            # calculate time and reset timer(Optional)
            iter_time.update(iter_timer.since_start())
            iter_timer.reset() # record the iteration time and reset timer

            # calculate remain time(Optional)
            remain_iter = (cfg.data.epochs - epoch + 1) * len(train_dataloader) + i + 1
            remain_time = utils.convert_seconds(remain_iter * iter_time.avg) # convert seconds into "hours:minutes:sceonds"

            logger.info(f"epoch: {epoch}/{cfg.data.epochs} iter: {i + 1}/{len(train_dataloader)} "
                  f"lr: {lr:8f} loss: {loss_buffer.latest:.4f}({loss_buffer.avg:.4f}) "
                  f"data_time: {data_time.latest:.2f}({data_time.avg:.2f}) "
                  f"iter_time: {iter_time.latest:.2f}({iter_time.avg:.2f}) eta: {remain_time}")

        # updata learning rate scheduler and epoch
        lr_scheduler.step()

        # log the epoch information
        logger.info(f"epoch: {epoch}/{cfg.data.epochs}, train loss: {loss_buffer.avg}, time: {epoch_timer.since_start()}s")
        iter_time.clear()
        data_time.clear()
        loss_buffer.clear()

        # write the important information into meta
        meta = {"epoch": epoch,
                "iter": iter}
    
        # save checkpoint
        checkpoint = os.path.join(cfg.log_dir, "epoch_{:05d}_{}.pth".format(epoch, iteration_ind))

        if (epoch % cfg.data.save_freq == 0) :
            utils.save_checkpoint(model=model,
                                filename=checkpoint,
                                meta=meta)

        if (epoch % cfg.data.eval_freq == 0):
            do_validation(model, val_dataloader, cfg, epoch, logger)
            model.train()

        epoch += 1


def do_validation(model, val_dataloader, cfg, epoch, logger):
    
    with torch.no_grad():
        model = model.eval() ##########
        
        point_sem_evaluator = evaluation.S3DISSemanticEvaluator(logger=logger)
        mid_sem_evaluator = evaluation.S3DISSemanticEvaluator(logger=logger)
        sem_evaluator = evaluation.S3DISSemanticEvaluator(logger=logger) 

        for i, batch in enumerate(val_dataloader):
            torch.cuda.empty_cache() # (empty cuda cache, Optional)

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

            scene_list = batch["scene_list"]
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
            
            val_scene_name = scene_list[0]
            scene_sem_gt = val_dataloader.dataset.get_scene_sem_gt(val_scene_name)
            superpoint = superpoint.cpu().detach().numpy()

            #### point-level semantic result 
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

            #### superpoint-level semantic result ---> point-level semantic result
            sp_semantic_scores = ret['sp_semantic_scores']
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
        
        logger.info("point semantic evaluation")
        point_sem_evaluator.evaluate() # point-level semantic result
        logger.info("middle-level semantic evalution")
        mid_sem_evaluator.evaluate()
        logger.info("superpoint semantic evaluation")
        sem_evaluator.evaluate() # superpoint-level semantic result


def extend_label_to_first_order_neighbor(model, cfg, logger, train_dataset):
    logger.info("extend label to first-order neighbor ...")
    
    train_dataset.test_mode = True 
    train_dataset.aug_flag = False
    train_dataset.subsample_train = False

    update_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=1,  
                                                   shuffle=False,
                                                   num_workers=cfg.dataloader.num_workers,
                                                   collate_fn=train_dataset.collate_fn,
                                                   pin_memory=True,
                                                   drop_last=False)
    
    with torch.no_grad():
        model = model.eval() ##########

        for i, batch in enumerate(update_dataloader):
            torch.cuda.empty_cache() # (empty cuda cache, Optional)

            ##### prepare input and forward

            voxel_coords = batch["voxel_locs"].cuda()               # [M, 1 + 3], long, cuda
            p2v_map = batch["p2v_map"].cuda()                       # [N], int, cuda
            v2p_map = batch["v2p_map"].cuda()                       # [M, 1 + maxActive], int, cuda

            coords_float = batch["locs_float"].cuda()               # [N, 3], float32, cuda
            feats = batch["feats"].cuda()                           # [N, C], float32, cuda

            superpoint = batch["superpoint"].cuda()                 # [N], long, cuda
            GIs = batch["GIs"]                       # igraph

            sp_batch_offsets = batch["sp_batch_offsets"].cuda()


            edge_u_list = batch["edge_u_list"].cuda()
            edge_v_list = batch["edge_v_list"].cuda()

            scene_list = batch["scene_list"]
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


            sp_semantic_scores = ret['sp_semantic_scores']
            sp_semantic_scores = torch.softmax(sp_semantic_scores, dim=-1)
            sp_semantic_value, sp_semantic_pred = sp_semantic_scores.max(1)[0], sp_semantic_scores.max(1)[1]
            sp_semantic_value = sp_semantic_value.cpu().detach().numpy()
            sp_semantic_pred = sp_semantic_pred.cpu().detach().numpy()

            scene_name = scene_list[0]
            
            train_dataset.extend_label_to_neighbor(scene_name, sp_semantic_value, sp_semantic_pred)


    train_dataset.generate_point_level_weak_label() # generate point-level pseudo label

    ##########################################
    train_dataset.test_mode = False 
    train_dataset.aug_flag = True 
    train_dataset.subsample_train = True


def propagation_label(model, cfg, logger, train_dataset, iteration_ind):
    """
    conduct label propagation in training set
    """
    logger.info("propagating label ...")
    
    train_dataset.test_mode = True 
    train_dataset.aug_flag = False
    train_dataset.subsample_train = False

    update_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=1,  
                                                   shuffle=False,
                                                   num_workers=cfg.dataloader.num_workers,
                                                   collate_fn=train_dataset.collate_fn,
                                                   pin_memory=True,
                                                   drop_last=False)
    
    with torch.no_grad():
        model = model.eval() ##########

        for i, batch in enumerate(update_dataloader):
            torch.cuda.empty_cache() # (empty cuda cache, Optional)

            ##### prepare input and forward

            voxel_coords = batch["voxel_locs"].cuda()               # [M, 1 + 3], long, cuda
            p2v_map = batch["p2v_map"].cuda()                       # [N], int, cuda
            v2p_map = batch["v2p_map"].cuda()                       # [M, 1 + maxActive], int, cuda

            coords_float = batch["locs_float"].cuda()               # [N, 3], float32, cuda
            feats = batch["feats"].cuda()                           # [N, C], float32, cuda

            superpoint = batch["superpoint"].cuda()                 # [N], long, cuda
            GIs = batch["GIs"]                       # igraph

            sp_batch_offsets = batch["sp_batch_offsets"].cuda()

            edge_u_list = batch["edge_u_list"].cuda()
            edge_v_list = batch["edge_v_list"].cuda()

            scene_list = batch["scene_list"]
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


            sp_semantic_scores = ret['sp_semantic_scores']
            sp_semantic_scores = torch.softmax(sp_semantic_scores, dim=-1)
            sp_semantic_value, sp_semantic_pred = sp_semantic_scores.max(1)[0], sp_semantic_scores.max(1)[1]
            sp_semantic_value = sp_semantic_value.cpu().detach().numpy()
            sp_semantic_pred = sp_semantic_pred.cpu().detach().numpy()


            edge_u_list = batch["edge_u_list"].numpy()
            edge_v_list = batch["edge_v_list"].numpy()
            edge_affinity = ret['edge_affinity'].cpu().detach().numpy()
            sp_batch_offsets = batch["sp_batch_offsets"].numpy()
            spnum = sp_batch_offsets[1]
            affinity_matrix = np.zeros((spnum, spnum))

            for u, v, aff in zip(edge_u_list, edge_v_list, edge_affinity):
                affinity_matrix[u][v] = aff


            scene_name = scene_list[0]

            train_dataset.weak_label_propagation(scene_name, sp_semantic_value, sp_semantic_pred, affinity_matrix, iteration_ind) 
        
    train_dataset.generate_point_level_weak_label() # generate point-level pseudo label

    ##########################################
    train_dataset.test_mode = False 
    train_dataset.aug_flag = True 
    train_dataset.subsample_train = True


def propagation_label_to_whole_scene(model, cfg, logger, train_dataset):
    """
    generate pseudo instance in training set
    """
    logger.info("propagating label to whole scene ...")
    
    train_dataset.test_mode = True 
    train_dataset.aug_flag = False
    train_dataset.subsample_train = False

    update_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=1,  
                                                   shuffle=False,
                                                   num_workers=cfg.dataloader.num_workers,
                                                   collate_fn=train_dataset.collate_fn,
                                                   pin_memory=True,
                                                   drop_last=False)
    
    with torch.no_grad():
        model = model.eval() ##########

        for i, batch in enumerate(update_dataloader):
            torch.cuda.empty_cache() # (empty cuda cache, Optional)

            ##### prepare input and forward

            voxel_coords = batch["voxel_locs"].cuda()               # [M, 1 + 3], long, cuda
            p2v_map = batch["p2v_map"].cuda()                       # [N], int, cuda
            v2p_map = batch["v2p_map"].cuda()                       # [M, 1 + maxActive], int, cuda

            coords_float = batch["locs_float"].cuda()               # [N, 3], float32, cuda
            feats = batch["feats"].cuda()                           # [N, C], float32, cuda

            superpoint = batch["superpoint"].cuda()                 # [N], long, cuda
            GIs = batch["GIs"]                       # igraph

            sp_batch_offsets = batch["sp_batch_offsets"].cuda()


            edge_u_list = batch["edge_u_list"].cuda()
            edge_v_list = batch["edge_v_list"].cuda()

            scene_list = batch["scene_list"]
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


            sp_semantic_scores = ret['sp_semantic_scores']
            sp_semantic_scores = torch.softmax(sp_semantic_scores, dim=-1)
            sp_semantic_value, sp_semantic_pred = sp_semantic_scores.max(1)[0], sp_semantic_scores.max(1)[1]
            sp_semantic_value = sp_semantic_value.cpu().detach().numpy()
            sp_semantic_pred = sp_semantic_pred.cpu().detach().numpy()

            pred_sp_offset_vectors = ret['pred_sp_offset_vectors'] 
            pred_sp_offset_vectors = pred_sp_offset_vectors.cpu().detach().numpy()


            scene_name = scene_list[0]

            train_dataset.propagate_label_to_whole_scene(scene_name, sp_semantic_value, sp_semantic_pred, pred_sp_offset_vectors) 
        
    train_dataset.generate_point_level_weak_label(add_occupancy_signal=True, add_instance_size_signal=True) # generate point-level pseudo label

    ##########################################
    train_dataset.test_mode = False 
    train_dataset.aug_flag = True 
    train_dataset.subsample_train = True


def get_checkpoint(logger, log_dir, epoch=0, checkpoint=""):
    if not checkpoint:
        if epoch > 0:
            checkpoint = os.path.join(log_dir, "epoch_{0:05d}.pth".format(epoch))
            assert os.path.isfile(checkpoint)
            logger.info("=> resume epoch_{0:05d}.pth ...".format(epoch))
        else:
            latest_checkpoint = glob.glob(os.path.join(log_dir, "*latest*.pth"))
            if len(latest_checkpoint) > 0:
                checkpoint = latest_checkpoint[0]
                logger.info("=> resume *lastest*.pth")
            else:
                checkpoint = sorted(glob.glob(os.path.join(log_dir, "*.pth")))
                if len(checkpoint) > 0:
                    checkpoint = checkpoint[-1]
                    epoch = int(checkpoint.split("_")[-1].split(".")[0])
                    logger.info("=> resume {}.pth".format(epoch))
                else:
                    logger.info("=> new training")

    return checkpoint, epoch + 1

def main(args):
    # ----------------------------------------------------------------------------------------
    # read config file
    cfg = utils.Config.fromfile(args.config)

    # ----------------------------------------------------------------------------------------
    # get logger file
    log_dir, logger = utils.collect_logger(prefix=os.path.splitext(os.path.basename(args.config))[0])

    #### NOTE: can initlize the logger manually
    # logger = utils.get_logger(log_file)

    # ----------------------------------------------------------------------------------------
    # backup the necessary file and directory(Optional, details for source code)
    # backup_list = ["train.py", "test.py", "modules", args.config]
    # backup_dir = os.path.join(log_dir, "backup")
    # utils.backup(backup_dir, backup_list)

    # ----------------------------------------------------------------------------------------
    # merge the paramters in args into cfg
    cfg = utils.merge_cfg_and_args(cfg, args)

    cfg.log_dir = log_dir
    
    # ----------------------------------------------------------------------------------------
    # set random seed
    seed = cfg.get("seed", 0)
    utils.set_random_seed(seed)

    # ----------------------------------------------------------------------------------------
    # model
    logger.info("=> creating model ...")

    # create model
    model = importlib.import_module(cfg.model.type)
    logger.info(f"=> load model {cfg.model.type}")
    model = model.Network(cfg.model)

    model = model.cuda()
    if args.num_gpus > 1:
        # convert the BatchNorm in model as SyncBatchNorm (NOTE: this will be error for low-version pytorch!!!)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # DDP wrap model
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[utils.get_local_rank()], find_unused_parameters=True)

    # logger.info("Model:\n{}".format(model)) (Optional print model)

    # ----------------------------------------------------------------------------------------
    # count the paramters of model (Optional)
    count_parameters = sum(utils.parameter_count(model).values())
    logger.info(f"#classifier parameters new: {count_parameters}")


    ########################################################################
    train_dataset, val_dataset, train_dataloader, val_dataloader = initialize_dataset(cfg)
    

    # ----------------------------------------------------------------------------------------
    # start training

    # step 1
    do_train(model, cfg, logger, train_dataloader, val_dataloader, 'semantic') # only for semantic
    
    # step 2 label propagation
    for iteration_ind, iteration_train_epochs in enumerate([200, 200, 200]): # [80, 80, 80] 

        logger.info('propagate label with affinity , {}-th iteration'.format(iteration_ind))
        propagation_label(model, cfg, logger, train_dataset, iteration_ind)

        cfg.data.epochs = iteration_train_epochs
        cfg.lr_scheduler.max_iters = iteration_train_epochs 
        cfg.loss.joint_training_epoch = -1
        cfg.loss.supervise_sp_offset = True
        do_train(model, cfg, logger, train_dataloader, val_dataloader, iteration_ind)
    
    # step 3 pseudo instances
    propagation_label_to_whole_scene(model, cfg, logger, train_dataset)
    cfg.data.epochs = 300
    cfg.lr_scheduler.max_iters = 300  
    cfg.loss.joint_training_epoch = -1
    cfg.loss.supervise_sp_offset = True
    cfg.loss.supervise_instance_size = True
    do_train(model, cfg, logger, train_dataloader, val_dataloader, 'whole_scene')




if __name__ == "__main__":
    # get the args
    args = get_parser()

    # # auto using the free gpus
    # utils.set_cuda_visible_devices(num_gpu=args.num_gpus)
    torch.backends.cudnn.benchmark = False

    main(args)
