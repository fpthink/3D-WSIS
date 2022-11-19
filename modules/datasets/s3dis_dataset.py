import os
import copy
import time
import math
import glob
import ast
import multiprocessing as mp
from typing import Dict, List, Sequence, Tuple, Union
import scipy.ndimage
import scipy.interpolate
import itertools
import igraph
import collections

# import gorilla
# import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData, PlyElement
from scipy import stats

# import segmentator
import pointgroup_ops

import utils

from modules.model import ecc

class S3DIS_Inst_spg(Dataset):
    def __init__(self, param_model, param_dataset, test_mode: bool=False):
                 
        # initialize dataset parameters
        self.logger = utils.derive_logger(__name__)         # __name__ == modules.datasets.scannetv2
        self.data_root = param_dataset.data_root                                        # data root path
        self.full_scale = list(map(int, ast.literal_eval(param_dataset.full_scale)))    # [128, 512]
        self.scale = param_dataset.scale                                                # 50
        self.max_npoint = param_dataset.max_npoint                                      # 250000
        self.with_elastic = param_dataset.with_elastic                                  # False

        self.task = param_dataset.task                                                  # "train"
        self.test_mode = test_mode                                              # False
        self.aug_flag = "train" in self.task

        self.test_area = param_dataset.test_area  # select one S3DIS area for validation

        self.subsample_train = param_dataset.subsample_train

        self.debug = getattr(param_dataset, "debug", False)

        self.annotation_num = getattr(param_dataset, "annotation_num", 1)

        self.CLASS_NUM = param_model.classes
        
        # load files
        self.load_files()
        self.weak_Label_init()

    
    def load_files(self):

        self.logger.info('load data from: {}'.format(self.data_root))

        file_names = sorted(glob.glob(os.path.join(self.data_root, "*.pth")))
        if "train" in self.task:
            file_names = [ _ for _ in file_names if self.test_area not in _ ]
        else:
            file_names = [ _ for _ in file_names if self.test_area in _ ]

        if self.debug: file_names = file_names[:20]

        self.files = [torch.load(i) for i in utils.track(file_names)]
        self.logger.info(f"{self.task} samples: {len(self.files)}")

        # ------------------ load superpoints & superpoints graph --------------------------

        path = self.data_root
        self.superpoints = {}
        self.superpoints_graph = {}
        self.scene2files = {}
        for f in utils.track(self.files):
            coords, colors, semantic_labels, instance_labels, superpoint, scene_name = f

            self.scene2files.update({scene_name: f})

            self.superpoints.update({scene_name: superpoint})

            spg = igraph.Graph.Read_Pickle(os.path.join(path, scene_name+'_spg.dat'))

            if 'train' in self.task:
                self.acquire_weak_label(coords, semantic_labels, instance_labels, superpoint, spg, self.annotation_num)

            self.superpoints_graph.update({scene_name: spg})
        
        if 'train' in self.task:
            self.logger.info('acquire weak label finish ! annotation_num: {}'.format(self.annotation_num))
        
        print(f'len files: {len(self.files)} superpoints: {len(self.superpoints)} superpoints_graph: {len(self.superpoints_graph)}')
        assert len(self.files) == len(self.superpoints)
        assert len(self.files) == len(self.superpoints_graph)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple:

        if "test" in self.task:
            raise Exception('not implementation!')
        elif "val" in self.task:
            xyz_origin, rgb, point_semantic_label_GT, point_instance_label_GT, superpoint, scene = self.files[index]

            spg = self.superpoints_graph[scene]
            spg_weak = self.weak_label_spg[scene]
            superpoint = self.superpoints[scene]
            ##############
            semantic_label = point_semantic_label_GT
            instance_label = point_instance_label_GT
            superpoint_graph = spg_weak 

        elif "train" in self.task:
            full_xyz_origin, rgb, point_semantic_label_GT, point_instance_label_GT, superpoint, scene = self.files[index]

            point_semantic_label_weak, point_instance_label_weak = self.scene_point_level_weak_label[scene]

            spg_weak = self.weak_label_spg[scene]

            superpoint = self.superpoints[scene]

            ##############

            semantic_label = point_semantic_label_weak
            instance_label = point_instance_label_weak
            superpoint_graph = spg_weak

            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html
            if self.subsample_train:
                choice_ind = np.random.choice(full_xyz_origin.shape[0], size=(full_xyz_origin.shape[0]//4), replace=False)
                xyz_origin = full_xyz_origin[choice_ind]
                rgb = rgb[choice_ind]
                semantic_label = semantic_label[choice_ind]
                instance_label = instance_label[choice_ind]
                superpoint = superpoint[choice_ind]
            else:
                xyz_origin = full_xyz_origin

        else:
            raise Exception('not implementation!')    




        superpoint_graph = copy.deepcopy(superpoint_graph) 


        if self.aug_flag:
            xyz_middle = self.data_aug_with_graph(xyz_origin, superpoint_graph, True, True, True)
        else:
            xyz_middle = self.data_aug_with_graph(xyz_origin, superpoint_graph, False, False, False)

        ### scale
        xyz = xyz_middle * self.scale

        ### offset
        xyz_offset = xyz.min(0)
        xyz -= xyz_offset

        ### crop
        valid_idxs = np.ones(len(xyz_middle), dtype=np.bool)
        if not self.test_mode:
            xyz, valid_idxs = self.crop_v2(xyz)

        xyz_middle = xyz_middle[valid_idxs]                                         # (n, 3) np.float64
        xyz = xyz[valid_idxs]                                                       # (n, 3) np.float64
        rgb = rgb[valid_idxs]                                                       # (n, 3) np.float32
        semantic_label = semantic_label[valid_idxs]                                 # (n,) np.float64

        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)    # (n,) np.float64

        superpoint = superpoint[valid_idxs]                                         # (n,) np.int64

        subset, new_superpoint = np.unique(superpoint, return_inverse=True)         # subset: (tn,) new_superpoint (n,) valid superpoints

        G = superpoint_graph.subgraph(subset)

        inst_num, inst_infos = self.get_instance_info(xyz_middle, instance_label.astype(np.int32))
        inst_info = inst_infos["instance_info"]             # [n, 9], (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        inst_pointnum = inst_infos["instance_pointnum"]     # [num_inst], list
        
        loc = torch.from_numpy(xyz).long()                  # [N, 3]
        loc_offset = torch.from_numpy(xyz_offset).long()
        loc_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb)
        if self.aug_flag:
            feat += torch.randn(3) * 0.1
        semantic_label = torch.from_numpy(semantic_label)

        instance_label = torch.from_numpy(instance_label)
        
        superpoint = torch.from_numpy(new_superpoint)

        inst_info = torch.from_numpy(inst_info)

        return scene, loc, loc_offset, loc_float, feat, semantic_label, instance_label, superpoint, G, inst_num, inst_info, inst_pointnum
    
    def data_aug_with_graph(self, xyz, graph, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        
        # change graph
        for v in graph.vs:
            v['superpoint_offset_vector'] = np.matmul(v['superpoint_offset_vector'], m)

        return np.matmul(xyz, m)

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def elastic(self, xyz, gran, mag):
        """Elastic distortion (from point group)

        Args:
            xyz (np.ndarray): input point cloud
            gran (float): distortion param
            mag (float): distortion scalar

        Returns:
            xyz: point cloud with elastic distortion
        """
        blur0 = np.ones((3, 1, 1)).astype("float32") / 3
        blur1 = np.ones((1, 3, 1)).astype("float32") / 3
        blur2 = np.ones((1, 1, 3)).astype("float32") / 3

        bb = np.abs(xyz).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype("float32") for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(xyz_):
            return np.hstack([i(xyz_)[:,None] for i in interp])
        return xyz + g(xyz) * mag

    def crop(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        r"""
        crop the point cloud to reduce training complexity

        Args:
            xyz (np.ndarray, [N, 3]): input point cloud to be cropped

        Returns:
            Union[np.ndarray, np.ndarray]: processed point cloud and boolean valid indices
        """
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs
    
    def crop_v2(self, xyz):

        _xyz = xyz.copy()
        valid_idxs = (_xyz.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        room_range_max = xyz.max(0)
        center = xyz[np.random.choice(len(xyz))][:3]

        _x, _y = max(room_range_max[0]-center[0], center[0]), max(room_range_max[1]-center[1], center[1])

        scale = np.arange(0, 1, 0.05)

        def count_points(s):
            _dx, _dy = _x * s, _y * s
            block_min = center - [ _dx, _dy, 0]
            block_max = center + [ _dx, _dy, 0]
            point_idxs = ((xyz[:, 0] >= block_min[0]) & (xyz[:, 0] <= block_max[0]) & (xyz[:, 1] >= block_min[1]) & (xyz[:, 1] <= block_max[1]))
            return point_idxs
        
        # binary search
        low, high = 0, len(scale)-1
        while low < high:
            mid = int(math.ceil((low+high)/2))

            if count_points(scale[mid]).sum() <= self.max_npoint:
                low = mid
            else:
                high = mid - 1

        point_idxs = count_points(scale[high])

        xyz_offset = xyz[point_idxs].min(0)
        _xyz -= xyz_offset
        return _xyz, point_idxs

    def get_instance_info(self,
                          xyz: np.ndarray,
                          instance_label: np.ndarray) -> Union[int, Dict]:
        r"""
        get the informations of instances (amount and coordinates)

        Args:
            xyz (np.ndarray, [N, 3]): input point cloud data
            instance_label (np.ndarray, [N]): instance ids of point cloud

        Returns:
            Union[int, Dict]: the amount of instances andinformations
                              (coordinates and the number of points) of instances
        """
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # [n, 9], float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # [num_inst], int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}

    def get_cropped_inst_label(self,
                               instance_label: np.ndarray,
                               valid_idxs: np.ndarray) -> np.ndarray:
        r"""
        get the instance labels after crop operation and recompact

        Args:
            instance_label (np.ndarray, [N]): instance label ids of point cloud
            valid_idxs (np.ndarray, [N]): boolean valid indices

        Returns:
            np.ndarray: processed instance labels
        """
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label


    def cloud_edge_feats(self, edgeattrs):
        edgefeats = np.asarray(edgeattrs['f'])
        return torch.from_numpy(edgefeats), None
    

    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        locs = []
        loc_offset_list = []
        locs_float = []
        feats = []
        semantic_labels = []

        instance_labels = []

        instance_infos = []  # [N, 9]
        instance_pointnum = []  # [total_num_inst], int

        batch_offsets = [0]
        sp_batch_offsets = [0]
        scene_list = []
        superpoint_list = []
        graph_list = []             # save superpoint graph
        superpoint_bias = 0

        superpoint_semantic_labels = []
        superpoint_instance_labels = []
        superpoint_offset_vector_list = []

        superpoint_instance_voxel_num_list = []

        superpoint_instance_size_list = []

        total_inst_num = 0
        for i, data in enumerate(batch):
            scene, loc, loc_offset, loc_float, feat, semantic_label, instance_label, superpoint, graph, inst_num, inst_info, inst_pointnum = data
            # scene:            scene name
            # loc:              xyz
            # loc_offset:       xyz_offset
            # loc_float:        
            # feat:             rgb
            # graph:            igraph
            
            # print(f'i: {i}')
            # self.debug_superpoint(superpoint)

            scene_list.append(scene)
            # print(f'superpoint bias: {superpoint_bias}')
            # print(f'a: superpoint num {superpoint.max() + 1}')
            superpoint += superpoint_bias
            # print(f'b: superpoint num {superpoint.max() + 1}')
            # superpoint_bias += (superpoint.max() + 1)   # superpoints idx: 0...max, so max + 1 is the real number of superpoints 
            superpoint_bias = (superpoint.max() + 1)   # superpoints idx: 0...max, so max + 1 is the real number of superpoints

            sp_batch_offsets.append(superpoint_bias) 

            invalid_ids = np.where(instance_label != -100)
            instance_label[invalid_ids] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + loc.shape[0])

            locs.append(torch.cat([torch.LongTensor(loc.shape[0], 1).fill_(i), loc], 1))
            loc_offset_list.append(loc_offset)
            locs_float.append(loc_float)
            feats.append(feat)
            semantic_labels.append(semantic_label)

            instance_labels.append(instance_label)
            superpoint_list.append(superpoint)
            graph_list.append(graph)

            superpoint_semantic_labels.append(torch.tensor(graph.vs['semantic_label']))
            superpoint_instance_labels.append(torch.tensor(graph.vs['instance_label']))
            superpoint_offset_vector_list.append(torch.tensor(graph.vs['superpoint_offset_vector']))
            superpoint_instance_voxel_num_list.append(torch.tensor(graph.vs['instance_voxel_num']))
            superpoint_instance_size_list.append(torch.tensor(graph.vs['instance_size']))

            instance_infos.append(inst_info)
            instance_pointnum.extend(inst_pointnum)

        GIs = [ecc.GraphConvInfo(graph_list, self.cloud_edge_feats)]         # preprocess graph edge, GIs is a class
        
        # ------------------ merge all the scenes in the batchd --------------------------
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]
        sp_batch_offsets = torch.tensor(sp_batch_offsets, dtype=torch.int)

        locs = torch.cat(locs, 0)                                   # long [N, 1 + 3], the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)     # float [N, 3]
        superpoint = torch.cat(superpoint_list, 0).long()           # long [N]
        assert len(np.unique(superpoint)) == (superpoint.max() + 1)

        feats = torch.cat(feats, 0).to(torch.float32)               # float [N, C]
        semantic_labels = torch.cat(semantic_labels, 0).long()      # long [N]
        instance_labels = torch.cat(instance_labels, 0).long()      # long [N]
        locs_offset = torch.stack(loc_offset_list)                  # long [B, 3]


        superpoint_semantic_labels = torch.cat(superpoint_semantic_labels, 0).long()
        superpoint_instance_labels = torch.cat(superpoint_instance_labels, 0).long()

        superpoint_offset_vector_list = torch.cat(superpoint_offset_vector_list, 0).to(torch.float32)

        superpoint_instance_voxel_num_list = torch.cat(superpoint_instance_voxel_num_list, 0).to(torch.float32)
        superpoint_instance_voxel_num_list = torch.log(superpoint_instance_voxel_num_list)

        superpoint_instance_size_list = torch.cat(superpoint_instance_size_list, 0).to(torch.float32)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float [N, 9] (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int [total_num_inst]

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None) # long [3]

        # -------------------------------- voxelize ---------------------------------------
        batch_size = len(batch)
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, batch_size, 4)
        # voxel_locs:   [M, 4]     M: number of voxel   M << N
        # p2v_map:      [N, 4]
        # v2p_map:      [M, 4]
        # print(f'voxel_locs: {voxel_locs.shape}')
        # print(f'p2v_map: {p2v_map.shape}')
        # print(f'v2p_map: {v2p_map.shape}')

        # -------------------------- prepare superpoint edges -----------------------------
        is1ins_labels = GIs[0].is1ins_labels

        edges = GIs[0].edges_for_ext
        edge_u_list = torch.tensor([ edge[0] for edge in edges ]).long()
        edge_v_list = torch.tensor([ edge[1] for edge in edges ]).long()
        

        return {"locs": locs, "locs_offset": locs_offset, "voxel_locs": voxel_locs,
                "scene_list": scene_list, "p2v_map": p2v_map, "v2p_map": v2p_map,
                "locs_float": locs_float, "feats": feats,
                "semantic_labels": semantic_labels, "instance_labels": instance_labels,
                "instance_info": instance_infos, "instance_pointnum": instance_pointnum,
                "offsets": batch_offsets, "spatial_shape": spatial_shape,
                "superpoint": superpoint, "GIs": GIs, 
                "is1ins_labels": is1ins_labels, "sp_batch_offsets": sp_batch_offsets,
                "edge_u_list": edge_u_list, "edge_v_list": edge_v_list,
                "superpoint_semantic_labels":superpoint_semantic_labels,
                "superpoint_instance_labels":superpoint_instance_labels,
                "superpoint_offset_vector": superpoint_offset_vector_list,
                "superpoint_instance_voxel_num": superpoint_instance_voxel_num_list,
                "superpoint_instance_size": superpoint_instance_size_list,
                }
    

    def get_scene_graph(self, scene_name):
        return self.superpoints_graph[scene_name]

    
    def get_scene_data(self, scene_name):
        return self.scene2files[scene_name]

    def get_scene_sem_gt(self, scene_name):
        xyz_origin, rgb, point_semantic_label_GT, point_instance_label_GT, superpoint, scene = self.scene2files[scene_name]

        return torch.from_numpy(point_semantic_label_GT.astype('int'))
    
    def get_scene_ins_gt(self, scene_name):
        xyz_origin, rgb, point_semantic_label_GT, point_instance_label_GT, superpoint, scene = self.scene2files[scene_name]

        return torch.from_numpy(point_instance_label_GT.astype('int'))

    def weak_Label_init(self):

        self._weak_label_spg_init()
        self.generate_point_level_weak_label()
    

    def _weak_label_spg_init(self):
        
        self.weak_label_spg = {}

        for i in range(len(self.files)):
            xyz_origin, rgb, semantic_label, instance_label, superpoint, scene = self.files[i]
            # superpoint = self.superpoints[scene]
            superpoint_graph = self.superpoints_graph[scene]
            spg_copy = copy.deepcopy(superpoint_graph)

            for e in spg_copy.es:
                s, t = e.source, e.target
                if (spg_copy.vs[s]['instance_label'] == -100) or (spg_copy.vs[t]['instance_label'] == -100):
                    e['is1ins'] = 0
                elif (spg_copy.vs[s]['instance_label'] == spg_copy.vs[t]['instance_label']):
                    e['is1ins'] = -1
                elif (spg_copy.vs[s]['instance_label'] != spg_copy.vs[t]['instance_label']):
                    e['is1ins'] = 1
                else:
                    raise Exception('unknown case!')

            self.weak_label_spg.update({scene: spg_copy})
    

    def cal_occupancy(self, xyz_origin, instance_label, superpoint_graph, add_occupancy_signal=False):

        if add_occupancy_signal == False:
            for v in superpoint_graph.vs:
                v['instance_voxel_num'] = 0

            return

        xyz = xyz_origin * self.scale

        xyz = torch.from_numpy(xyz).long()
        xyz = torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(0), xyz], 1)

        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(xyz, 1, 4)
        p2v_map = p2v_map.numpy()

        """
        G = igraph.Graph(n=spnum, edges=edges.tolist(), directed=True,
                        edge_attrs={'f': edge_feats, 'is1ins': is1ins},
                        vertex_attrs={'v': list(range(spnum)), 'semantic_label': sp_semantic, 'instance_label': sp_instance, 'superpoint_offset_vector':sp_offset_vector})
        """
        
        for v in superpoint_graph.vs:
            sp_ins_label = v['instance_label']
            ins_mask = (instance_label == sp_ins_label)
            ins_voxel_ind = p2v_map[ins_mask]
            ins_voxel_num = len(np.unique(ins_voxel_ind))
            v['instance_voxel_num'] = ins_voxel_num
    

    def cal_instance_size(self, superpoint_graph, add_instance_size_signal=False):

        if add_instance_size_signal == False:
            for v in superpoint_graph.vs:
                v['instance_size'] = 0
        
            return
        
        max_instance_radius = collections.defaultdict(int)

        for v in superpoint_graph.vs:
            instance_label = int(v['instance_label'])
            superpoint_offset_vector = np.array(v['superpoint_offset_vector'])
            assert superpoint_offset_vector.shape == (3,)
            radius = np.linalg.norm(superpoint_offset_vector, ord=2)
            max_instance_radius[instance_label] = max(max_instance_radius[instance_label], radius)
        
        for v in superpoint_graph.vs:
            instance_label = int(v['instance_label'])
            v['instance_size'] = max_instance_radius[instance_label]
    

    def generate_point_level_weak_label(self, add_occupancy_signal=False, add_instance_size_signal=False):

        self.scene_point_level_weak_label = {}

        for index in range(len(self.files)):
            xyz_origin, rgb, semantic_label, instance_label, superpoint, scene = self.files[index]

            superpoint = self.superpoints[scene]

            # generate point-level pseudo labels
            weak_label_superpoint_graph = self.weak_label_spg[scene]


            point_num = len(xyz_origin)
            weak_semantic_label = np.ones(point_num) * -100.
            weak_instance_label = np.ones(point_num) * -100.

            for v in weak_label_superpoint_graph.vs:
                if v['semantic_label'] != -100 and v['instance_label'] != -100:
                    spId = v['v']
                    spMask = np.where(superpoint == spId)[0]
                    weak_semantic_label[spMask] = v['semantic_label']
                    weak_instance_label[spMask] = v['instance_label']
            
            self.cal_occupancy(xyz_origin, weak_instance_label, weak_label_superpoint_graph, add_occupancy_signal)
            self.cal_instance_size(weak_label_superpoint_graph, add_instance_size_signal)
            
            self.scene_point_level_weak_label.update({scene:(weak_semantic_label, weak_instance_label)})

        ##################################################

        if self.task != 'train':
            return

        # GT
        GT_all, GT_label = 0, 0

        # propagation label
        # semantic
        semantic_label_num, correct_semantic_label_num = 0, 0
        instance_label_num, correct_instance_label_num = 0, 0

        # special
        floor_wall_sem_num, floor_wall_correct_sem_num = 0, 0


        for index in range(len(self.files)):
            xyz_origin, rgb, point_semantic_label_GT, point_instance_label_GT, superpoint, scene = self.files[index]

            point_semantic_label_weak, point_instance_label_weak = self.scene_point_level_weak_label[scene]

            assert len(xyz_origin) == len(point_semantic_label_GT) == len(point_instance_label_GT) == len(point_semantic_label_weak) == len(point_instance_label_weak)

            # GT
            GT_all += len(xyz_origin)
            GT_label += (point_semantic_label_GT != -100).sum()

            # propagation
            # semantic
            semantic_label_num += (point_semantic_label_weak != -100).sum()
            correct_semantic_label_num += ((point_semantic_label_weak == point_semantic_label_GT) & (point_semantic_label_weak != -100)).sum()

            floor_wall_sem_num += ((point_semantic_label_weak != -100) & ((point_semantic_label_weak == 0)|(point_semantic_label_weak == 1))).sum()
            floor_wall_correct_sem_num += ((point_semantic_label_weak == point_semantic_label_GT) 
                                            & (point_semantic_label_weak != -100) 
                                            & ((point_semantic_label_weak == 0)|(point_semantic_label_weak == 1))).sum()

            # instance
            instance_label_num += ((point_instance_label_weak != -100) & (point_semantic_label_weak != 0) & (point_semantic_label_weak != 1)).sum()
            correct_instance_label_num += ((point_instance_label_weak == point_instance_label_GT) 
                                            & (point_instance_label_weak != -100) 
                                            & (point_semantic_label_weak != 0) 
                                            & (point_semantic_label_weak != 1)).sum()
        
        self.logger.info("propagation label information:")
        self.logger.info("------------- semantic -------------")
        self.logger.info("semantic labeled / all points num: {:.2%} ( floor & wall: {:.2%} other: {:.2%})".format(
                                                                                                        semantic_label_num / GT_all, 
                                                                                                        floor_wall_sem_num / GT_all, 
                                                                                                        (semantic_label_num - floor_wall_sem_num) / GT_all
                                                                                                        ))
        self.logger.info("semantic labeled / GT labels num: {:.2%} ( floor & wall: {:.2%} other: {:.2%})".format(
                                                                                                        semantic_label_num / GT_label, 
                                                                                                        floor_wall_sem_num / GT_label, 
                                                                                                        (semantic_label_num - floor_wall_sem_num) / GT_label
                                                                                                        ))
        self.logger.info("propagation semantic label accuracy: {:.2%}".format(correct_semantic_label_num / semantic_label_num))
        self.logger.info("floor & wall accuracy: {:.2%}".format(floor_wall_correct_sem_num / floor_wall_sem_num))
        self.logger.info("other accuracy: {:.2%}".format((correct_semantic_label_num - floor_wall_correct_sem_num) / (semantic_label_num - floor_wall_sem_num)))

        self.logger.info("------------- instance (without floor & wall) -------------")
        self.logger.info("instance labeled / all: {:.2%}".format(instance_label_num / GT_all))
        self.logger.info("instance labeled / GT labels num: {:.2%}".format(instance_label_num / GT_label))
        self.logger.info("propagation instance label accuracy: {:.2%}".format(correct_instance_label_num / instance_label_num))

    def weak_label_propagation(self, scene_name, sp_semantic_value, superpoint_pred_semantic, superpoint_affinity_matrix, iterations_num):
        """
        conduct label propagation

        affinity matrix : predicted by network
        adjacency matrix 
        semantic matrix : predicted by network
        """
        
        superpoint_graph = self.superpoints_graph[scene_name]

        superpoint_num = superpoint_graph.vcount()
        superpoint_semantic_label = superpoint_graph.vs['semantic_label']
        superpoint_semantic_label = np.array(superpoint_semantic_label)

        adjacency_matrix = superpoint_graph.get_adjacency().data #  <class 'igraph.datatypes.Matrix'>  .data 
        adjacency_matrix = np.array(adjacency_matrix)
        _adj = np.eye(superpoint_num)
        adjacency_matrix = adjacency_matrix + _adj 

        assert adjacency_matrix.shape == superpoint_affinity_matrix.shape
        
        scores_list = []
        pseudo_label_list = []

        for i in range(self.CLASS_NUM): # scannet v2 has 20 categories

            if (superpoint_semantic_label == i).sum() == 0: 
                continue
            
            semantic_martix = np.zeros(adjacency_matrix.shape)


            semantic_martix[((superpoint_pred_semantic==i)&(sp_semantic_value > 0.7))] = ((superpoint_pred_semantic==i)&(sp_semantic_value > 0.7)).astype('int')
            for _ind, flag in enumerate((superpoint_semantic_label == i)):
                if flag:
                    semantic_martix[_ind][_ind] = 1

            weight_matrix = superpoint_affinity_matrix * adjacency_matrix * semantic_martix


            d_matrix = np.sum(weight_matrix, axis=1, keepdims=True)
            d_matrix[d_matrix == 0] += 1 # prevent division by 0
            trans_matrix = weight_matrix / d_matrix

            
            t = trans_matrix
            for _ in range(iterations_num): # 0 = 1 iteration  1 = 2 iteration
                trans_matrix = np.dot(trans_matrix, t)
            
            instance_prob = np.zeros(trans_matrix.shape)
            instance_prob[superpoint_semantic_label == i] = trans_matrix[superpoint_semantic_label == i]

            instance_scores = np.max(instance_prob, axis=0)
            instance_pseudo_label = np.argmax(instance_prob, axis=0)

            scores_list.append(instance_scores)
            pseudo_label_list.append(instance_pseudo_label)


        scores_list = np.array(scores_list)
        pseudo_label_list = np.array(pseudo_label_list)

        # Choose the weak label with the highest value as the pseudo label
        _ind = np.argmax(scores_list, axis=0)
        pseudo_label = np.choose(_ind, pseudo_label_list)
        pseudo_label_scores = np.choose(_ind, scores_list)

        pseudo_label_final = np.ones(superpoint_num) * -100
        unknown_region_mask = (pseudo_label_scores!=0) & (superpoint_semantic_label==-100)
        pseudo_label_final[unknown_region_mask] = pseudo_label[unknown_region_mask]

        assert pseudo_label_final.shape == (superpoint_num, )


        spg_copy = copy.deepcopy(superpoint_graph)
        """
        G = igraph.Graph(n=spnum, edges=edges, directed=True,
                        edge_attrs={'f': edges_features, 'is1ins': is1ins},
                        vertex_attrs={'v': list(range(spnum)), 'semantic_label': sp_semantic, 'instance_label': sp_instance, 'superpoint_offset_vector':sp_offset_vector})
        """
        xyz_origin, rgb, point_semantic_label_GT, point_instance_label_GT, superpoint, scene = self.scene2files[scene_name]
        for i, ind in enumerate(pseudo_label_final):
            i, ind = int(i), int(ind)
            if ind != -100:
                spg_copy.vs[i]['semantic_label'] = superpoint_graph.vs[ind]['semantic_label']
                spg_copy.vs[i]['instance_label'] = superpoint_graph.vs[ind]['instance_label']

                spMask = (superpoint == ind)
                sp_center = xyz_origin[spMask].mean(0)
                instance_center = sp_center + superpoint_graph.vs[ind]['superpoint_offset_vector']

                pseudo_spMask = (superpoint == i)
                pseudo_sp_center = xyz_origin[pseudo_spMask].mean(0)
                offset_vector = instance_center - pseudo_sp_center
                spg_copy.vs[i]['superpoint_offset_vector'] = offset_vector

        for e in spg_copy.es:
            s, t = e.source, e.target
            if (spg_copy.vs[s]['instance_label'] == -100) or (spg_copy.vs[t]['instance_label'] == -100):
                e['is1ins'] = 0
            elif (spg_copy.vs[s]['instance_label'] == spg_copy.vs[t]['instance_label']):
                e['is1ins'] = -1
            elif (spg_copy.vs[s]['instance_label'] != spg_copy.vs[t]['instance_label']):
                e['is1ins'] = 1
            else:
                raise Exception('unknown case!')
        
        self.weak_label_spg.update({scene_name: spg_copy})

        # show scene label
        # show_weak_label(scene_name, xyz_origin, superpoint, point_instance_label_GT,  spg_copy, iterations_num)
        
        # Remember to call self.generate_weak_label() to generate point-level pseudo labels
    

    def extend_label_to_neighbor(self, scene_name, sp_semantic_value, sp_semantic_pred):


        superpoint_graph = self.superpoints_graph[scene_name]
        spg_copy = copy.deepcopy(superpoint_graph)

        xyz_origin, rgb, point_semantic_label_GT, point_instance_label_GT, superpoint, scene = self.scene2files[scene_name]

        for ind, v in enumerate(superpoint_graph.vs):
            if v['semantic_label'] != -100 and v['instance_label'] != -100:
                # ind = v['v']
                for neighbor_sp in superpoint_graph.neighbors(vertex=ind, mode='all'):
                    if (sp_semantic_pred[neighbor_sp] == v['semantic_label']) and (sp_semantic_value[neighbor_sp] > 0.8) \
                        and (superpoint_graph.vs[neighbor_sp]['semantic_label']==-100 and superpoint_graph.vs[neighbor_sp]['instance_label']==-100):

                        spg_copy.vs[neighbor_sp]['semantic_label'] = superpoint_graph.vs[ind]['semantic_label']
                        spg_copy.vs[neighbor_sp]['instance_label'] = superpoint_graph.vs[ind]['instance_label']

                        spMask = (superpoint == ind)
                        sp_center = xyz_origin[spMask].mean(0)
                        instance_center = sp_center + superpoint_graph.vs[ind]['superpoint_offset_vector']

                        pseudo_spMask = (superpoint == neighbor_sp)
                        pseudo_sp_center = xyz_origin[pseudo_spMask].mean(0)
                        offset_vector = instance_center - pseudo_sp_center
                        spg_copy.vs[neighbor_sp]['superpoint_offset_vector'] = offset_vector
        
        for e in spg_copy.es:
            s, t = e.source, e.target
            if (spg_copy.vs[s]['instance_label'] == -100) or (spg_copy.vs[t]['instance_label'] == -100):
                e['is1ins'] = 0
            elif (spg_copy.vs[s]['instance_label'] == spg_copy.vs[t]['instance_label']):
                e['is1ins'] = -1
            elif (spg_copy.vs[s]['instance_label'] != spg_copy.vs[t]['instance_label']):
                e['is1ins'] = 1
            else:
                raise Exception('unknown case!')

        self.weak_label_spg.update({scene_name: spg_copy})

        # show scene label
        # show_weak_label(scene_name, xyz_origin, superpoint, point_instance_label_GT,  spg_copy, 'semantic')
    
    def propagate_label_to_neighbor(self, scene_name, sp_semantic_value, sp_semantic_pred):

        # superpoint_graph = self.superpoints_graph[scene_name]

        weak_label_superpoint_graph = self.weak_label_spg[scene_name]

        spg_copy = copy.deepcopy(weak_label_superpoint_graph)

        xyz_origin, rgb, point_semantic_label_GT, point_instance_label_GT, superpoint, scene = self.scene2files[scene_name]

        for ind, v in enumerate(weak_label_superpoint_graph.vs):
            if v['semantic_label'] != -100 and v['instance_label'] != -100:
                # ind = v['v']
                for neighbor_sp in weak_label_superpoint_graph.neighbors(vertex=ind, mode='all'):
                    # if (sp_semantic_pred[neighbor_sp] == v['semantic_label']) and (sp_semantic_value[neighbor_sp] > 0.7) \
                    #     and (weak_label_superpoint_graph.vs[neighbor_sp]['semantic_label']==-100 and weak_label_superpoint_graph.vs[neighbor_sp]['instance_label']==-100):
                    if (sp_semantic_pred[neighbor_sp] == v['semantic_label'])  \
                        and (weak_label_superpoint_graph.vs[neighbor_sp]['semantic_label']==-100 and weak_label_superpoint_graph.vs[neighbor_sp]['instance_label']==-100):

                        spg_copy.vs[neighbor_sp]['semantic_label'] = weak_label_superpoint_graph.vs[ind]['semantic_label']
                        spg_copy.vs[neighbor_sp]['instance_label'] = weak_label_superpoint_graph.vs[ind]['instance_label']

                        spMask = (superpoint == ind)
                        sp_center = xyz_origin[spMask].mean(0)
                        instance_center = sp_center + weak_label_superpoint_graph.vs[ind]['superpoint_offset_vector']

                        pseudo_spMask = (superpoint == neighbor_sp)
                        pseudo_sp_center = xyz_origin[pseudo_spMask].mean(0)
                        offset_vector = instance_center - pseudo_sp_center
                        spg_copy.vs[neighbor_sp]['superpoint_offset_vector'] = offset_vector
        
        for e in spg_copy.es:
            s, t = e.source, e.target
            if (spg_copy.vs[s]['instance_label'] == -100) or (spg_copy.vs[t]['instance_label'] == -100):
                e['is1ins'] = 0
            elif (spg_copy.vs[s]['instance_label'] == spg_copy.vs[t]['instance_label']):
                e['is1ins'] = -1
            elif (spg_copy.vs[s]['instance_label'] != spg_copy.vs[t]['instance_label']):
                e['is1ins'] = 1
            else:
                raise Exception('unknown case!')

        self.weak_label_spg.update({scene_name: spg_copy})

        # show scene label
        # show_weak_label(scene_name, xyz_origin, superpoint, point_instance_label_GT,  spg_copy, 'semantic')




    def propagate_label_to_whole_scene(self, scene_name, sp_semantic_value, sp_semantic_pred, pred_sp_offset_vectors):

        xyz_origin, rgb, point_semantic_label_GT, point_instance_label_GT, superpoint, scene = self.scene2files[scene_name]

        superpoint_graph = self.superpoints_graph[scene_name]

        
        prior_instance_center = []
        prior_instance_label = []
        prior_semantic_label = []
        

        for spID, v in enumerate(superpoint_graph.vs):
            if v['semantic_label'] != -100 and v['instance_label'] != -100:
                
                spMask = (superpoint == spID)
                sp_center = xyz_origin[spMask].mean(0)
                instance_center = sp_center + superpoint_graph.vs[spID]['superpoint_offset_vector']

                prior_instance_center.append(instance_center)
                prior_instance_label.append(v['instance_label'])
                prior_semantic_label.append(v['semantic_label'])
        
        prior_instance_center = np.array(prior_instance_center)
        prior_instance_label = np.array(prior_instance_label)
        prior_semantic_label = np.array(prior_semantic_label)

       
        spg_copy = copy.deepcopy(superpoint_graph)
        
        ins_sp_set = collections.defaultdict(set) 

        for spID, v in enumerate(spg_copy.vs):
            if v['semantic_label'] != -100 and v['instance_label'] != -100 : 
                continue
            
            spMask = (superpoint == spID)
            sp_center = xyz_origin[spMask].mean(0)
            sp_pred_instance_center = sp_center + pred_sp_offset_vectors[spID]
            sp_pred_semantic_label = sp_semantic_pred[spID]

            
            if (prior_semantic_label == sp_pred_semantic_label).sum() == 0:
                continue 

            prior_selected_ind = np.where(prior_semantic_label == sp_pred_semantic_label)[0]
            

            _prior_instance_center = prior_instance_center[prior_selected_ind]
            _prior_instance_label = prior_instance_label[prior_selected_ind]
            _prior_semantic_label = prior_semantic_label[prior_selected_ind]

            _center_dis = _prior_instance_center - sp_pred_instance_center
            _center_dis = np.linalg.norm(_center_dis, ord=2, axis=1)

            closest_ind = np.argmin(_center_dis)
            if _center_dis[closest_ind] > 1.2: 
                continue


            closest_prior_ind = prior_selected_ind[closest_ind]
            ins_sp_set[closest_prior_ind].add(spID)


        def cal_pseudo_instance_center(sp_list):
            pseudo_instance_center = np.zeros(3)
            point_n = 0
            for spID in sp_list:
                spMask = (superpoint == spID)
                pseudo_instance_center += xyz_origin[spMask].sum(0)
                point_n += spMask.sum(0)

            return pseudo_instance_center / point_n

        pseudo_instance_center_list = [] # 
 
        for closest_prior_ind, sp_set in ins_sp_set.items():
            sp_list = list(sp_set)
            pseudo_instance_center = cal_pseudo_instance_center(sp_list)
            assert pseudo_instance_center.shape == (3, )
            pseudo_instance_center_list.append(pseudo_instance_center) # show

            for spID in sp_list:
                spMask = (superpoint == spID)
                sp_center = xyz_origin[spMask].mean(0)
                spg_copy.vs[spID]['semantic_label'] = prior_semantic_label[closest_prior_ind]
                spg_copy.vs[spID]['instance_label'] = prior_instance_label[closest_prior_ind]
                _offset = pseudo_instance_center - sp_center
                assert _offset.shape == (3, )
                spg_copy.vs[spID]['superpoint_offset_vector'] = _offset
        
        self.weak_label_spg.update({scene_name: spg_copy})

        # show_weak_label(scene_name, xyz_origin, superpoint, point_instance_label_GT,  spg_copy, 'whole_scene_pseudo_label')
        # show_pseudo_center(scene_name, np.array(pseudo_instance_center_list))
    
    def acquire_weak_label(self, xyz, semantic_labels, instance_labels, superpoint, graph, annotation_num=1):
    
        superpoint = superpoint.astype('int')
        spnum = len(np.unique(superpoint))
        instance_sp_list = collections.defaultdict(list)
        sp_semantic = np.ones(spnum) * -100
        sp_instance = np.ones(spnum) * -100

        for spId in np.unique(superpoint):

            spMask = np.where(superpoint == spId)[0]
            sp_point_num = len(spMask)

            # semantic label
            sp_sem_label = semantic_labels[spMask]
            sp_sem_label = stats.mode(sp_sem_label)[0][0]
            sp_semantic[spId] = sp_sem_label

            # instance label
            sp_ins_label = instance_labels[spMask]
            sp_ins_label = stats.mode(sp_ins_label)[0][0]
            sp_instance[spId] = sp_ins_label

            instance_sp_list[sp_ins_label].append((spId, sp_point_num))
        
        choice_sp_list = []
        for ins_label in np.unique(instance_labels):
            if ins_label not in instance_sp_list:
                continue
            
            ############### The selection is based on the size of the superpoint, which is approximately equal to point-level random selection. ###############
            sp_list = instance_sp_list[ins_label]
            spId_list = np.array([ _[0] for _ in sp_list])
            sp_point_num = np.array([ _[1] for _ in sp_list])
            sp_prob = sp_point_num / sp_point_num.sum()
            
            if annotation_num < spId_list.shape[0]:
                choice_sp = np.random.choice(spId_list, size=annotation_num, p=sp_prob, replace=False)
            else:
                choice_sp = spId_list
            choice_sp_list.extend(list(choice_sp))

            ############### Pick the largest superpoint ###############
            # sp_list = instance_sp_list[ins_label]
            # sp_list = sorted(sp_list, key=lambda x: x[1], reverse=True) # desc
            # choice_sp = sp_list[0][0]
            # choice_sp_list.append(choice_sp)

            # calculate the centor of instances
            weak_label_instance_mask = np.isin(superpoint, choice_sp)
            weak_label_instance_center = np.mean(xyz[weak_label_instance_mask], axis=0)

            for _choice_sp in choice_sp:
                assert graph.vs[_choice_sp]['v'] == _choice_sp
                spMask = (superpoint == _choice_sp)
                sp_center = np.mean(xyz[spMask], axis=0)
                _sp_offset_vector = weak_label_instance_center - sp_center
                graph.vs[_choice_sp]['superpoint_offset_vector'] = _sp_offset_vector
            

        for vi, v in enumerate(graph.vs):
            assert vi == v['v']
            if v['v'] not in choice_sp_list:
                v['semantic_label'] = -100
                v['instance_label'] = -100
                v['superpoint_offset_vector'] = np.array([0., 0., 0.])



def show_weak_label(scene_name, xyz, superpoint, point_instance_label_GT, graph, discribe):
    
    # instance_color_table = np.random.randint(low=0, high=255, size=(len(np.unique(point_instance_label_GT)), 3))
    instance_color_table = np.random.randint(low=0, high=255, size=(200, 3))
    color = np.zeros((len(xyz), 3))

    for v in graph.vs:
        if v['semantic_label'] != -100 and v['instance_label'] != -100 and v['semantic_label'] != 0 and v['semantic_label'] != 1: 
            spId = v['v']
            instance_label = int(v['instance_label'])
            spMask = (superpoint == spId)
            color[spMask] = instance_color_table[instance_label]
    
    xyz_rgb = np.concatenate((xyz, color), axis=1)
    vertex = np.array([tuple(i) for i in xyz_rgb], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    d = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([d])
    save_path = r''
    plydata.write(os.path.join(save_path, scene_name+'_{}.ply'.format(discribe)))
            

def show_pseudo_center(scene_name, choice_center_xyz_list):
    
    red_color = np.zeros((len(choice_center_xyz_list), 3))
    red_color[:, 0] = 255

    xyz_rgb = np.concatenate((choice_center_xyz_list, red_color), axis=1)

    vertex = np.array([tuple(i) for i in xyz_rgb], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    d = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([d])
    
    save_path = r''
    plydata.write(os.path.join(save_path, scene_name+'_pseudo_center.ply'))