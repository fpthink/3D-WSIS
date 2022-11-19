import os
import os.path as osp
import json
import argparse
import multiprocessing as mp

import torch
import plyfile
import numpy as np
import itertools
import igraph
import glob
from scipy import stats
from plyfile import PlyData, PlyElement
from torch_scatter import scatter
from sklearn.neighbors import KDTree
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from numpy import linalg as LA



g_classes = [
    'ceiling',
    'floor',
    'wall',
    'beam',
    'column',
    'window',
    'door',
    'table',
    'chair',
    'sofa',
    'bookcase',
    'board',
    'clutter']
    
g_class2label = {cls: i for i, cls in enumerate(g_classes)}



def proc_data(room_path):

    #####
    # print(room_path)

    area_room_name = os.path.basename(room_path).split('.')[0]
    print(area_room_name)

    graph_save_path = osp.join(data_save_path, area_room_name + "_spg.dat")
    pth_path = osp.join(data_save_path, area_room_name + '.pth')

    if osp.exists(graph_save_path) and osp.exists(pth_path):
        print('{} has been processed!'.format(area_room_name))
        return

    room_data = np.load(room_path)

    xyz, rgb, semantic_labels, instance_labels, point_level_superpoint = \
    room_data[:, :3], room_data[:, 3:6], room_data[:, 6].astype('int'), room_data[:, 7].astype('int'), room_data[:, 8].astype('int')


    sem_valid = (semantic_labels != -100)
    ins_valid = (instance_labels != -100)
    assert (sem_valid != ins_valid).sum() == 0


    # check label
    assert -100 not in semantic_labels
    assert -100 not in instance_labels
    assert len(np.unique(point_level_superpoint)) == point_level_superpoint.max()+1, '{} error'.format(area_room_name)

    xyz_shift = -xyz.mean(0)
    xyz = np.ascontiguousarray(xyz + xyz_shift)
    rgb = np.ascontiguousarray(rgb) / 127.5 - 1

    # visualization 
    if vis_path != None:
        print('{} visualization: {}'.format(area_room_name, vis_path))
        show_instance(xyz, instance_labels, vis_path, area_room_name)

    # generate superpoint graph
    graph = build_graph_10NBR(area_room_name, xyz, point_level_superpoint, semantic_labels, instance_labels)

    graph_save_path = osp.join(data_save_path, area_room_name + "_spg.dat")
    print(f'save_path_graph => {graph_save_path}')

    graph.write_pickle(graph_save_path)

    # save data to *.pth
    pth_path = osp.join(data_save_path, area_room_name + '.pth')
    torch.save((xyz, rgb, semantic_labels, instance_labels, point_level_superpoint, area_room_name), pth_path)
    print("Saving to {}".format(pth_path))

    return 



def build_graph_10NBR(room_name, xyz, superpoint, semantic_labels, instance_labels):

    # Note that here we check the instance labels and semantic labels
    # if semantic labels == -100 (unannotated), instance labels should be -100
    # if semantic labels > 1, instance labels should not be -100
    assert semantic_labels.shape == instance_labels.shape
    invalid_sp_mask = (semantic_labels == -100)
    assert (instance_labels[invalid_sp_mask] == -100).sum() == invalid_sp_mask.sum()
    
    print('building {} graph ...'.format(room_name))

    unique_sp = np.unique(superpoint)
    spnum = len(unique_sp)

    # make sure that the superpoint idx counts from zero
    assert spnum == (superpoint.max() + 1)
    sp_semantic = np.ones(spnum) * -100         # superpoint semantic label
    sp_instance = np.ones(spnum) * -100         # superpoint instance label
    sp_offset_vector = np.zeros((spnum, 3))     # instance center to each superpoint center (instance center - superpoint center)


    # calculate each instance center
    instance_center = dict()
    for insID in np.unique(instance_labels):
        instance_mask = (instance_labels == insID)
        instance_center[insID] = np.mean(xyz[instance_mask], axis=0)



    # calculate the superpoint center as the superpoint coordinate
    assert len(xyz) == len(superpoint)
    superpoint_center = np.zeros((spnum, 3), dtype='float32')
    for spID in unique_sp:
        spMask = np.where(superpoint == spID)[0]
        superpoint_center[spID] = xyz[spMask].mean(0)
    


    # edge collect
    edges = set()
    tree = KDTree(superpoint_center)
    # ind = tree.query_radius(superpoint_center, r=0.4, return_distance=False, count_only=False)
    # for s, t_list in enumerate(ind):
    #     cnt = 0
    #     # note that t_list[0] is the superpoint itself, so we enumerate from 1
    #     for t in t_list[1:]:
    #         if cnt >= 10:
    #             break
    #         if (s, t) not in edges:
    #             edges.add((s, t))
    #             edges.add((t, s))
    #             cnt += 1


    # find 10 NN
    ind = tree.query(superpoint_center, k=11, return_distance=False)
    for s, t_list in enumerate(ind):
        for t in t_list[1:]:
            if (s, t) not in edges:
                edges.add((s, t))
                edges.add((t, s))



    edges = sorted(list(edges)) # [(a, b), (c, d)]

    # 
    is1ins = []
    assert len(superpoint) == len(instance_labels)
    for spId in np.unique(superpoint):
        spMask = np.where(superpoint == spId)[0]

        # semantic label
        sp_sem_label = semantic_labels[spMask]
        # stats.mode: return an array of the modal (most common) value in the passed array
        # mode: array of modal values, count: array of counts for each mode
        sp_sem_label = stats.mode(sp_sem_label)[0][0]
        sp_semantic[spId] = sp_sem_label

        # supeproint instance label -> edge label
        sp_ins_label = instance_labels[spMask]
        sp_ins_label = stats.mode(sp_ins_label)[0][0]
        sp_instance[spId] = sp_ins_label

        # superpoint offset vector
        sp_center = np.mean(xyz[spMask], axis=0)
        _sp_offset_vector = instance_center[sp_ins_label] - sp_center
        assert _sp_offset_vector.shape == (3,)
        sp_offset_vector[spId] = _sp_offset_vector
    
    for s, t in edges:
        s_ins_label = sp_instance[s]
        t_ins_label = sp_instance[t]
        s_sem_label = sp_semantic[s]
        t_sem_label = sp_semantic[t]
        
        if (s_ins_label == -100 and t_ins_label == -100):
            is1ins.append((s_sem_label == t_sem_label))
        else:
            is1ins.append((s_ins_label == t_ins_label))
    

    assert len(edges) == len(is1ins)

    # print('computing edge features ... ')
    superpoints_features, edges_features = compute_edges_feature(xyz, superpoint, edges)

    G = igraph.Graph(n=spnum,
                    edges=edges,
                    directed=True,
                    edge_attrs={
                        'f': edges_features,
                        'is1ins': is1ins},
                    vertex_attrs={
                        'v': list(range(spnum)),
                        'semantic_label': sp_semantic,
                        'instance_label': sp_instance,
                        'superpoint_feature': superpoints_features,
                        'superpoint_offset_vector': sp_offset_vector})

    if vis_path != None:
        show_graph(room_name, xyz, superpoint, edges)

    return G



def show_graph(scene_name, xyz, superpoint, edges):
    ######## show graph ########
    sp_center = dict() 
    for sp in np.unique(superpoint):
        sp_points = xyz[ superpoint == sp ]
        center = sp_points.mean(0)
        sp_center[sp] = center
    
    # print('graph info:')
    # print('superpoint num:', len(sp_center))
    # print('edge num:', len(edges))
    

    # python3.6  dict is ordered
    sp2ind = { sp:ind for ind, sp in enumerate(sp_center.keys()) }

    vertex = np.array([tuple(i) for i in sp_center.values()], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    edges_vs = []
    # for (u, v), flag in zip(edges, is1ins):
    #     if flag==True:
    #         edges_vs.append((sp2ind[u], sp2ind[v], 0, 0, 255))
    #     else:
    #         edges_vs.append((sp2ind[u], sp2ind[v], 255, 0, 0))
    for u, v in edges:
        edges_vs.append((sp2ind[u], sp2ind[v]))
    

    # edge = np.array(edges_vs , 
    #         dtype=[('vertex1', 'i4'), ('vertex2', 'i4'), ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    edge = np.array(edges_vs , 
            dtype=[('vertex1', 'i4'), ('vertex2', 'i4')])
    
    
    d1 = PlyElement.describe(vertex, 'vertex')
    d2 = PlyElement.describe(edge, 'edge')
    plydata = PlyData([d1, d2])
    plydata.write(os.path.join(vis_path, scene_name+'_graph.ply'))


def compute_edges_feature(xyz, superpoint, edges):
    sp_num = len(np.unique(superpoint))
    edge_num = len(edges)

    sp_centroids = np.zeros((sp_num, 3), dtype='float32')
    sp_length = np.zeros((sp_num, 1), dtype='float32')
    sp_surface = np.zeros((sp_num, 1), dtype='float32')
    sp_volume = np.zeros((sp_num, 1), dtype='float32')
    sp_point_count = np.zeros((sp_num, 1), dtype='uint64')

    edge_delta_mean = np.zeros((edge_num, 3), dtype='float32')
    edge_delta_std = np.zeros((edge_num, 3), dtype='float32')
    edge_delta_norm = np.zeros((edge_num, 1), dtype='float32')
    edge_delta_centorid = np.zeros((edge_num, 3), dtype='float32')
    edge_length_ratio = np.zeros((edge_num, 1), dtype='float32')
    edge_surface_ratio = np.zeros((edge_num, 1), dtype='float32')
    edge_volume_ratio = np.zeros((edge_num, 1), dtype='float32')
    edge_point_count_ratio = np.zeros((edge_num, 1), dtype='float32')

    # --------- compute the superpoint features ----------
    sp_xyz_dict = dict()
    for spID in np.unique(superpoint): # np.unique wiil sort superpoint ID, superpoint ID should increase in order 
        spMask = np.where(superpoint == spID)[0]
        xyz_sp = xyz[spMask]
        sp_xyz_dict[spID] = xyz_sp
        sp_point_count[spID] = len(xyz_sp)
        if len(xyz_sp) == 1:
            sp_centroids[spID] = xyz_sp
            sp_length[spID] = 0
            sp_surface[spID] = 0
            sp_volume[spID] = 0
        elif len(xyz_sp) == 2:
            sp_centroids[spID] = np.mean(xyz_sp, axis=0)
            sp_length[spID] = np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
            sp_surface[spID] = 0
            sp_volume[spID] = 0
        else:
            ev = LA.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
            ev = -np.sort(-ev[0]) #descending order
            sp_centroids[spID] = np.mean(xyz_sp, axis=0)

            try:
                sp_length[spID] = ev[0]
            except TypeError:
                sp_length[spID] = 0

            try:
                sp_surface[spID] = np.sqrt(ev[0] * ev[1] + 1e-10)
            except TypeError:
                sp_surface[spID] = 0

            try:
                sp_volume[spID] = np.sqrt(ev[0] * ev[1] * ev[2] + 1e-10)
            except TypeError:
                sp_volume[spID] = 0

    # --------- compute the superpoint edges features -----------
    for ei, (s, t) in enumerate(edges):
        edge_delta_centorid[ei] = sp_centroids[s] - sp_centroids[t]
        edge_length_ratio[ei] = sp_length[s] / (sp_length[t] + 1e-6)
        edge_surface_ratio[ei] = sp_surface[s] / (sp_surface[t] + 1e-6)
        edge_volume_ratio[ei] = sp_volume[s] / (sp_volume[t] + 1e-6)
        edge_point_count_ratio[ei] = sp_point_count[s] / (sp_point_count[t] + 1e-6)

        xyz_source_sp, xyz_target_sp = sp_xyz_dict[s], sp_xyz_dict[t]

        """
        xyz_source_sp and xyz_target_sp have the same number of points
        """
        if len(xyz_source_sp) > len(xyz_target_sp):
            xyz_source_sp = xyz_source_sp[np.random.choice(len(xyz_source_sp), len(xyz_target_sp), replace=False)] # 采样至和 xyz_target 相同点数
        elif len(xyz_source_sp) < len(xyz_target_sp):
            xyz_target_sp = xyz_target_sp[np.random.choice(len(xyz_target_sp), len(xyz_source_sp), replace=False)]

        delta = xyz_source_sp - xyz_target_sp
        if len(delta) > 1:
            edge_delta_mean[ei] = np.mean(delta, axis=0)
            edge_delta_std[ei] = np.std(delta, axis=0)
            edge_delta_norm[ei] = np.mean(np.sqrt(np.sum(delta ** 2, axis=1)))
        else:
            edge_delta_mean[ei] = delta
            edge_delta_std[ei] = np.array([0, 0, 0])
            edge_delta_norm[ei] = np.sqrt(np.sum(delta ** 2))
    

    edges_features = np.concatenate([edge_delta_mean, edge_delta_std, edge_delta_centorid, edge_length_ratio, edge_surface_ratio,
                                     edge_volume_ratio, edge_point_count_ratio], axis=1) # 3+3+3+1+1+1+1

    superpoints_features = np.concatenate([sp_centroids, sp_length, sp_surface, sp_volume, sp_point_count], axis=1)

    return superpoints_features, edges_features




def show_instance(xyz, instance_labels, save_path, room_name):
    
    instance_num = len(np.unique(instance_labels))

    instance_color_table = np.random.randint(low=0, high=255, size=(instance_num, 3))
    color = np.zeros((len(xyz), 3))

    for i, ins_id in enumerate(np.unique(instance_labels)):
        if ins_id == -100:
            continue

        ins_mask = (instance_labels == ins_id)
        color[ins_mask] = instance_color_table[i]
    
    xyz_rgb = np.concatenate((xyz, color), axis=1)
    vertex = np.array([tuple(i) for i in xyz_rgb], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    d = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([d])
    plydata.write(os.path.join(save_path, room_name+f'_ins_gt.ply')) 






if __name__ == "__main__":

    
    global data_root, data_save_path, vis_path

    parser = argparse.ArgumentParser(description="S3DIS data prepare")
    parser.add_argument("--data_root", type=str, required=True, help="S3DIS data path")
    parser.add_argument("--save_dir", type=str, required=True, help="save data")
    parser.add_argument("--vis_dir", type=str, default=None, help="visual path, give to save visualization")
    args = parser.parse_args()

    # Please set your path here
    data_root = args.data_root 
    data_save_path = args.save_dir
    vis_path = args.vis_dir


    if vis_path != None and not os.path.isdir(vis_path):
        os.mkdir(vis_path)


    # !!! save data at $S3DIS_DATA/data
    data_save_path = os.path.join(data_save_path, 'data')

    if not os.path.isdir(data_save_path):
        os.mkdir(data_save_path)

    
    all_room_list = list(glob.glob(os.path.join(data_root, '*.npy')))

    assert len(all_room_list) == 272


    # # multi-thread processing
    p = mp.Pool(processes=mp.cpu_count())
    p.map(proc_data, all_room_list)
    p.close()
    p.join()

    # single thread  # for debug
    # for room in all_room_list:
    #     proc_data(room)

