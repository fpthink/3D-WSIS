"""
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
"""


import os
import os.path as osp
import json
import argparse
import multiprocessing as mp

import torch
import plyfile
import numpy as np
import open3d as o3d
import segmentator
import itertools
import igraph
from scipy import stats
from plyfile import PlyData, PlyElement
from torch_scatter import scatter
from sklearn.neighbors import KDTree

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from numpy import linalg as LA
import numpy.matlib
import collections
from sklearn import preprocessing


G_LABEL_NAMES = [
    "unannotated",
    "wall",
    "floor",
    "chair",
    "table",
    "desk",
    "bed",
    "bookshelf",
    "sofa",
    "sink",
    "bathtub",
    "toilet",
    "curtain",
    "counter",
    "door",
    "window",
    "shower curtain",
    "refridgerator",
    "picture",
    "cabinet",
    "otherfurniture"
]

def f_test(scene):
    fn = osp.join(data_root, f"scans_test/{scene}/{scene}_vh_clean_2.ply")
    print(fn)

    save_path = osp.join(split_save_path, scene + ".pth")
    if osp.exists(save_path):
        return

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
    faces = np.array([list(x) for x in f.elements[1]])[:, 0, :] # [nFaces, 3]
    faces = np.ascontiguousarray(faces)


    # generate superpoint
    mesh_file = fn
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    _vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    _faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(_vertices, _faces).numpy()

    semantic_labels = np.zeros(points.shape[0]) 
    instance_labels = np.zeros(points.shape[0])

    # superpoint graph 

    # save graph
    graph = build_weak_label_graph(scene, points[:, :3], faces, superpoint, semantic_labels, instance_labels)
    save_path_graph = osp.join(split_save_path, scene + "_spg.dat")
    print(f'save_path_graph => {save_path_graph}')
    graph.write_pickle(save_path_graph)

    # save data
    torch.save((coords, colors, semantic_labels, instance_labels, superpoint, scene), save_path)
    print("Saving to {}".format(save_path))



def f(scene):
    fn = osp.join(data_root, f"scans/{scene}/{scene}_vh_clean_2.ply")
    fn2 = osp.join(data_root, f"scans/{scene}/{scene}_vh_clean_2.labels.ply")
    fn3 = osp.join(data_root, f"scans/{scene}/{scene}_vh_clean_2.0.010000.segs.json")
    fn4 = osp.join(data_root, f"scans/{scene}/{scene}.aggregation.json")
    print(fn)

    save_path = osp.join(split_save_path, scene + ".pth")
    if osp.exists(save_path):
        return

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords_shift = -points[:, :3].mean(0)
    coords = np.ascontiguousarray(points[:, :3] + coords_shift)
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
    faces = np.array([list(x) for x in f.elements[1]])[:, 0, :] # [nFaces, 3]
    faces = np.ascontiguousarray(faces)

    f2 = plyfile.PlyData().read(fn2)
    semantic_labels = remapper[np.array(f2.elements[0]["label"])]

    with open(fn3) as jsondata:
        d = json.load(jsondata)
        seg = d["segIndices"]
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    instance_segids = []
    labels = []
    with open(fn4) as jsondata:
        d = json.load(jsondata)
        for x in d["segGroups"]:
            # if g_raw2scannetv2[x["label"]] != "wall" and g_raw2scannetv2[x["label"]] != "floor":
            instance_segids.append(x["segments"])  # Walls and floors also need to be spread
            labels.append(x["label"])
            assert(x["label"] in g_raw2scannetv2.keys())
    if(scene == "scene0217_00" and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]):
        instance_segids = instance_segids[: int(len(instance_segids) / 2)]
    check = []
    for i in range(len(instance_segids)): check += instance_segids[i]
    assert len(np.unique(check)) == len(check)

    instance_labels = np.ones(semantic_labels.shape[0]) * -100
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
        assert(len(np.unique(semantic_labels[pointids])) == 1)


    # obtain superpoint
    mesh_file = fn
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    _vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    _faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(_vertices, _faces).numpy()

    # save graph
    graph = build_weak_label_graph(scene, points[:, :3], faces, superpoint, semantic_labels, instance_labels)
    save_path_graph = osp.join(split_save_path, scene + "_spg.dat")
    print(f'save_path_graph => {save_path_graph}')
    graph.write_pickle(save_path_graph)

    # save data
    torch.save((coords, colors, semantic_labels, instance_labels, superpoint, scene), save_path)
    print("Saving to {}".format(save_path))




def build_weak_label_graph(scene, xyz, faces, superpoint, semantic_labels=None, instance_labels=None):
    print('building {} graph ...'.format(scene))

    # faces = faces.numpy()
    # superpoint = superpoint.numpy()
    spnum = len(np.unique(superpoint))

    # Make sure that superpoint ID is continuously incremented from 0
    assert spnum == (superpoint.max() + 1) 
    sp_semantic = np.ones(spnum) * -100
    sp_instance = np.ones(spnum) * -100
    sp_offset_vector = np.zeros((spnum, 3))

    # calculate the center of instance
    instance_center = dict()
    for insID in np.unique(instance_labels):
        instance_mask = (instance_labels == insID)
        instance_center[insID] = np.mean(xyz[instance_mask], axis=0)
    
    edges = set()      # save edge

    for face in faces:
        spId = superpoint[face]
        face_unique_spId = np.unique(spId)
        # if len(face_unique_spId) == 1:  # points in the superpoint have common spId 
        #     edges.add((face_unique_spId[0], face_unique_spId[0])) 
        if len(face_unique_spId) == 1:  # points in the superpoint have common spId 
            continue
        for a, b in itertools.combinations(face_unique_spId, 2):
            edges.add((a, b))
            edges.add((b, a))
    
    
    ############################

    assert len(xyz) == len(superpoint)
    superpoint_center = np.zeros((spnum, 3), dtype='float32')
    for spID in np.unique(superpoint):
        spMask = np.where(superpoint == spID)[0]
        superpoint_center[spID] = xyz[spMask].mean(0)

    tree = KDTree(superpoint_center)

    ind = tree.query_radius(superpoint_center, r=0.3, return_distance=False, count_only=False)
    
    for s, t_list in enumerate(ind):
        cnt = 0
        for t in t_list[1:]: # 0-th is myself
            if cnt >= 5:
                break
            if (s, t) not in edges:
                edges.add((s, t))
                edges.add((t, s))
                cnt += 1
            

    ############################

    
    edges = sorted(list(edges)) # [(a, b), (c, d)]
    
    if instance_labels is None:
        is1ins = [False] * len(edges)
    else:
        is1ins = []
        assert len(superpoint) == len(instance_labels)
        for spId in np.unique(superpoint):
            spMask = np.where(superpoint == spId)[0]

            # semantic label
            sp_sem_label = semantic_labels[spMask]
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
            is1ins.append((s_ins_label == t_ins_label))
        
    assert len(edges) == len(is1ins)

    superpoints_features, edges_features = compute_edges_feature(xyz, superpoint, edges)
    scaler = preprocessing.StandardScaler().fit(edges_features) 
    scaler.transform(edges_features, copy=False)

    G = igraph.Graph(n=spnum, edges=edges, directed=True,
                        edge_attrs={'f': edges_features, 'is1ins': is1ins},
                        vertex_attrs={'v': list(range(spnum)), 'semantic_label': sp_semantic, 'instance_label': sp_instance,
                                        'superpoint_feature':superpoints_features, 'superpoint_offset_vector':sp_offset_vector})
    
    for e in G.es:
        s, t = e.source, e.target
        if (G.vs[s]['instance_label'] == -100) or (G.vs[t]['instance_label'] == -100):
            e['is1ins'] = 0  
        elif (G.vs[s]['instance_label'] == G.vs[t]['instance_label']):
            e['is1ins'] = -1 
        elif (G.vs[s]['instance_label'] != G.vs[t]['instance_label']):
            e['is1ins'] = 1 
        else:
            raise Exception('unknown case!')


    return G

    # show_graph(scene, xyz, superpoint, edges)
    


def show_weak_label(scene_name, xyz, superpoint, choice_sp_list):
    color_table = np.random.randint(low=0, high=255, size=(len(choice_sp_list), 3))
    color = np.zeros((len(xyz), 3))

    for i, spId in enumerate(choice_sp_list):
        spMask = (superpoint == spId)
        color[spMask] = color_table[i]
    
    xyz_rgb = np.concatenate((xyz, color), axis=1)
    vertex = np.array([tuple(i) for i in xyz_rgb], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    d = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([d])
    save_path = r'' # !
    plydata.write(os.path.join(save_path, scene_name+'_weak_label.ply'))

def show_graph(scene_name, xyz, superpoint, edges):
    ######## show graph ########
    sp_center = dict() 
    for sp in np.unique(superpoint):
        sp_points = xyz[ superpoint == sp ]
        center = sp_points.mean(0)
        sp_center[sp] = center
    
    print('graph info:')
    print('superpoint num:', len(sp_center))
    print('edge num:', len(edges))

    # dict is ordered
    sp2ind = { sp:ind for ind, sp in enumerate(sp_center.keys()) }

    vertex = np.array([tuple(i) for i in sp_center.values()], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    edges_vs = []

    for u, v in edges:
        edges_vs.append((sp2ind[u], sp2ind[v]))
    

    edge = np.array(edges_vs , 
            dtype=[('vertex1', 'i4'), ('vertex2', 'i4')])
    
    
    d1 = PlyElement.describe(vertex, 'vertex')
    d2 = PlyElement.describe(edge, 'edge')
    plydata = PlyData([d1, d2])
    save_path = r''  # set save path
    plydata.write(os.path.join(save_path, scene_name+'_dense_graph.ply'))
    print("Saving to {}".format(save_path))

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
    for spID in np.unique(superpoint): # np.unique will automatically sort
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
        Sampling to the same number of points
        """
        if len(xyz_source_sp) > len(xyz_target_sp):
            xyz_source_sp = xyz_source_sp[np.random.choice(len(xyz_source_sp), len(xyz_target_sp), replace=False)] 
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
    # superpoint feature dimension  3+1+1+1+1

    return superpoints_features, edges_features



def get_parser():
    parser = argparse.ArgumentParser(description="ScanNet data prepare")
    parser.add_argument("--data_root", required=True, help="scannetv2 data path")
    parser.add_argument("--data_split", required=True, help="data split (train / val / test)")
    parser.add_argument("--data_root_processed", required=True, help="processed scannetv2 data path")
    args_cfg = parser.parse_args()
    return args_cfg


if __name__ == "__main__":

    """
    For example:

    python prepare_data_inst.py \
        --data_root /dataset/3d_datasets/scannetv2 \
        --data_split train \
        --data_root_processed /dataset/3d_datasets/3D_WSIS \

    python prepare_data_inst.py \
        --data_root /dataset/3d_datasets/scannetv2 \
        --data_split val \
        --data_root_processed /dataset/3d_datasets/3D_WSIS \

    python prepare_data_inst.py \
        --data_root /dataset/3d_datasets/scannetv2 \
        --data_split test \
        --data_root_processed /dataset/3d_datasets/3D_WSIS \
    """

    
    args = get_parser()

    print('data_root: {}'.format(args.data_root))
    print('data_split: {}'.format(args.data_split))
    print('data_root_processed: {}'.format(args.data_root_processed))
    
    os.makedirs(args.data_root_processed, exist_ok=True)
    
    global data_root
    data_root = args.data_root


    meta_data_dir = osp.join(args.data_root, "meta_data")

    def get_raw2scannetv2_label_map():
        lines = [line.rstrip() for line in open(osp.join(meta_data_dir, "scannetv2-labels.combined.tsv"))]
        lines_0 = lines[0].split("\t")
        # print(lines_0)
        # print(len(lines))
        lines = lines[1:]
        raw2scannet = {}
        for i in range(len(lines)):
            label_classes_set = set(G_LABEL_NAMES)
            elements = lines[i].split("\t")
            raw_name = elements[1]
            # if (elements[1] != elements[2]):
                # print(f"{i}: {elements[1]} {elements[2]}")
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2scannet[raw_name] = "unannotated"
            else:
                raw2scannet[raw_name] = nyu40_name
        return raw2scannet

    g_raw2scannetv2 = get_raw2scannetv2_label_map()


    # Map relevant classes to {0,1,...,19}, and ignored classes to -100
    remapper = np.ones(150) * (-100)
    for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
        remapper[x] = i

    global split_save_path
    split_save_path = osp.join(args.data_root_processed, args.data_split)
    if osp.exists(split_save_path):
        print('{} exist!'.format(split_save_path))
    else:
        os.makedirs(split_save_path, exist_ok=True)
        print("create {} folder".format(split_save_path))

    scene_list = []
    file_path = osp.join(meta_data_dir, "scannetv2_{}.txt".format(args.data_split))
    fin = open(file_path, "r")
    for line in fin:
        line = line.strip()
        scene_list.append(line)
    
    assert len(scene_list) == 1201 or len(scene_list) == 312 or len(scene_list) == 100


    files = sorted(list(map(lambda x: f"scans/{x}/{x}_vh_clean_2.ply", scene_list)))
    assert len(files) == len(scene_list)

    if args.data_split != "test":
        files2 = sorted(list(map(lambda x: f"scans/{x}/{x}_vh_clean_2.labels.ply", scene_list)))
        files3 = sorted(list(map(lambda x: f"scans/{x}/{x}_vh_clean_2.0.010000.segs.json", scene_list)))
        files4 = sorted(list(map(lambda x: f"scans/{x}/{x}.aggregation.json", scene_list)))
        assert len(files) == len(files2)
        assert len(files) == len(files3)
        assert len(files) == len(files4), f"{len(files)} {len(files4)}"



    p = mp.Pool(processes=mp.cpu_count())
    if args.data_split == "test":
        p.map(f_test, scene_list)
    else:
        p.map(f, scene_list)
    p.close()
    p.join()

    