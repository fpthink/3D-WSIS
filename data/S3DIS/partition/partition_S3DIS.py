"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    Script for partioning into simples shapes
"""
import os.path
import sys
import numpy as np
import time
import argparse
from timeit import default_timer as timer
sys.path.append("./partition/cut-pursuit/build/src")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")

sys.path.append("./cut-pursuit/build/src")
sys.path.append("./ply_c")

import libcp
import libply_c
from graphs import *
from provider import *



def object_name_to_label(object_name):
    class_map = {
        "ceiling" : 0, 
        "floor" : 1, 
        "wall" : 2, 
        "beam" : 3, 
        "column" : 4, 
        "window" : 5, 
        "door" : 6,
        "table" : 7, 
        "chair" : 8, 
        "sofa" : 9, 
        "bookcase" : 10, 
        "board" : 11, 
        "clutter" : 12,
    }

    return class_map.get(object_name, 12) # stairs --> clutter

def read_room_data(raw_path):

    objects = glob.glob(os.path.join(raw_path, "Annotations/*.txt"))
    objects = sorted(objects) 

    points_list = []

    i_object = 0

    for single_object in objects:
        object_name = os.path.splitext(os.path.basename(single_object))[0]
        print("        adding object " + str(i_object) + " : "  + object_name)
        object_class = object_name.split('_')[0]
        object_label = object_name_to_label(object_class)
        
        points = np.loadtxt(single_object)
        semantic_labels = np.ones((points.shape[0], 1)) * object_label
        instance_labels = np.ones((points.shape[0], 1)) * i_object

        data_label = np.concatenate([points, semantic_labels, instance_labels], 1)
        points_list.append(data_label)

        i_object = i_object + 1

    data_label = np.concatenate(points_list, 0) # N*8 = XYZRGBLI
    xyz, rgb, semantic_label, instance_label = data_label[:, :3], data_label[:, 3:6], data_label[:, 6], data_label[:, 7]

    xyz = xyz.astype('float32')
    rgb = rgb.astype('uint8')
    semantic_label = semantic_label.astype('uint8')
    instance_label = instance_label.astype('uint32')

    return xyz, rgb, semantic_label, instance_label


def generate_SPG_superpoint(xyz, rgb, labels):

    """
    https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/S3DIS.md

    --voxel_width 0.03 --reg_strength 0.03
    """

    # param 
    voxel_width = 0.03
    reg_strength = 0.03
    lambda_edge_weight = 1.
    k_nn_adj = 10
    k_nn_geof = 45
    n_labels = 13

    # voxel-level
    xyz, rgb, labels, dump, p2v_map = libply_c.prune(xyz.astype('f4'), voxel_width, rgb.astype('uint8'), labels.astype('uint8'), np.zeros(1, dtype='uint8'), n_labels, 0)

    #---compute 10 nn graph-------
    graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
    #---compute geometric features-------
    geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')

    features = np.hstack((geof, rgb/255.)).astype('float32')#add rgb as a feature for partitioning
    features[:,3] = 2. * features[:,3] #increase importance of verticality (heuristic)

    graph_nn["edge_weight"] = np.array(1. / ( lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
    print("        minimal partition...")
    components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                    , graph_nn["edge_weight"], reg_strength)
    
    point_level_superpoint = in_component[p2v_map]

    return point_level_superpoint


def show_superpoint(xyz, superpoint_id, scene_name, save_dir):

    sp_color = np.zeros(xyz.shape)

    superpoint_num = superpoint_id.max() + 1

    sp_color_table = np.random.randint(low=0, high=255, size=(superpoint_num, 3))

    sp_color = sp_color_table[superpoint_id]

    superpoint_xyz_rgb = np.concatenate((xyz, sp_color), axis=1)
    vertex = np.array([tuple(i) for i in superpoint_xyz_rgb], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    d = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([d])
    plydata.write(os.path.join(save_dir, scene_name+'.ply'))  # use mesh to see superpoint visulization               



def main():

    parser = argparse.ArgumentParser(description="S3DIS data prepare")
    parser.add_argument("--data_root", type=str, required=True, help="S3DIS data path")
    parser.add_argument("--save_dir", type=str, required=True, help="save SPG superpoint")
    parser.add_argument("--vis_dir", type=str, default=None, help="visual path, give to save superpoint visualization")
    args = parser.parse_args()

    data_root = args.data_root 
    print('data root: ', data_root)
    save_dir = args.save_dir
    print('save dir: ', save_dir)
    vis_dir = args.vis_dir
    print('vis dir: ', vis_dir)

    Areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]

    for area in Areas:

        area_dir = os.path.join(data_root, area)

        rooms = [os.path.join(area_dir, o) for o in os.listdir(area_dir) 
                if os.path.isdir(os.path.join(area_dir, o))]

        for i, room in enumerate(rooms):
            
            room_name = os.path.basename(room)
            save_path = os.path.join(save_dir, area + '_' + room_name + '.npy')

            if os.path.isfile(save_path):
                print('{} existing'.format(area + '_' + room_name + '.npy'))
                continue
            

            print('processing {} room ( {}/{} ): {}'.format(area, i+1, len(rooms), room_name))
            xyz, rgb, semantic_label, instance_label = read_room_data(room)
            point_level_superpoint = generate_SPG_superpoint(xyz, rgb, semantic_label)

            save_data = np.concatenate([xyz, rgb, semantic_label.reshape((-1, 1)), instance_label.reshape((-1, 1)), point_level_superpoint.reshape((-1, 1))], axis=1)

            print('save superpoint ==> ', save_path)
            np.save(save_path, save_data)

            if vis_dir != None:
                print('save superpoint visualization ==> ', vis_dir)
                show_superpoint(xyz, point_level_superpoint, area + '_' + room_name, vis_dir)
            
            print('----------------------------------------------------------------')



if __name__ == '__main__':

    main()