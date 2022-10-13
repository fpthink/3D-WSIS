# modify from PointGroup
# Written by Li Jiang
import os
import os.path as osp
import logging
from typing import Optional
from operator import itemgetter
from copy import deepcopy

import torch
import numpy as np
import open3d as o3d

import utils

COLORSEMANTIC = np.array([
    [171, 198, 230], # rgb(171, 198, 230)
    [143, 223, 142], # rgb(143, 223, 142)
    [0, 120, 177],   # rgb(0, 120, 177)
    [255, 188, 126], # rgb(255, 188, 126)
    [189, 189, 57],  # rgb(189, 189, 57)
    [144, 86, 76],   # rgb(144, 86, 76)
    [255, 152, 153], # rgb(255, 152, 153)
    [222, 40, 47],   # rgb(222, 40, 47)
    [197, 176, 212], # rgb(197, 176, 212)
    [150, 103, 185], # rgb(150, 103, 185)
    [200, 156, 149], # rgb(200, 156, 149)
    [0, 190, 206],   # rgb(0, 190, 206)
    [252, 183, 210], # rgb(252, 183, 210)
    [219, 219, 146], # rgb(219, 219, 146)
    [255, 127, 43],  # rgb(255, 127, 43)
    [234, 119, 192], # rgb(234, 119, 192)
    [150, 218, 228], # rgb(150, 218, 228)
    [0, 160, 55],    # rgb(0, 160, 55)
    [110, 128, 143], # rgb(110, 128, 143)
    [80, 83, 160]    # rgb(80, 83, 160)
    ])

COLOR20 = np.array([
    [230, 25, 75],   # rgb(230, 25, 75)
    [60, 180, 75],   # rgb(60, 180, 75)
    [255, 225, 25],  # rgb(255, 225, 25)
    [0, 130, 200],   # rgb(0, 130, 200)
    [245, 130, 48],  # rgb(245, 130, 48)
    [145, 30, 180],  # rgb(145, 30, 180)
    [70, 240, 240],  # rgb(70, 240, 240)
    [240, 50, 230],  # rgb(240, 50, 230)
    [210, 245, 60],  # rgb(210, 245, 60)
    [250, 190, 190], # rgb(250, 190, 190)
    [0, 128, 128],   # rgb(0, 128, 128)
    [230, 190, 255], # rgb(230, 190, 255)
    [170, 110, 40],  # rgb(170, 110, 40)
    [255, 250, 200], # rgb(255, 250, 200)
    [128, 0, 0],     # rgb(128, 0, 0)
    [170, 255, 195], # rgb(170, 255, 195)
    [128, 128, 0],   # rgb(128, 128, 0)
    [255, 215, 180], # rgb(255, 215, 180)
    [0, 0, 128],     # rgb(0, 0, 128)
    [128, 128, 128]  # rgb(128, 128, 128)
    ])


COLOR40 = np.array([
    [88,170,108],  # rgb(88,170,108)
    [174,105,226], # rgb(174,105,226)
    [78,194,83],   # rgb(78,194,83)
    [198,62,165],  # rgb(198,62,165)
    [133,188,52],  # rgb(133,188,52)
    [97,101,219],  # rgb(97,101,219)
    [190,177,52],  # rgb(190,177,52)
    [139,65,168],  # rgb(139,65,168)
    [75,202,137],  # rgb(75,202,137)
    [225,66,129],  # rgb(225,66,129)
    [68,135,42],   # rgb(68,135,42)
    [226,116,210], # rgb(226,116,210)
    [146,186,98],  # rgb(146,186,98)
    [68,105,201],  # rgb(68,105,201)
    [219,148,53],  # rgb(219,148,53)
    [85,142,235],  # rgb(85,142,235)
    [212,85,42],   # rgb(212,85,42)
    [78,176,223],  # rgb(78,176,223)
    [221,63,77],   # rgb(221,63,77)
    [68,195,195],  # rgb(68,195,195)
    [175,58,119],  # rgb(175,58,119)
    [81,175,144],  # rgb(81,175,144)
    [184,70,74],   # rgb(184,70,74)
    [40,116,79],   # rgb(40,116,79)
    [184,134,219], # rgb(184,134,219)
    [130,137,46],  # rgb(130,137,46)
    [110,89,164],  # rgb(110,89,164)
    [92,135,74],   # rgb(92,135,74)
    [220,140,190], # rgb(220,140,190)
    [94,103,39],   # rgb(94,103,39)
    [144,154,219], # rgb(144,154,219)
    [160,86,40],   # rgb(160,86,40)
    [67,107,165],  # rgb(67,107,165)
    [194,170,104], # rgb(194,170,10)
    [162,95,150],  # rgb(162,95,150)
    [143,110,44],  # rgb(143,110,44)
    [146,72,105],  # rgb(146,72,105)
    [225,142,106], # rgb(225,142,106)
    [162,83,86],   # rgb(162,83,86)
    [227,124,143]  # rgb(227,124,143)
    ])

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array([
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter",
    "desk", "curtain", "refridgerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"])
CLASS_COLOR = {
    "unannotated": [0, 0, 0],
    "floor": [143, 223, 142],
    "wall": [171, 198, 230],
    "cabinet": [0, 120, 177],
    "bed": [255, 188, 126],
    "chair": [189, 189, 57],
    "sofa": [144, 86, 76],
    "table": [255, 152, 153],
    "door": [222, 40, 47],
    "window": [197, 176, 212],
    "bookshelf": [150, 103, 185],
    "picture": [200, 156, 149],
    "counter": [0, 190, 206],
    "desk": [252, 183, 210],
    "curtain": [219, 219, 146],
    "refridgerator": [255, 127, 43],
    "bathtub": [234, 119, 192],
    "shower curtain": [150, 218, 228],
    "toilet": [0, 160, 55],
    "sink": [110, 128, 143],
    "otherfurniture": [80, 83, 160]}
SEMANTIC_IDX2NAME = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf", 11: "picture",
    12: "counter", 14: "desk", 16: "curtain", 24: "refridgerator", 28: "shower curtain", 33: "toilet",  34: "sink", 36: "bathtub", 39: "otherfurniture"}


def visualize_instance_mask(clusters: np.ndarray,
                            room_name: str,
                            visual_dir: str,
                            data_root: str,
                            cluster_scores: Optional[np.ndarray]=None,
                            semantic_pred: Optional[np.ndarray]=None,
                            color: int=20,
                            **kwargs):
    logger = utils.derive_logger(__name__)
    assert color in [20, 40]
    colors = globals()[f"COLOR{color}"]
    mesh_file = osp.join(data_root, room_name, room_name + "_vh_clean_2.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    pred_mesh = deepcopy(mesh)
    points = np.array(pred_mesh.vertices)
    inst_label_pred_rgb = np.zeros_like(points)  # np.ones(rgb.shape) * 255 #
    logger.info(f"room_name: {room_name}")
    for cluster_id, cluster in enumerate(clusters):
        if logger is not None:
            # NOTE: remove the handlers are not FileHandler to avoid 
            #       outputing this message on console(StreamHandler)
            #       and final will recover the handlers of logger
            handler_storage = []
            for handler in logger.handlers:
                if not isinstance(handler, logging.FileHandler):
                    handler_storage.append(handler)
                    logger.removeHandler(handler)
            message = f"{cluster_id:<4}: pointnum: {int(cluster.sum()):<7} "
            if semantic_pred is not None:
                semantic_label = np.argmax(np.bincount(semantic_pred[np.where(cluster == 1)[0]]))
                semantic_id = int(SEMANTIC_IDXS[semantic_label])
                semantic_name = SEMANTIC_IDX2NAME[semantic_id]
                message += f"semantic: {semantic_id:<3}-{semantic_name:<15} "
            if cluster_scores is not None:
                score = float(cluster_scores[cluster_id])
                message += f"score: {score:.4f} "
            logger.info(message)
            for handler in handler_storage:
                logger.addHandler(handler)
        inst_label_pred_rgb[cluster == 1] = colors[cluster_id % len(colors)]
    rgb = inst_label_pred_rgb

    pred_mesh.vertex_colors = o3d.utility.Vector3dVector(rgb / 255)
    points[:, 1] += (points[:, 1].max() + 0.5)
    pred_mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh += pred_mesh
    o3d.io.write_triangle_mesh(osp.join(visual_dir, room_name+".ply"), mesh)

# TODO: add the semantic visualization


def visualize_pts_rgb(rgb, room_name, data_root, output_dir, mode="test"):
    if "test" in mode:
        split = "scans_test"
    else:
        split = "scans"
    mesh_file = osp.join(data_root, split, room_name, room_name + "_vh_clean_2.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    pred_mesh = deepcopy(mesh)
    pred_mesh.vertex_colors = o3d.utility.Vector3dVector(rgb / 255)
    points = np.array(pred_mesh.vertices)
    # points[:, 2] += 3
    points[:, 1] += (points[:, 1].max() + 0.5)
    pred_mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh += pred_mesh
    o3d.io.write_triangle_mesh(osp.join(output_dir, room_name+".ply"), mesh)


def get_coords_color(data_root: str,
                     result_root: str,
                     room_split: str="train",
                     room_name: str="scene0000_00",
                     task: str="instance_pred"):
    input_file = os.path.join(data_root, room_split, room_name + "_inst_nostuff.pth")
    assert os.path.isfile(input_file), f"File not exist - {input_file}."
    if "test" in room_split:
        xyz, rgb, edges, scene_idx = torch.load(input_file)
    else:
        xyz, rgb, label, inst_label = torch.load(input_file)
    rgb = (rgb + 1) * 127.5

    if (task == "semantic_gt"):
        assert "test" not in room_split
        label = label.astype(np.int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif (task == "instance_gt"):
        assert "test" not in room_split
        inst_label = inst_label.astype(np.int)
        print(f"Instance number: {inst_label.max() + 1}")
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

    elif (task == "semantic_pred"):
        assert room_split != "train"
        semantic_file = os.path.join(result_root, room_split, "semantic", room_name + ".npy")
        assert os.path.isfile(semantic_file), f"No semantic result - {semantic_file}."
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif (task == "instance_pred"):
        assert room_split != "train"
        instance_file = os.path.join(result_root, room_split, room_name + ".txt")
        assert os.path.isfile(instance_file), f"No instance result - {instance_file}."
        f = open(instance_file, "r")
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #
        for i in range(len(masks) - 1, -1, -1):
            mask_path = os.path.join(result_root, room_split, masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if (float(masks[i][2]) < 0.09):
                continue
            mask = np.loadtxt(mask_path).astype(np.int)
            print(f"{i} {masks[i][2]}: {SEMANTIC_IDX2NAME[int(masks[i][1])]} pointnum: {mask.sum()}")
            inst_label_pred_rgb[mask == 1] = COLOR20[i % len(COLOR20)]
        rgb = inst_label_pred_rgb

    if "test" not in room_split:
        sem_valid = (label != -100)
        xyz = xyz[sem_valid]
        rgb = rgb[sem_valid]

    return xyz, rgb


def visualize_instance_mask_lite(clusters: np.ndarray,
                                 points: np.ndarray,
                                 visual_path: str,
                                 color: int=20,):
    assert color in [20, 40]
    colors = globals()[f"COLOR{color}"]
    inst_label_pred_rgb = np.zeros_like(points)  # np.ones(rgb.shape) * 255 #
    for cluster_id, cluster in enumerate(clusters):
        inst_label_pred_rgb[cluster == 1] = colors[cluster_id % len(colors)]
    rgb = inst_label_pred_rgb

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(rgb / 255)
    o3d.io.write_point_cloud(visual_path, pc)

