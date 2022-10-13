"""
Generate instance and semantic groundtruth .txt files (for evaluation)
"""

import argparse
import numpy as np
import glob
import torch
import os
import plyfile

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
    
    python prepare_data_inst_gttxt.py \
        --data_root /dataset/3d_datasets/scannetv2 \
        --data_split val \
        --data_root_processed /dataset/3d_datasets/3D_WSIS \
    """

    semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    semantic_label_names = ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"]

    args = get_parser()

    split_path = os.path.join(args.data_root_processed, args.data_split)
    files = sorted(glob.glob(f"{split_path}/scene*.pth"))
    rooms = [torch.load(i) for i in files]
    # exit()

    split_gt_path = os.path.join(args.data_root_processed, args.data_split + "_gt")
    if not os.path.exists(split_gt_path):
        os.mkdir(split_gt_path)

    # -----------------------------------------------------------------
    # instance segmentaiton
    print("begin to prepare instance groundtruth")
    for i, room in enumerate(rooms):
        xyz, rgb, label, instance_label, coords_shift, sample_idx = room     # label 0~19 -100;  instance_label 0~instance_num-1 -100
        scene_name = files[i].split("/")[-1][:12]
        print(f"{i + 1}/{len(rooms)} {scene_name}")

        save_path = os.path.join(split_gt_path, scene_name + "_ins.txt")
        if os.path.exists(save_path):
            continue

        instance_label_new = np.zeros(instance_label.shape, dtype=np.int32)  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_label.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_label == inst_id)[0]
            sem_id = int(label[instance_mask[0]])
            if(sem_id == -100): sem_id = 0
            semantic_label = semantic_label_idxs[sem_id]
            instance_label_new[instance_mask] = semantic_label * 1000 + inst_id + 1

        np.savetxt(save_path, instance_label_new, fmt="%d")
    print("instance groundtruth preparation finish")
    
    # -----------------------------------------------------------------
    # semantic segmentaiton
    print("begin to prepare semantic groundtruth")
    for i, f in enumerate(files):
        scene_name = f.split("/")[-1][:12]
        save_path = os.path.join(split_gt_path, scene_name + "_sem.txt")
        print(f"{i + 1}/{len(rooms)} {scene_name}")
        if os.path.exists(save_path):
            continue

        ply_file = os.path.join(args.data_root, "scans", scene_name, f"{scene_name}_vh_clean_2.labels.ply")
        ply = plyfile.PlyData().read(ply_file)
        label = np.array(ply.elements[0]["label"])

        np.savetxt(save_path, label, fmt="%d")
    print("semantic groundtruth preparation finish")






