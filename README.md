# Learning Inter-Superpoint Affinity for Weakly Supervised 3D Instance Segmentation

This is the official PyTorch implementation of the papers :

**Learning Inter-Superpoint Affinity for Weakly Supervised 3D Instance Segmentation (ACCV 2022)**  [[arxiv]](https://arxiv.org/abs/2210.05534)

by Linghua Tang, Le Hui, and Jin Xie. 

## Installation

1\) Requirements
* Python 3.7.0
* Pytorch 1.7.1
* CUDA 10.1
* GPU NVIDIA TITAN RTX

2\) Anaconda Virtual Environment
```
conda create -n 3DWSIS python=3.7
conda activate 3DWSIS
```

3\) Clone the repository.
```
git clone https://github.com/fpthink/3D-WSIS.git --recursive
```

4\) Install the requirements.
```
cd 3DWSIS
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

5\) Install `spconv` and `pointgroup_ops ` 

[[spconv]](https://github.com/llijiang/spconv) [[pointgroup_ops]](https://github.com/dvlab-research/PointGroup/tree/master/lib/pointgroup_ops)

Please refer to [PointGroup](https://github.com/dvlab-research/PointGroup) to install.


## ScanNet v2

### Data Preparation 

Please refer to the `ScanNetV2.md` in `data/ScanNetV2` to process data.

### Training

Please set `$ScanNetV2_DATA` on `Line 29` of `config/ScanNet_v2_3D_WSIS.yaml`.

```
CUDA_VISIBLE_DEVICES=0 python train_scannetv2.py --config config/ScanNet_v2_3D_WSIS.yaml
```

### Evaluation

```
CUDA_VISIBLE_DEVICES=0 python test_scannetv2.py --config config/ScanNet_v2_3D_WSIS.yaml --pretrain log/ScanNet_v2_3D_WSIS/epoch_00120_whole_scene.pth
```

## S3DIS

### Data Preparation

Please refer to the `S3DIS.md` in `data/S3DIS` to process data.

### Training

Please set `$S3DIS_DATA/data` on `Line 29` of `config/S3DIS_Area5_3D_WSIS.yaml`.

```
CUDA_VISIBLE_DEVICES=0 python train_s3dis.py --config config/S3DIS_Area5_3D_WSIS.yaml
```

```
CUDA_VISIBLE_DEVICES=0 python test_s3dis.py --config config/S3DIS_Area5_3D_WSIS.yaml --pretrain log/S3DIS_Area5_3D_WSIS/epoch_00300_whole_scene.pth
```

## Pretrained Model

### ScanNet v2 validation :
[[Baidu Cloud]](https://pan.baidu.com/s/1F-LP-2nozqZqfLQjbxn63g?pwd=jsj3) [[Google Dirve]](https://drive.google.com/drive/folders/10wS-yfrP6xfxnKzAFOdKOL4MBgENkQup?usp=sharing)

Its performance on ScanNet-v2 validation set is 29.8/48.4/67.7 in terms of mAP/mAP50/mAP25. 

### S3DIS Area5 :

[[Baidu Cloud]](https://pan.baidu.com/s/1EcEl2dA8Dk8qipOk9QvlVA?pwd=tog9) [[Google Dirve]](https://drive.google.com/drive/folders/1T66iuECxOUYKgX4Axf7nl_rpFIYmG-Z9?usp=sharing)

Its performance on S3DIS Area5 set is 22.4/35.2/47.2/43.2/44.7/51.8/41.3 in terms of mAP/mAP50/mAP25/mCov/mWCov/mPrec/mRec.

> Note :  Due to the randomness of weak label generation, the results of network training fluctuate slightly. 

## Acknowledgement
This repo is built upon several repos, e.g., [PointGroup](https://github.com/dvlab-research/PointGroupt), [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet), [HAIS](https://github.com/hustvl/HAIS), [spconv](https://github.com/traveller59/spconv) and [ScanNet](https://github.com/ScanNet/ScanNet).

## TODO
- [X] release S3DIS dataset


## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{tang20223dwsis,
    author    = {Tang, Linghua and Hui, Le and Xie, Jin},
    title     = {Learning Inter-Superpoint Affinity for Weakly Supervised 3D Instance Segmentation},
    booktitle = {ACCV},
    year      = {2022},
}
@inproceedings{hui2022graphcut,
    author    = {Hui, Le and Tang, Linghua and Shen, Yaqi and Xie, Jin and Yang, Jian},
    title     = {Learning Superpoint Graph Cut for 3D Instance Segmentation},
    booktitle = {NeurIPS},
    year      = {2022},
}
```

