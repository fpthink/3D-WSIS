# SPG

This code is modified from [superpoint_graph](https://github.com/loicland/superpoint_graph)

Modified content :
```
partition/ply_c/ply_c.cpp
partition/partition_S3DIS.py
```

In order to obtain point-level superpoint, we modify partition/ply_c/ply_c.cpp to additionally return point-to-voxel map. 

# Installation

Please refer to [superpoint_graph](https://github.com/loicland/superpoint_graph) for environment installation.


Compile the libply_c and libcp libraries:
```
CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION
cd partition/ply_c
cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
cd ..
cd cut-pursuit
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```

# S3DIS Download

Download [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html) and extract `Stanford3dDataset_v1.2.zip` or `Stanford3dDataset_v1.2_Aligned_Version.zip` to `$S3DIS_DIR`, where `$S3DIS_DIR` is set to dataset directory.

To fix some issues with `Stanford3dDataset_v1.2_Aligned_Version.zip` as reported in SPG issue [#29](https://github.com/loicland/superpoint_graph/issues/29), apply path `S3DIS_fix.diff` with:
```
cp S3DIS_fix.diff $S3DIS_DIR; cd $S3DIS_DIR; git apply S3DIS_fix.diff; rm S3DIS_fix.diff; cd -
```



$S3DIS_DIR directory :
```
$S3DIS_DIR
  |-- Area_1
  |-- Area_2
  |-- Area_3
  |-- Area_4
  |-- Area_5
  |-- Area_6
```

# SPG Superpoint Generation

To obtain point-level SPG superpoint run:
```
python partition/partition_S3DIS.py --data_root $S3DIS_DIR --save_dir $SP_DIR --vis_dir $VIS_DIR
```

`$SP_DIR` is the directory used to save SPG superpoint data.

`$VIS_DIR` is the directory used to save SPG superpoint visualization. Use [MeshLab](https://github.com/cnr-isti-vclab/meshlab) to view the visualization file `.ply`.

# Data Preparation

1\) prepare S3DIS data for network training :

`$S3DIS_DATA` is set to the directory where you want to save the processed data.
```
python prepare_S3DIS_inst_data.py --data_root $SP_DIR --save_dir $S3DIS_DATA --vis_dir $VIS_DIR
```

2\) prepare the `.txt` instance ground-truth files as the following:

```
python prepare_data_inst_gttxt.py --data_dir $S3DIS_DATA/data --save_dir $S3DIS_DATA/labels
```

After running such command, the structure of `$S3DIS_DATA` directory is as following :
```
$S3DIS_DATA
    |-- data
    |   |-- Area_1_WC_1.pth
    |   |-- Area_1_WC_1_spg.dat
    |   |-- Area_1_conferenceRoom_1.pth
    |   |-- Area_1_conferenceRoom_1_spg.dat
    |   |-- ...
    `-- labels
        |-- Area_1_WC_1.txt
        |-- Area_1_conferenceRoom_1.txt
        |-- ...
```