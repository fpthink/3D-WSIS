# ScanNet-v2 Download
Download [ScanNet-v2 Dataset](http://www.scan-net.org/), where `$ScanNetV2_DIR` is set to original dataset directory.

`$ScanNetV2_DIR` directory :
```
$ScanNetV2_DIR
    |-- meta_data
    |   |-- scannetv2-labels.combined.tsv
    |   |-- scannetv2_test.txt
    |   |-- scannetv2_train.txt
    |   `-- scannetv2_val.txt
    |-- scans
    |   |-- scene0000_00
    |   |-- scene0000_01
    |   |-- scene0000_02
    |   `-- ...
    `-- scans_test
        |-- scene0707_00
        |-- scene0708_00
        |-- scene0709_00
        `-- ...
```

# Installation

Please refer to [here](https://github.com/Karbo123/segmentator) to install `segmentator`. 

`segmentator` is used for superpoint generation in ScanNet-v2.

# Data Preparation

1\) Prepare ScanNet-v2 data :

`$ScanNetV2_DATA` is set to the directory where you want to save the processed data.
```
python prepare_data_inst_ScanNetV2.py 
    --data_root $ScanNetV2_DIR 
    --data_split train 
    --data_root_processed $ScanNetV2_DATA

python prepare_data_inst_ScanNetV2.py 
    --data_root $ScanNetV2_DIR 
    --data_split val 
    --data_root_processed $ScanNetV2_DATA
```

2\) Prepare the `.txt` instance ground-truth files as the following :
```
python prepare_data_inst_gttxt.py 
        --data_root $ScanNetV2_DIR
        --data_split val 
        --data_root_processed $ScanNetV2_DATA
```

3\) After running such command, the structure of `$ScanNetV2_DATA` directory is as following:
```
$ScanNetV2_DATA
    |-- train
    |   |-- scene0000_00.pth
    |   |-- scene0000_00_spg.dat
    |   |-- scene0000_01.pth
    |   |-- scene0000_01_spg.dat
    |   `-- ...
    |-- val
    |   |-- scene0011_00.pth
    |   |-- scene0011_00_spg.dat
    |   |-- scene0011_01.pth
    |   |-- scene0011_01_spg.dat
    |   `-- ...
    `-- val_gt
        |-- scene0011_00.txt
        |-- scene0011_00_ins.txt
        |-- scene0011_00_sem.txt
        `-- ...
```

