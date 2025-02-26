# DSI4PR
DSI4PR: Dual Stream Interaction for High-Precision Image to PointCloud Place Recognition

Hardware configuration
an Intel i7 13700K CPU and an Nvidia GeForce
RTX4090 SUPRIM X GPU. 

Setting up the environment
This code is tested on Ubuntu 20.04.6 with Python 3.7.16 with torch 1.13.0+cu116 and CUDA 11.6 with following packages.
numpy                     1.21.6
opencv-python             4.10.0.84 
pillow                    9.5.0 
scipy                     1.7.3 
scikit-learn              1.0.2  
tensorboard               2.11.2  
timm                      0.9.2  
torchvision               0.14.0+cu116 
tqdm                      4.67.1   
albumentations            1.3.0  
pyyaml                    6.0.1 

Dataset
KITTI
Download from https://www.cvlibs.net/datasets/kitti/eval_odometry.php
KITTI360
Download from https://www.cvlibs.net/datasets/kitti-360/download.php.

For KITTI "00" training
python --expid=kitti
For KITTI360 "3,4,5,6,7,9,10" training
python --expid=kitti360

For KITTI evaluating Recall@1
python evaluate.py --expid kitti --eval_sequence 02 --threshold_dist 10

For KITTI360 evaluating Recall@1
python evaluate.py --expid kitti360 --eval_sequence 0000 --threshold_dist 10

For zero-shot transfer evaluating KITTI To KITTI360 Recall@1
python evaluate.py --expid kitti --eval_sequence 0000 --threshold_dist 10











## Acknowledgments

This project is developed based on the work from [Shubodh Sai's Project]. Special thanks to Shubodh Sai for their open-source contributions.




https://github.com/Shubodh/lidar-image-pretrain-VPR.git
