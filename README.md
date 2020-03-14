# graph-operations-search
This repo contains the source code for our CVPR'20 work: **Adaptive Interaction Modeling via Graph Operations Search**.

## Prerequisites
1. Pytorch 0.4.1. (If you use anaconda, you can create an environment with file [/environment/pytorch_0.4.1.yaml](/environment/pytorch_0.4.1.yaml).)
2. Compiled GPU-compatible ROI Align. (You could refer to [this repo](https://github.com/jwyang/faster-rcnn.pytorch) for the guidance of installation.)

## Data preparation
1. Download the [something-something-v1](https://20bn.com/datasets/something-something/v1) and [something-something-v2](https://20bn.com/datasets/something-something/v2), and extract frames for videos.
2. Extract region proposals for each frame using RPN. We format them in json files,
```
{"73": [[[0.435894, 0.549888, 0.800421, 0.99875], ...]], 
...}
```
in which the coordinates of the top-left and bottom-right vertices of the boxes are normalized to [0,1] for our convenience.

## Pretrain backbone
Before training the graph operations, we train the backbone backbone on target dataset. It is easy to modify our code to train the backbone. And we provide the pretrained checkpoint in [google drive](https://drive.google.com/open?id=19Hm89jj2Wk-iNSDcFoWv8HWxjMkysCZw). 

## Train and test
Modify the arguments in the script accordingly, and train or test with the command
```
cd experiments
bash run.sh
```

## Contact
If you have any problem please email me at lihaoxin05@gmail.com or lihx39@mail2.sysu.edu.cn. I may not look at the issues.