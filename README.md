# NVIDIA AICITY CHALLENGE 2021 - Track 4

## Good Practices and A Strong Baseline for Traffic Anomaly Detection

This repository contains a customized source code of Track 4 in the NVIDIA AI City Challenge at CVPR 2021.

### Introduction

The Track 4 of NVIDIA AI CITY 2021 comprised of detecting anomalies based on different video feeds available from multiple cameras at intersections and along highways. 

The paper of this code can be found in [arxiv](https://arxiv.org/abs/2105.03827)

Detailed information of NVIDIA AICity Challenge 2021 can be found [here](https://www.aicitychallenge.org/).


Overview of the architecture of our anomaly detection framework, which consists of three main pipelines.

### Requirements

1. Python 3.7
2. To run the model from scratch, please request data from the organizers.
3. Install Yolo-v4 from [here](https://github.com/AlexeyAB/darknet) and use pretrained model on MS-COCO.

To run model from scratch, please follow these steps:

### Pipeline 1
![](assets/pipeline.png)


-- Pre Processing
1. Run `python pre_processing/stabilize_video.py` to apply DVS. 
2. Run `python pre_processing/background_modeling.py` to segment the processed images. 
3. Run pretrained Yolo v3 model on the background images folder and save it as `result.json`.
 

### Pipeline 2

1. Run dynamic track module.

### Pipeline 3

1. Run Post Process module.