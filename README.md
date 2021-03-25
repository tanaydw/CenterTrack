# CenterTrack, Perspective Mapping and Satellite Images

## Introduction

The goal of this project is to generate roadway images with just artifacts and not sending the entire image. **Visual-Inertial-Semantic Scene Representation** is the primary approach explored in the project. In this technique, a 3D object detector is used to estimate the size and distance of the artifact of our interest. This information is then sent to the cloud which can be used either for big data analytics such as traffic estimation or for storage. The stored information is usually retreived by another car B to recreate a scene, thereby reducing the cost associated with running the entire pipeline again. Therefore, the overall architecture is divided into two steps both of which are explained in details below.

#### Step 1:- Localization and Metadata Generation

The overall pipeline for the first step is shown in the figure below:-

<p align="center"> <img src='images/car_a_pipeline.PNG' align="center" height="230px"> </p> 

An image from camera is prerocessed and fed as input to a 3D object detection algorithm. For this project, we have used [CenterTrack](https://github.com/xingyizhou/CenterTrack) with a few modifications as to output the required metadata for our use case. The backbone of the network is [DCNv2](https://github.com/CharlesShang/DCNv2) and it has been tested to give better results than other standard backbones such as ResNet or VGG. The network outputs all the essential information, such as object category, 3D bounding box and box orientation, that is required to exactly localize the object in a space and links it with a particular GPS coordinate so that it can be traced back to this particular location by another car B or the cloud. The entire GPS → metadata mapping is stored as a dictionary and transferred to the cloud, which recreates and analyzes the data in real-time setting.

#### Step 2:- Recreating the Scene

<p align="center"> <img src='images/car_b_pipeline.PNG' align="center" height="230px"> </p>

The basic assumption for this step is that a GPS sensor and a local segmented satellite map is available on car B. First, a basic path prediction algorithm is implemented to estimate the direction in which the car is moving, which defines the orientation of the car. The information obtained from this step is used to extract the region of interest on a full scale satellite map that helps to concentrate at a particular location at which the car is present, rather than the entire city. Parallely, GPS-coordinate information is sent to the cloud for requesting the metadata. The cloud provides the metadata, which is then expanded and overlayed with proper scaling on the extracted satellite map to recreate the entire scene of the location.

## Installation

Please refer to [INSTALL.ipynb](INSTALL.ipynb) notebook for installation instructions.


## Inference

Please refer to [Inference.ipynb](Inference.ipynb) notebook for running the code on a video/image. A sample code is shown in the notebook for demo purpose as well.


## Training CenterTrack

CenterTrack is the primary 3D object detection algorithm that we have used in this project. The weights provided by us are pre-trained on NuScenes dataset on which we have done transfer learning using CARLA simulator for 10 more epochs. It can be further trained and fine-tuned as and when required. We refer you to original source of [CenterTrack](https://github.com/xingyizhou/CenterTrack) for training steps. Once the model is trained, the final weights can be used with this project for transferring metadata and recreating the scene.    
