# CVCP-Fusion: On Implicit Depth Estimation for 3D Bounding Box Prediction

3D Object Detection and Tracking using center points in the bird-eye view.

<p align="center"> <img src='docs/teaser.png' align="center" height="230px"> </p>

## Use CenterPoint

### Installation

Please refer to [INSTALL](docs/INSTALL.md) to set up libraries needed for distributed training and sparse convolution.

### Benchmark Evaluation and Training 

Please refer to [GETTING_START](docs/GETTING_START.md) to prepare the data. Then follow the instruction there to reproduce our detection and tracking results. All detection configurations are included in [configs](configs).

### Develop

If you are interested in training CenterPoint on a new dataset, use CenterPoint in a new task, or use a new network architecture for CenterPoint, please refer to [DEVELOP](docs/DEVELOP.md). Feel free to send us an email for discussions or suggestions. 

### ToDo List
- [ ] Support visualization with Open3D  
- [ ] Colab demo 
- [ ] Docker   

## License

CenterPoint is release under MIT license (see [LICENSE](LICENSE)). It is developed based on a forked version of [det3d](https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07). We also incorperate a large amount of code from [CenterNet](https://github.com/xingyizhou/CenterNet)
and [CenterTrack](https://github.com/xingyizhou/CenterTrack). See the [NOTICE](docs/NOTICE) for details. Note that both nuScenes and Waymo datasets are under non-commercial licenses. 

## Acknowlegement
This project is not possible without multiple great opensourced codebases. We list some notable examples below.  

* [det3d](https://github.com/poodarchu/det3d)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [CenterTrack](https://github.com/xingyizhou/CenterTrack)
* [CenterNet](https://github.com/xingyizhou/CenterNet) 
* [mmcv](https://github.com/open-mmlab/mmcv)
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
