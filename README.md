# Fusing Local Similarities for Retrieval-based 3D Orientation Estimation of Unseen Objects
PyTorch implementation of Paper "Fusing Local Similarities for Retrieval-based 3D Orientation Estimation of Unseen Objects" (ECCV 2022)
* [[project page](https://sailor-z.github.io/projects/Unseen_Object_Pose.html)]
* [[paper](https://arxiv.org/abs/2203.08472)]

# Citation
```bibtex
If you find the code useful, please consider citing:
@article{zhao2022fusing,
  title={Fusing Local Similarities for Retrieval-based 3D Orientation Estimation of Unseen Objects},
  author={Zhao, Chen and Hu, Yinlin and Salzmann, Mathieu},
  journal={arXiv preprint arXiv:2203.08472},
  year={2022}
}
```
# Setup
Our code has been tested with the the following dependencies: Python 3.7.11, Pytorch 1.7.1, Python-Blender 2.8, Pytorch3d 0.6.0, Python-OpenCV 3.4.2.17, Imutils 0.5.4. Please start by installing all the dependencies:

    conda create -n UnseenObjectPose python=3.7.11
    source activate PoseFromShape
    conda install -c conda-forge imutils
    conda install -c pytorch pytorch=1.7.1 torchvision
    conda install pytorch3d -c pytorch3d
    conda install -c jewfrocuban python-blender
    pip install opencv-python

# Data Processing
First please download the LineMOD dataset we used in our experiments from [LineMOD](https://u.pcloud.link/publink/show?code=XZrVD8VZCwypoMMPVA5QF0WeevE3SyyaeR07). The data should be organized as

    UnseenObjectPose
    |-- data
        |-- linemod_zhs
            |-- models
            |-- real

### Rendering
We generate 10,000 reference images for each object by rendering. The number of reference images and data path can be changed by modifying ```cfg["RENDER"]["NUM"]``` and ```cfg["RENDER"]["OUTPUT_PATH"]```, respectively. Please run the following code for rendering:

    cd ./Render
    python data_generation.py
