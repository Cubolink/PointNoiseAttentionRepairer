# Project Phoenix-CH: Pointcloud Healing and Object Estimation through cross-attentIon eXploration.

This repo started as a fork of [PointAttN](https://github.com/ohhhyeahhh/PointAttN). Please refer to their repo for the original implementation.
There are some implementations taken from [Convolutional Occupancy Networks](https://pengsongyou.github.io/conv_onet) too.
Please, consider visiting their original pages.

## 0. About this work


In this project, we explore and develop some AI models to restore incomplete point clouds of archaeological artifacts,
combining the strengths of implicit representations and point-cloud-based approaches.
We use [PointAttN](https://github.com/ohhhyeahhh/PointAttN), and two custom variations. 

### Motivation

Archaeology seeks to understand human history through ancient artifacts,
but these objects often suffer from deterioration or incompleteness.
Digital models obtained through scanning frequently exhibit defects, such as missing geometry due to occlusion.

To address these challenges, some researches have explored reconstruction using implicit representations
**([DeepMend](dsfs), [ConvONet](https://pengsongyou.github.io/conv_onet))**, although these methods often require watertight meshes. Alternatively,
**[PointAttN](https://github.com/ohhhyeahhh/PointAttN)** and **[DRDAP](https://github.com/ivansipiran/Data-driven-cultural-heritage)**
focused on directly working with point clouds to repair objects. Notably, **DRDAP** and **DeepMend**
specifically explored generating only the missing geometry rather than reconstructing the entire object,
something really useful for repairing archaeological objects.


## 1. Environment setup

### Install related libraries

This code has been tested on Ubuntu 20.04, python 3.10.14, torch 1.13.1 with cuda 11.7, and using CUDA 12.2 in the machine.
Please install related libraries before running this code, we have provided 4 ways of doing this:

#### A. Using conda yml files (not recommended)

You can, theoretically, setup the environment with
```
conda env create -f environments/environment.yml
```

I wasn't able to replicate the environment in other machines with this method, so just in case I exported other kind of .yml.

Conda files:
* `environments/environment.yml` was exported with conda, contains all the environment used to run the models.
* `environments/environment_from_history.yml` was exported with conda --from-history. It contains only the explicitly installed files, but the pip-installed libraries are missing.
* `environments/environment_no_builds.yml` was exported with conda --no-builds. It also doesn't include the pip-installed libraries.

I provided the `requirements.txt` using `pip freeze` too, but I'm pretty sure it won't ever work.

#### B. Installing all libraries (recommended)

Sadly the only method that worked for me to install in different machines was to manually re-create the environment using `conda` and `pip`. I have provided the exact commands that I used in `setup.sh`.
**DO NOT RUN THIS SCRIPT DIRECTLY**, you should run each line of the file, step by step, until you reach the end.

You can use this command to look at the file. Of course, it's better to just use a text editor.
```
cat setup.sh
```

### Compile Pytorch 3rd-party modules

please compile Pytorch 3rd-party modules [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch). A simple way is using the following command:

```
cd $PointAttN_Home/utils/ChamferDistancePytorch/chamfer3D
python setup.py install
```

If you followed the `setup.sh` step by step, you already compiled and installed this module.

## 2. Train

### Prepare training datasets

Aiming for repairing cultural heritage objects, we used this [Dataset for precolombian reconstruction](https://github.com/PJaramilloV/Precolombian-Dataset.git).
In this work, I called it **CHS** instead, standing for *Cultural Heritage Sharp*.

### Train a model

To train the PointAttN, PointAttNA or PointAttNB models, modify the dataset path in `cfgs/PointAttNX.yaml `, run:

```
python train.py -c PointAttNX.yaml
```

## 3. Test

### Pretrained models

You can use the models provided in this [zip file](https://anakena.dcc.uchile.cl/~jocruz/resources/pretrained_models.zip). These were models trained with chs.

### Test for paper result

To test any PointAttN variation on CHS benchmark, download  the pretrained model and put it into its `log/PointAttN_cd_debug_chs` directory. Ensure it matches the corresponding `cfgs/PointAttNX_chs.yaml`'s _`load_model`_ path, and then run:

```
python test_chs.py -c PointAttNX.yaml
```

## 4. Acknowledgement

1. We include the following PyTorch 3rd-party libraries:  
   [1] [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)

2. Some of the code of this project is borrowed from [VRC-Net](https://github.com/paul007pl/MVP_Benchmark)  

3. A lot of the code of this project is taken from [PointAttN](https://github.com/ohhhyeahhh/PointAttN), since this repo
stared as a fork.

4. There is also code from [Convolutional Occupancy Networks](https://pengsongyou.github.io/conv_onet) too.

## 5. Cite this work

If you use PointAttN in your work, please cite the original paper:

```
@article{Wang_Cui_Guo_Li_Liu_Shen_2024,
   title={PointAttN: You Only Need Attention for Point Cloud Completion},
   volume={38}, 
   url={https://ojs.aaai.org/index.php/AAAI/article/view/28356}, DOI={10.1609/aaai.v38i6.28356}, 
   number={6}, 
   journal={Proceedings of the AAAI Conference on Artificial Intelligence},
   author={Wang, Jun and Cui, Ying and Guo, Dongyan and Li, Junxia and Liu, Qingshan and Shen, Chunhua},
   year={2024},
   month={Mar.},
   pages={5472-5480}
}
```

![Infographic](Poster_2024-11.png "Poster for a Workshop in Computer Vision done in 2024-11")