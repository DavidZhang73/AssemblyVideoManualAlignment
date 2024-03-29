<div align="center">

# Aligning Step-by-Step Instructional Diagrams to Video Demonstrations

[![Pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Pytorch Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/pages/open-source/)
[![Pytorch Lightning Template](https://img.shields.io/badge/-Pytorch--Lightning--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/DavidZhang73/pytorch-lightning-template)
[![Conference](http://img.shields.io/badge/CVPR-2023-4b44ce.svg)](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Aligning_Step-by-Step_Instructional_Diagrams_to_Video_Demonstrations_CVPR_2023_paper.pdf)
[![ArXiv](http://img.shields.io/badge/ArXiv-2303.13800-B31B1B.svg)](https://arxiv.org/abs/2303.13800)
[![Project Website](http://img.shields.io/badge/Website-Project-61d4ff.svg)](https://academic.davidz.cn/en/publication/zhang-cvpr-2023/)
[![Dataset Website](http://img.shields.io/badge/Website-Dataset-61d4ff.svg)](https://iaw.davidz.cn)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

</div>

## Description

Official PyTorch implementation of CVPR 2023 Aligning Step-by-Step Instructional Diagrams to Video Demonstrations.

## How to run

**Data Preparation**

1. Download the dataset in JSON format from [here](https://iaw.davidz.cn).
2. Follow the instructions to download the data.
3. Resize the short side of both page and step images to 224px.
4. Use the script `script/gen_image_pickle.py` to generate the image pickle files.
5. Resize the short side of the videos to 224px.
6. Following the split files to split the video into 10-second long clips and store the frames in numpy format.

**Installation**

```bash
# clone project
git clone https://github.com/DavidZhang73/AssemblyVideoManualAlignment.git

# [Optional] create conda virtual environment
conda create -n <env_name> python=<3.8|3.9|3.10>
conda activate <env_name>

# [Optional] use mamba instead of conda
conda install mamba -n base -c conda-forge

# [Optional] install pytorch according to the official guide to support GPU acceleration, etc.
# https://pytorch.org/get-started/locally/

# install requirements
pip install -r requirements.txt
```

**Train**

```bash
python src/main.py fit -c configs/exp/ours.yaml -c configs/exp/{exp_name}.yaml --trainer.logger.name {log_name}
```

**Inference**

```bash
python src/main.py test -c configs/exp/ours.yaml -c configs/exp/{exp_name}.yaml --trainer.logger.name {log_name}
```

## Citation

```
@inproceedings{Zhang2023Aligning,
  author    = {Zhang, Jiahao and Cherian, Anoop and Liu, Yanbin and Ben-Shabat, Yizhak and Rodriguez, Cristian and Gould, Stephen},
  title     = {Aligning Step-by-Step Instructional Diagrams to Video Demonstrations},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
}
```
