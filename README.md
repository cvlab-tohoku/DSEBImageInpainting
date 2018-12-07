# Deep Structured Energy-Based Image Inpainting
Fazil Altinel, Mete Ozay, Takayuki Okatani  -  [http://www.vision.is.tohoku.ac.jp/](http://www.vision.is.tohoku.ac.jp/us/home/)

![](/files/readmeImages/iterations.gif)

If you make use of this code, please cite the following paper:
```
@INPROCEEDINGS{altinel2018dsebii, 
author={F. Altinel and M. Ozay and T. Okatani}, 
booktitle={2018 24th International Conference on Pattern Recognition (ICPR)}, 
title={Deep Structured Energy-Based Image Inpainting}, 
year={2018}, 
volume={}, 
number={}, 
pages={423-428},
doi={10.1109/ICPR.2018.8546025}, 
ISSN={1051-4651}, 
month={Aug},}
```

## Overview
This repository contains TensorFlow implementation of "[Deep Structured Energy-Based Image Inpainting](https://arxiv.org/abs/1801.07939)" paper (accepted to ICPR 2018).

![](/files/readmeImages/lfSonGithub.png)

+ Network Architecture:
```
Input(x)  -> CONV1(KernelSize=8, NumFilter= 32, Stride=4) -> CONV2(KernelSize=4, NumFilter= 64, Stride=2) -> CONV3(KernelSize=3, NumFilter= 64, Stride=1) -> FC1(512)
                                                                                                                                                                      > Energy_x(y^)
Input(y^) -> CONV1(KernelSize=8, NumFilter= 32, Stride=4) -> CONV2(KernelSize=4, NumFilter= 64, Stride=2) -> CONV3(KernelSize=3, NumFilter= 64, Stride=1) -> FC1(512)
```
+ Learning rates that used during training:
```
For energy update: Learning rate = 0.01, momentum = 0.9.
For parameter update: Learning rate = 0.001.
```

## Files
```
files/
├── imgs/ - Test images folder
├── model/ - Model files folder
└── results/ - Test results folder
inpaint.py - Loads the model file and generates inpainted image(s) for given image(s).
utils.py - Various utilities for 'inpaint.py'
```

## Dependencies
Tests are performed with following version of libraries:

+ Python 3.4
+ Numpy 1.11.3
+ TensorFlow 1.0.1

## Running
Download CelebA dataset (Align&Cropped Images): [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

Download the model file trained on CelebA dataset: http://vision.is.tohoku.ac.jp/~altinel/uploadFiles/celebA.tar.gz. Extract and locate the files under `files/model/`.

Run the command below for all testing set of CelebA dataset:
```
$ python inpaint.py --allTest 1 --allImagesPath /path/to/all/dataset/folder/
```

Run the command below for testing images under `files/imgs/`:
```
$ python inpaint.py --allTest 0 --allImagesPath /path/to/all/dataset/folder/ --testImagesPath files/imgs/
```

Result images will be located under `files/results/`.

## License
The source code is licensed under [GNU General Public License v3.0](./LICENSE).
