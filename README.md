# Deep Structured Energy-Based Image Inpainting
Fazil Altinel, Mete Ozay, Takayuki Okatani  -  [http://www.vision.is.tohoku.ac.jp/](http://www.vision.is.tohoku.ac.jp/us/home/)

![](/files/readmeImages/iterations.gif)

```
@article{altinel2018dsebii,
 title={Deep Structured Energy-Based Image Inpainting},
 author={Fazil Altinel and Mete Ozay and Takayuki Okatani},
 journal={arXiv preprint arXiv:1801.07939},
 year={2018}
}
```

## Overview
This repository contains TensorFlow implementation of "Deep Structured Energy-Based Image Inpainting" paper.

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
$ python inpaint.py --allTest True --allImagesPath /path/to/all/dataset/folder/
```

Run the command below for testing images under `files/imgs/`:
```
$ python inpaint.py --allTest False --testImagesPath files/imgs/
```

Result images will be located under `files/results/`.

## License
The source code is licensed under [GNU General Public License v3.0](./LICENSE).
