# Deep Structured Energy-Based Image Inpainting
Fazil Altinel, Mete Ozay, Takayuki Okatani  -  [http://www.vision.is.tohoku.ac.jp/](http://www.vision.is.tohoku.ac.jp/us/home/)

![](/files/readmeImages/iterations.gif)

## Overview
This repository contains TensorFlow implementation of Deep Structured Energy-Based Image Inpainting paper.

+ Network Architecture
```
Input(x) -> CONV1(KernelSize=8, NumFilter= 32, Stride=4) -> CONV2(KernelSize=4, NumFilter= 64, Stride=2) -> CONV3(KernelSize=3, NumFilter= 64, Stride=1) -> FC1(512)
                                                                                                             > Energy_x(y^)
Input(y^) -> CONV1(KernelSize=8, NumFilter= 32, Stride=4) -> CONV2(KernelSize=4, NumFilter= 64, Stride=2) -> CONV3(KernelSize=3, NumFilter= 64, Stride=1) -> FC1(512)
```
+ Learning rates that used during training.
```
For energy update: Learning rate = 0.01, momentum = 0.9.
For parameter update: Learning rate = 0.001.
```

## Files
```
files/
├── imgs/ - Test images folder.
├── model/ - Model files folder.
└── results/ - Test results folder.
inpaint.py - 
utils.py - 
```
Model File: http://vision.is.tohoku.ac.jp/~altinel/uploadFiles/celebA.tar.gz

## Dependencies
Tests are performed with following version of libraries:

+ Python 3.4
+ Numpy 1.11.3
+ TensorFlow 1.0.1

## Running

## License
The source code is licensed under [GNU General Public License v3.0](./LICENSE).

