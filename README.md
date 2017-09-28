# kaggle-carvana
Code for the competition Carvana on Kaggle

## System Requirement

Not all requirement are needed, but python should be 2.x and tensorflow should be 1.0+. Also, the scripts in this repository may not work properly on windows.

> Ubuntu 16.04  
> Nvidia CUDA 8.0 with driver version 378.13  
> Python 2.7  
> Tensorflow 1.2  
> Sklearn 0.18.1  
> Scikit-image 0.12.3

## data conversion

There are some conventions about where we put the original files. (If the quota is not enough, use a soft link.)

> Images:  train-images/\*.jpg  
> Labels:  train-masks/\*.gif  
> Test:    test-images/\*.jpg  

There are also some conventions about where the tfrecord files are put to. Make sure you have the writing permissions.

> Train:   train-data/data\*.tfrecord  
> Test:    test-data/data\*.tfrecord  

RUN this script to generate tfrecords files.

    bash convert-image-into-tfrecords.sh 

