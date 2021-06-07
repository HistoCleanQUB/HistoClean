Welcome to the HistoClean Gitbub repository!

HistoClean is an open source image processing tool for use in developing deep learning models. Here, we bring together the best image manipulation packages into one easy to use application.

This repository contains all relavent files relating to the paper "HistoClean: open-source software for histological image pre-processing and augmentation to improve development of robust convolutional neural networks"

WARNING: HistoClean is currenlty in pre-release and may contain many bugs.  It is recomended you make a copy of any datasets before applying the application.

## The latest binary release can be found [Here](https://github.com/HistoCleanQUB/HistoClean/releases)

Currently this application is only availble for Windows, but there are plans to port to MacOS and Linux in the coming weeks.

The current version (v0.1) consists of five modules:

1) Image patching - Divide large images into patches for use in convolutional neural networks or other computer vision tasks.  Based on the "Patchify" Python Package - https://pypi.org/project/patchify/


2) Dataset balancing - Balance an infinate number of image classes by applying random rotation and mirroring to existing images. Class balancing is essential to prevent bias when training deep learining models.

3) Whitespace Filtering - Set a minimum histological tissue (foreground) threshold for images. Allows for the quick and easy removal on non-informative images. Based on the openCV library - https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

![Whitespace](https://user-images.githubusercontent.com/83717897/117258469-ea7cf900-ae44-11eb-9624-220353e30280.JPG)

4) Image Normalisaton - Match the RGB histograms of images to a target image. This helps remove variations in staining.  Based on the Scikit-image package - https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html

![Normalisatio](https://user-images.githubusercontent.com/83717897/117258492-f10b7080-ae44-11eb-8e20-0fd1ab40b0d0.JPG)

5) Image preprocessing/ augmentation - Add a vast variety of image processsing techniques to your image set. These pre-processing techniques can help accentiate desired features, or add noise to help prevent overfitting during training of deep learning models. This module is based arround both the openCV and Imgaug (https://github.com/aleju/imgaug) libraries.

![Untitled-1](https://user-images.githubusercontent.com/83717897/117258506-f5378e00-ae44-11eb-9dbf-d76b5453f83b.jpg)



