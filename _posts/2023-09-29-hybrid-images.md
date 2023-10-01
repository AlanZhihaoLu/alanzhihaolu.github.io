---
layout: post
title:  "Creating Hybrid Images (Python, OpenCV)"
published: false
---

## What's a hybrid image?
Put simply: a hybrid image shows two images at non-overlapping spatial frequency bands. To create a hybrid image, one image is low-pass filtered and the other is high-pass filtered. Then, the images are squished together to form a single image.

Researchers in visual cognition are interested in hybrid images because low spatial frequencies (LSFs) and high spatial frequencies (HSFs) contain different types of information. LSFs represent "slow-changing" information across the image, so they give a general feel for the global visual structures in the image. On the other hand, HSFs represent "fast-changing" information across the image, so they give fine local details (i.e., 'sharpness') to the image. Since LSFs and HSFs contain different types of information, researchers have been curious to determine whether and how the visual system differentially processes these spatial frequency bands. Having two different images at LSFs and HSFs allows researchers to identify under what conditions the visual system prioritizes processing LSFs vs. HSFs. 

## How to make a hybrid image?
There are two ways to make hybrid images. You can use a convolution-based method or a Fourier-based method. The original introduction of hybrid images in Schyns & Oliva (1994) described a Fourier-based method, so that is the method I will use here. Do note, however, that it is possible to convert between convolutional and spectral filters (which I will describe here).

## Getting started
Take an image of a city and an image of a library:

![city](/assets/images/city.png) ![library](/assets/images/library.png)   

We're going to create a hybrid image of the two. First, import necessary libraries.
```python
import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
```

The main idea is to convert the image into Fourier space, then use a filter to filter out HSFs or LSFs. Let's convert the image of the city into Fourier space. 