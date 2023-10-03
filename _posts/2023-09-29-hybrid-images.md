---
layout: post
title:  "Creating Hybrid Images (Python, OpenCV): Part 1"
published: true
---

## What's a hybrid image?
Put simply: a hybrid image shows two images at non-overlapping spatial frequency bands. To create a hybrid image, one image is low-pass filtered and the other is high-pass filtered. Then, the images are squished together to form a single image.

Researchers in visual cognition are interested in hybrid images because low spatial frequencies (LSFs) and high spatial frequencies (HSFs) contain different types of information. LSFs represent "slow-changing" information across the image, so they give a general feel for the global visual structures in the image. On the other hand, HSFs represent "fast-changing" information across the image, so they give fine local details (i.e., 'sharpness') to the image. Since LSFs and HSFs contain different types of information, researchers have been curious to determine whether and how the visual system differentially processes these spatial frequency bands. Having two different images at LSFs and HSFs allows researchers to identify under what conditions the visual system prioritizes processing LSFs vs. HSFs. 

## How to make a hybrid image?
There are two ways to make hybrid images. You can use a convolution-based method or a Fourier-based method. The original introduction of hybrid images in Schyns & Oliva (1994) described a Fourier-based method, so that is the method I will use here. Do note, however, that it is possible to convert between convolutional and spectral filters (which I will describe here).

#### Getting started
Take an image of a city and an image of a library:  
![city](/assets/images/hybrid/city_gray.png) ![library](/assets/images/hybrid/library_gray.png)   

We're going to create a hybrid image of the two. First, import necessary libraries.
```python
import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
```
#### Step 1: Convert to Fourier space
The main idea is to convert the image into Fourier space, then use a filter to filter out HSFs or LSFs. Let's first convert the image of the city into Fourier space. Thankfully, doing so is incredibly easy, since Fourier transform functions are already implemented and available in `numpy.fft`.
```python
city_fft2 = fft2(city) #Compute the 2D discrete Fourier transform.
city_shifted = fftshift(city_fft2) #Shift DC component to center.
```

A visualization of the resulting `city_shifted` reveals a mesmerizing spectral representation of the original city image:  
![city_shifted](/assets/images/hybrid/city_shifted.png)  

The zero-frequency is located in the center (thanks to `fftshift`). The amplitude of each frequency is represented by its brightness in this visualization.  
Note that the maximum frequencies in the x- and y-directions are `image_width/2 - 1` and `image_height/2 - 1` respectively. This is because the oscillation is limited by the fact that values cannot oscillate within-pixel. Therefore, at least two pixels are needed to produce one cycle.

#### Step 2: Filter the spectral representation
From here, the `city_shifted` can be filtered however you want. Perhaps the simplest method of filtering is to impose a hard cut-off from frequencies higher/lower than desired. For example, we can simply generate a circular mask centered about the zero-frequency and set all values outside the circular mask equal to 0.  
```python
height, width = np.shape(city_shifted)
circle_mask = np.zeros((height, width)) #All zeros
circle_mask = cv2.circle(circle_mask, (width//2, height//2), 20, (255,255,255), -1) #Generate circular mask of radius=20
city_filtered = city_shifted * circle_mask #Apply filter
```
Visualizing `city_filtered`:  
![city_filtered](/assets/images/hybrid/city_shifted_circle_mask.png)  

#### Step 3: Convert back to pixel space
`city_filtered` can then be easily converted back to an image by using `ifft2` and `ifftshift` to undo the Fourier transforms.  
```python
city_low_pass = ifft2(ifftshift(city_filtered)) #First undo the fftshift, then undo the fft2
```  
Visualizing `city_low_pass`:  
![city_low_pass](/assets/images/hybrid/city_circle_mask_low.png)  

The resulting `city_low_pass` image is a blurry, LSF-only (i.e., low-pass) version of the original city image.

#### Step 4: Rinse and Repeat Steps 1-3 for HSF image
Steps 1-3 can be followed to generate a HSF-only (i.e., high-pass) version of the original library image.  
```python
library_fft2 = fft2(library) #Compute the 2D discrete Fourier transform.
library_shifted = fftshift(library_fft2) #Shift DC component to center.
```
The only difference is that an inverted circular mask is applied to the spectral representation of the library image.  
```python
height, width = np.shape(library_shifted)
circle_mask = np.ones((height, width)) #All ones
circle_mask = cv2.circle(circle_mask, (width//2, height//2), 20, (0,0,0), -1) #Generate circular mask of radius=20
library_filtered = library_shifted * circle_mask #Apply filter
```
Visualizing `library_filtered`:  
![library_filtered](/assets/images/hybrid/library_shifted_circle_mask.png)  

Convert back to image:  
```python
library_high_pass = ifft2(ifftshift(library_filtered)) #First undo the fftshift, then undo the fft2
```  
Visualizing `library_high_pass`:  
![library_high_pass](/assets/images/hybrid/library_circle_mask_high.png)  
The resulting `library_high_pass` keeps only areas that have rapidly changing intensities in the original library image, most noticeably in the sharp edges that define object boundaries.

#### Step 5: Combine images
Finally, HSF and LSF images are combined to create a hybrid image.  
However, there is an additional step to do before combining the two images - the HSF and LSF images need to be on the scale!  
Due to removing components that contributed significantly to the overall intensity, the `city_low_pass` and `library_high_pass` no longer exist on a 0-255 intensity range. Both images need to be normalized.  
```python
def normalize255(img_mat):
    img_mat = abs(img_mat) #Remove imaginary parts (imaginary parts should be essentially zero)
    img_mat = img_mat/np.max(img_mat) #Scale to 0-1
    img_mat = np.array(img_mat*255) #Scale to 0-255
    return img_mat

city_low_pass = normalize255(city_low_pass) #Normalize city_low_pass
library_high_pass = normalize255(library_high_pass) #Normalize library_high_pass
```  
Finally, average the two normalized images:
```python
city_library_hybrid = (city_low_pass + library_high_pass) / 2
```  
We end up with:  
![city_library_hybrid](/assets/images/hybrid/city_library_hybrid.png)  

## Additional Notes
In case you're interested in recreating the above visualizations, I've included here the functions I used.  
**Note:** Fourier transforms generally result in very high DC components - so much so that the intensity of the DC component will eclipse any other components.  
In other words, if you were to visualize, for example, `fftshift(fft2(city))` directly - you would see a single bright white dot at the DC component with everything else (relatively) black:  
![city_shifted_no_log](/assets/images/hybrid/city_shifted_no_log.png)  

Since the distribution of spectral intensity is so lopsided towards the DC component, to generate a nicer looking visualization, you can take the log of the Fourier transform:  
![city_shifted](/assets/images/hybrid/city_shifted.png)  

With that said, here is all the code I used in this post:

```python
#Simple function for displaying an image in a window.
def show_image(input_image):
    cv2.imshow('Displaying Image...', input_image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

#Function that normalizes to 0-255, then converts to BGR grayscale values.
def convert_to_gray(img_mat):
    img_mat = img_mat/np.max(img_mat)
    uint_img = np.array(img_mat*255).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    return grayImage

#Function that normalizes to 0-255
def normalize255(img_mat):
    img_mat = abs(img_mat)
    img_mat = img_mat/np.max(img_mat)
    img_mat = np.array(img_mat*255)
    return img_mat

##City image transforms and visualizations
city_fft2 = fft2(city) #Compute the 2D discrete Fourier transform.
city_shifted = fftshift(city_fft2) #Shift DC component to center.
cv2.imwrite('city_shifted.png',convert_to_gray(np.log(abs(city_shifted)))) #Note: abs() can be used to remove imaginary part

height, width = np.shape(city_shifted)
circle_mask = np.zeros((height, width)) #All zeros
circle_mask = cv2.circle(circle_mask, (width//2, height//2), 20, (255,255,255), -1) #Generate circular mask of radius=20
city_filtered = city_shifted * circle_mask #Apply filter
cv2.imwrite('city_filtered.png',convert_to_gray(np.log(abs(city_shifted)) * circle_mask))

city_low_pass = ifft2(ifftshift(city_filtered)) #First undo the fftshift, then undo the fft2
city_low_pass = normalize255(city_low_pass) #Normalize city_low_pass
cv2.imwrite('city_low_pass.png',convert_to_gray(city_low_pass))

##Library image transforms and visualizations
library_fft2 = fft2(library) #Compute the 2D discrete Fourier transform.
library_shifted = fftshift(library_fft2) #Shift DC component to center.
cv2.imwrite('library_shifted.png',convert_to_gray(np.log(abs(library_shifted))))

height, width = np.shape(library_shifted)
circle_mask = np.ones((height, width)) #All ones
circle_mask = cv2.circle(circle_mask, (width//2, height//2), 20, (0,0,0), -1) #Generate circular mask of radius=20
library_filtered = library_shifted * circle_mask #Apply filter
cv2.imwrite('library_filtered.png',convert_to_gray(np.log(abs(library_shifted)) * circle_mask))

library_high_pass = ifft2(ifftshift(library_filtered)) #First undo the fftshift, then undo the fft2
library_high_pass = normalize255(library_high_pass) #Normalize library_high_pass
cv2.imwrite('library_high_pass.png',convert_to_gray(library_high_pass))

##Generate city-library hybrid
city_library_hybrid = (city_low_pass + library_high_pass) / 2
cv2.imwrite('city_library_hybrid.png',convert_to_gray(city_library_hybrid))
```  

## That's all folks!
Stay tuned for part 2!