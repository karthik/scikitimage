# -*- coding: utf-8 -*-
"""
Created on Sat Mar 1 22:34:12 2014

@author: rishabgargeya
"""

import skimage 
from skimage import io, color
from skimage.exposure import rescale_intensity
from skimage.util import view_as_blocks
import numpy as np
import scipy.ndimage as ndi

#read image
rgb = io.imread('21_training.tif')

#change color scheme to LAB w/ type float
lab = color.rgb2lab(skimage.img_as_float(rgb))

#isolate luminosity channel
l_chan1 = lab[:,:,0]

#float between -1 and 1
l_chan1 /= np.max(np.abs(l_chan1))

#median filter of image w/ 5x5 window to remove noise before CLAHE
l_chan_med = ndi.median_filter(l_chan1, size=5)

#define CLAHE and other pertaining functions

MAX_REG_X = 16  # max. # contextual regions in x-direction */
MAX_REG_Y = 16  # max. # contextual regions in y-direction */
NR_OF_GREY = 16384  # number of grayscale levels to use in CLAHE algorithm

def equalize_adapthist(image, ntiles_x=8, ntiles_y=8, clip_limit=0.01,
                       nbins=256):
    args = [None, ntiles_x, ntiles_y, clip_limit * nbins, nbins]
    image = skimage.img_as_uint(image)
    args[0] = rescale_intensity(image, out_range=(0, NR_OF_GREY - 1))
    out = _clahe(*args)
    image[:out.shape[0], :out.shape[1]] = out
    image = rescale_intensity(image)
    return image
    
def _clahe(image, ntiles_x, ntiles_y, clip_limit, nbins=128):
    
    ntiles_x = min(ntiles_x, MAX_REG_X)
    ntiles_y = min(ntiles_y, MAX_REG_Y)
    ntiles_y = max(ntiles_x, 2)
    ntiles_x = max(ntiles_y, 2)

    if clip_limit == 1.0:
        return image  # is OK, immediately returns original image.

    map_array = np.zeros((ntiles_y, ntiles_x, nbins), dtype=int)

    y_res = image.shape[0] - image.shape[0] % ntiles_y
    x_res = image.shape[1] - image.shape[1] % ntiles_x
    image = image[: y_res, : x_res]

    x_size = image.shape[1] / ntiles_x  # Actual size of contextual regions
    y_size = image.shape[0] / ntiles_y
    n_pixels = x_size * y_size

    if clip_limit > 0.0:  # Calculate actual cliplimit
        clip_limit = int(clip_limit * (x_size * y_size) / nbins)
        if clip_limit < 1:
            clip_limit = 1
    else:
        clip_limit = NR_OF_GREY  # Large value, do not clip (AHE)

    bin_size = 1 + NR_OF_GREY / nbins
    aLUT = np.arange(NR_OF_GREY)
    aLUT /= bin_size
    img_blocks = view_as_blocks(image, (y_size, x_size))

    # Calculate greylevel mappings for each contextual region
    for y in range(ntiles_y):
        for x in range(ntiles_x):
            sub_img = img_blocks[y, x]
            hist = aLUT[sub_img.ravel()]
            hist = np.bincount(hist)
            hist = np.append(hist, np.zeros(nbins - hist.size, dtype=int))
            hist = clip_histogram(hist, clip_limit)
            hist = map_histogram(hist, 0, NR_OF_GREY - 1, n_pixels)
            map_array[y, x] = hist

    # Interpolate greylevel mappings to get CLAHE image
    ystart = 0
    for y in range(ntiles_y + 1):
        xstart = 0
        if y == 0:  # special case: top row
            ystep = y_size / 2.0
            yU = 0
            yB = 0
        elif y == ntiles_y:  # special case: bottom row
            ystep = y_size / 2.0
            yU = ntiles_y - 1
            yB = yU
        else:  # default values
            ystep = y_size
            yU = y - 1
            yB = yB + 1

        for x in range(ntiles_x + 1):
            if x == 0:  # special case: left column
                xstep = x_size / 2.0
                xL = 0
                xR = 0
            elif x == ntiles_x:  # special case: right column
                xstep = x_size / 2.0
                xL = ntiles_x - 1
                xR = xL
            else:  # default values
                xstep = x_size
                xL = x - 1
                xR = xL + 1

            mapLU = map_array[yU, xL]
            mapRU = map_array[yU, xR]
            mapLB = map_array[yB, xL]
            mapRB = map_array[yB, xR]

            xslice = np.arange(xstart, xstart + xstep)
            yslice = np.arange(ystart, ystart + ystep)
            interpolate(image, xslice, yslice,
                        mapLU, mapRU, mapLB, mapRB, aLUT)

            xstart += xstep  # set pointer on next matrix */

        ystart += ystep

    return image
    
def clip_histogram(hist, clip_limit):
    
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = int(n_excess / hist.size)  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    hist[excess_mask] = clip_limit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = (hist >= upper) & (hist < clip_limit)
    mid = hist[mid_mask]
    n_excess -= mid.size * clip_limit - mid.sum()
    hist[mid_mask] = clip_limit

    while n_excess > 0:  # Redistribute remaining excess
        index = 0
        while n_excess > 0 and index < hist.size:
            step_size = int(hist[hist < clip_limit].size / n_excess)
            step_size = max(step_size, 1)
            indices = np.arange(index, hist.size, step_size)
            under = hist[indices] < clip_limit
            hist[under] += 1
            n_excess -= hist[under].size
            index += 1

    return hist

def map_histogram(hist, min_val, max_val, n_pixels):
    
    out = np.cumsum(hist).astype(float)
    scale = ((float)(max_val - min_val)) / n_pixels
    out *= scale
    out += min_val
    out[out > max_val] = max_val
    return out.astype(int)
    
def interpolate(image, xslice, yslice,
                mapLU, mapRU, mapLB, mapRB, aLUT):
    norm = xslice.size * yslice.size  # Normalization factor
    # interpolation weight matrices
    x_coef, y_coef = np.meshgrid(np.arange(xslice.size),
                                 np.arange(yslice.size))
    x_inv_coef, y_inv_coef = x_coef[:, ::-1] + 1, y_coef[::-1] + 1

    view = image[yslice[0]: yslice[-1] + 1, xslice[0]: xslice[-1] + 1]
    im_slice = aLUT[view]
    new = ((y_inv_coef * (x_inv_coef * mapLU[im_slice]
                          + x_coef * mapRU[im_slice])
            + y_coef * (x_inv_coef * mapLB[im_slice]
                        + x_coef * mapRB[im_slice]))
           / norm)
    view[:, :] = new
    return image

# End of Function Definitions for CLAHE ---------------------------------------

#apply CLAHE on median filtered l channel for increased contrast before data analysis
l_chan_clahe = equalize_adapthist(l_chan_med, ntiles_x=8, ntiles_y=8, clip_limit=0.01,
                       nbins=256)

#show fully pre-processed image
#list of variables for reference: rgb, lab, l_chan1, l_chan_med, l_chan_clahe
skimage.io.imshow(l_chan_clahe)


#start removal of Optic Disk (big white circle where blood vessels branch out from)

#apply grayscale morphological closing 
l_close = ndi.grey_closing(l_chan1, size = (10,10))

skimage.io.imshow(l_close)

#binarize by thresholding to try and segment optic disk
"""
I'm trying to work out here how to perfom the binary thresold and invert/overly  
the output marker image - however, my plot histograms end up returning all white 
and neither otsu nor adaptive give me what I want, can you help me generate a 
similar output to figure 3(b) from the paper from Sopharak? thanks
"""


#image = l_close

#block_size = 1000
#binary_adaptive = threshold_adaptive(image, block_size, offset=10)
#binary = (image > l_close)

p2 = np.percentile(l_close, 1)
p98 = np.percentile(l_close, 99)
img_rescale = rescale_intensity(l_close, in_range=(p2, p98))

coins = img_rescale
binary = coins < 0.9


skimage.io.imshow(binary)

import Image

background = l_chan1
overlay = binary

new_img = Image.blend(background, overlay, 0.5)
skimage.io.imshow(new_img)







from skimage.morphology import disk
from skimage.morphology import opening, closing

selem2=disk(8)
selem=disk(10)

opened = opening(binary, selem2)
closed = closing(opened, selem)

marker = l_chan_clahe - closed
#skimage.io.imshow(closed)










