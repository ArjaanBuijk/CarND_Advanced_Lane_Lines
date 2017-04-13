
# coding: utf-8

# # **Advanced Finding of Lane Lines on the Road** 
# ***
# In this project, I am applying the tools learned to identify lane lines on the road. My approach was to take the Jupyter notebook of Project 1, which did a decent job already on finding the lanes, and expand it with the new techniques learned in Lesson 16. As in Project 1, I first developed the pipeline on a series of individual images, and later apply the result to the provided video streams. 

# Steps:
# 1. Camera calibration
# 2. Distortion correction
# 3. Color & Gradient threshold
# 4. Perspective transform
# 5. Detect lane lines
# 6. Determine the lane curvature

# In[1]:

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
import pickle
from skimage import exposure
from tqdm import tqdm

#get_ipython().magic('matplotlib inline')


# In[2]:

# some global utility functions, mainly for visualization


# function to show an image with title
def show_image(image, title, cmap=None ):
    fig, ax = plt.subplots(1, 1, figsize=(24, 9))
    plt.title(title)
    if cmap:
        plt.imshow(image, cmap=cmap) # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    else:
        plt.imshow(image)  
    plt.show()
    
# function to plot original & modifed images side-by-side
def plot_orig_and_changed_image(image1, description1='Original Image',
                                image2=None, description2='Changed Image',
                                file_out=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    if image1.ndim == 3 and image1.shape[2] ==3:
        ax1.imshow(image1)
    else:
        ax1.imshow(image1.squeeze(), cmap='gray')
    ax1.set_title(description1, fontsize=50)
    
    if image2 is not None:
        if image2.ndim == 3 and image2.shape[2] == 3:
            ax2.imshow(image2)
        else:
            ax2.imshow(image2.squeeze(), cmap='gray')
        ax2.imshow(image2, cmap='gray')
        ax2.set_title(description2, fontsize=50)
    
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #plt.show()
    
    if file_out is not None:
        f.savefig(file_out)
        print('Writen file: '+file_out)
        
    plt.close(f)

# function to draw straight lines on top of an rgb image
def draw_lines_on_image(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness` on an rgb image.    
    Lines are drawn on the image inplace (mutates the image).
    """
    for line in lines:
        p1 = line[0]
        p2 = line[1]
        cv2.line(img, p1, p2, color, thickness)
        
# function to draw the outline of square windows on top of an rgb image
def draw_windows_on_image(img, 
                          window_centroids, window_width, window_height, 
                          color=[0, 255, 0]):
    dw = 0.5*window_width
    dh = 0.5*window_height 
    yc = img.shape[0]-dh   # y at center of window
    
    # store the outlines of each window
    lines = []
    for centroids in window_centroids: # levels of windows
        for xc in centroids:  # x at center of left & right windows at each level
            p1 = (int(xc+dw), int(yc+dh)) # top right      - make it all integer, because draw_lines needs pixel #s.
            p2 = (int(xc+dw), int(yc-dh)) # bottom right
            p3 = (int(xc-dw), int(yc-dh)) # bottom left
            p4 = (int(xc-dw), int(yc+dh)) # top left
            lines.append((p1,p2))
            lines.append((p2,p3))
            lines.append((p3,p4))
            lines.append((p4,p1))
            
        yc = yc - window_height    
            
    # draw lines on the image in green
    draw_lines_on_image(img, lines, color=[0, 255, 0], thickness=2)


# # 1. Camera Calibration

# In[3]:

# prepare object points
nx = 9 # The number of inside corners in x
ny = 6 # The number of inside corners in y

# Make a list of calibration images
file_dir = "camera_cal"
file_dir_out = "camera_cal_output"
files = os.listdir(file_dir)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

print('Finding chessboard in calibration images')
for file in files:
    file_in=file_dir+"/"+file
    file_out=file_dir_out+"/Corners_on_"+file
    
    image = mpimg.imread(file_in)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found
    # - draw corners
    # - append object points & image points to storage arrays
    if ret == True:
        objpoints.append(objp)    # note: same for all calibration images !
        imgpoints.append(corners)
        
        # Draw and display the corners
        image2 = np.copy(image)
        cv2.drawChessboardCorners(image2, (nx, ny), corners, ret)
        plot_orig_and_changed_image(image1=image , description1=file,
                                    image2=image2, description2='With Corners',
                                    file_out=file_out)
    else:
        plot_orig_and_changed_image(image1=image, description1=file,
                                    image2=None , description2=None)
        


# In[4]:

print('Calibrating the Camera')
# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                   imgpoints, 
                                                   gray.shape[::-1], 
                                                   None, None)

# Save the calibrated camera data
pickle_file = 'calibrated_camera.pickle'

try:
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(
            {
                'ret': ret,
                'mtx': mtx,
                'dist': dist,
                'rvecs': rvecs,
                'tvecs': tvecs,
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
    
print('Cached calibrated camera data in pickle file: '+pickle_file)


# # 2. Distortion Correction

# In[5]:

pickle_file = 'calibrated_camera.pickle'
print('Reading calibration data from pickle file: '+pickle_file)

with open(pickle_file, 'rb') as f:
    pickle_data    = pickle.load(f)
    ret    = pickle_data['ret']
    mtx    = pickle_data['mtx']
    dist   = pickle_data['dist']
    rvecs  = pickle_data['rvecs']
    tvecs  = pickle_data['tvecs']
    del pickle_data  # Free up memory

def undistort(image):
    return cv2.undistort(image, mtx, dist, None, mtx)
    


# In[6]:

##print('Testing the Calibrated Camera')
##
### Make a list of test images
##file_dir = "camera_cal"
##file_dir_out = "camera_cal_output"
##files = os.listdir(file_dir)
##
##for file in files:
##    file_in=file_dir+"/"+file
##    file_out=file_dir_out+"/Distortion_Correction_"+file
##    
##    image = mpimg.imread(file_in)
##    
##    dst = undistort(image)
##    
##    plot_orig_and_changed_image(image1=image , description1=file,
##                               image2=dst, description2='Distortion Corrected',
##                               file_out=file_out)


# #  3. Color & Gradient threshold

# In[7]:

# function to convert RGB image to grayscale
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# function to Adaptive Gaussian Thresholding
# See - http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
def adaptive_gaussian_threshold(gray, thresholds=(0, 255)):
    # Otsu's thresholding after Gaussian filtering
    #blur = cv2.GaussianBlur(gray,(3,3),0)
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    #blur = cv2.GaussianBlur(gray,(9,9),0)
    #blur = cv2.GaussianBlur(gray,(15,15),0)
    blur = np.array(blur, dtype=np.uint8) # else it sometimes crashes...
    ret3,th3 = cv2.threshold(blur,thresholds[0],thresholds[1],cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3

# function to turn off pixels outside crop-window
def crop_binary_image(img,crop_left=None, crop_right=None,
                          crop_bot=None, crop_top=None):
    if crop_left is None:   crop_left=0
    if crop_right is None:  crop_right=img.shape[1]
    if crop_bot is None:    crop_bot=im.shape[0]
    if crop_top is None:    crop_top=0
    
    image = np.copy(img)
    
    image[:,0:crop_left] = 0
    image[:,crop_right:] = 0
    image[0:crop_top,:]  = 0
    image[crop_bot:,:]   = 0
    
    return image

# function to convert binary image to RGB
def binary_to_rgb(binary_img):
    # Create an new rgb image
    # binary_image is 2D array with either 0 or 1 values
    # --> Where value=1, [R,G,B] = [255, 255, 255] (colors those new image pixels white)
    rgb_img = np.stack((binary_img, binary_img, binary_img),axis=-1)*255
    return rgb_img
    
    
def abs_sobel_thresh(img, orient='x', ksize=5, thresh=(0,255), rgb_in=True ):
    # Convert to grayscale
    if rgb_in:
        gray = grayscale(img)
    else:
        gray = img
    
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=ksize))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=ksize))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return binary_output

def mag_thresh(img, ksize=5, mag_thresh=(0, 255), rgb_in=True):
    # Convert to grayscale
    if rgb_in:
        gray = grayscale(img)
    else:
        gray = img
        
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, ksize=5, thresh=(0, np.pi/2), rgb_in=True):
    # SEE: https://goo.gl/nXmgN0 
    # Grayscale
    # Note: Make sure you use the correct grayscale conversion depending on how you've read
    #       in your images. 
    #       Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread(). 
    #       Use cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().
    if rgb_in:
        gray = grayscale(img)
    else:
        gray = img
        
    # Calculate the x and y gradients
    # x direction (the 1, 0 at the end denotes x direction)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    # y direction (the 0, 1 at the end denotes y direction)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    # -> pixels have a value of 1 or 0, based on the strength of the gradient.
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
        
def pipeline(img, show_all=False):
    img = np.copy(img)
    
    # ============================================================================
    # Combine some gradient thresholds
    # Choose a Sobel kernel size
    ksize = 5 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', ksize=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(img, orient='y', ksize=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(img, ksize=ksize, mag_thresh=(30, 100))
    # Note: threshold is angle: 0 = horizontal; +/- np.pi/2 = vertical
    dir_binary = dir_threshold(img, ksize=15, thresh=(0.7, 1.3))

    if show_all:
        show_image(gradx,      title='Thresholded X-Gradient', cmap='gray')
        show_image(grady,      title='Thresholded Y-Gradient', cmap='gray')
        show_image(mag_binary, title='Thresholded Magnitude' , cmap='gray')
        show_image(dir_binary, title='Thresholded Grad. Dir.', cmap='gray')

    # apply Adaptive Gaussian threshold to all of them
    gradx = adaptive_gaussian_threshold(gradx, thresholds=(0, 1))
    grady = adaptive_gaussian_threshold(grady, thresholds=(0, 1))
    mag_binary = adaptive_gaussian_threshold(mag_binary, thresholds=(0, 1))
    dir_binary = adaptive_gaussian_threshold(dir_binary, thresholds=(0, 1))

    # Combined thresholds
    # 
    # For example, here is a selection for pixels where both the x and y gradients 
    # meet the threshold criteria, or the gradient magnitude and direction are both 
    # within their threshold values.

    grad_binary = np.zeros_like(gradx)
    grad_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    if show_all:
        show_image(gradx,       title='AGT X-Gradient' , cmap='gray')
        show_image(grady,       title='AGT Y-Gradient' , cmap='gray')
        show_image(mag_binary,  title='AGT Magnitude'  , cmap='gray')
        show_image(dir_binary,  title='AGT Grad. Dir.' , cmap='gray')
        show_image(grad_binary, title='AGT Grad Binary', cmap='gray')
            
        
    # ============================================================================
    # Threshold the HSV space
    
    #OK-FOR-PROJECT-VIDEO v_thresh=(225, 255)
    h_thresh=(0  , 90)
    s_thresh=(100, 255)
    v_thresh=(175, 255)
    
    
    # Convert to HSV color space and separate the S channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    h_channel = hsv[:,:,0]   # [0-179]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    
    #print ('h_max={0}'.format(np.amax(h_channel)))
    #print ('s_max={0}'.format(np.amax(s_channel)))
    #print ('v_max={0}'.format(np.amax(v_channel)))
    # Threshold color channels
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
    
    hsv_binary = np.zeros_like(h_channel)
    hsv_binary[((h_binary == 1) & (s_binary == 1)) & (v_binary == 1)] = 1
    
    if show_all:
        show_image(h_channel,  title='h_hsv_channel', cmap='gray')
        show_image(s_channel,  title='s_hsv_channel', cmap='gray')
        show_image(v_channel,  title='v_hsv_channel', cmap='gray')
        show_image(h_binary,   title='Thresholded h channel', cmap='gray')
        show_image(s_binary,   title='Thresholded s channel', cmap='gray')
        show_image(v_binary,   title='Thresholded v channel', cmap='gray')
        show_image(hsv_binary, title='Thresholded hsv channel', cmap='gray')
    
    # ============================================================================
    # Threshold HLS space
    
    h_thresh=(0  , 90)
    l_thresh=(175, 255)
    s_thresh=(100, 255)
    
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]   # [0-179]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    #print ('h_max={0}'.format(np.amax(h_channel)))
    #print ('l_max={0}'.format(np.amax(l_channel)))
    #print ('s_max={0}'.format(np.amax(s_channel)))
    
    # Threshold color channels
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    hls_binary = np.zeros_like(h_channel)
    hls_binary[((h_binary == 1) & (s_binary == 1)) & (l_binary == 1)] = 1
    
    if show_all:
        show_image(h_channel,  title='h_hls_channel', cmap='gray')
        show_image(l_channel,  title='l_hls_channel', cmap='gray')
        show_image(s_channel,  title='s_hls_channel', cmap='gray')
        show_image(h_binary,   title='Thresholded h channel', cmap='gray')
        show_image(l_binary,   title='Thresholded l channel', cmap='gray')
        show_image(s_binary,   title='Thresholded s channel', cmap='gray')
        show_image(hls_binary, title='Thresholded hls channel', cmap='gray')
        
    # ============================================================================d
    # Stack each channel
    color_binary = np.dstack((grad_binary, hsv_binary, hls_binary))
    
    # Combine the binary thresholds
    combined_binary = np.zeros_like(grad_binary)
    combined_binary[(grad_binary == 1) | (hsv_binary == 1) | (hls_binary == 1) ] = 1  
    
    return color_binary, combined_binary


# In[8]:

##print('Testing the color & gradient thresholding')
##
### Make a list of test images
##file_dir = "test_images"
##file_dir_out = "test_images_output"
##files = os.listdir(file_dir)
##
###files = ['project_video_output0994.jpg',
###         'challenge_video_output0001.jpg',
###         'challenge_video_output0239.jpg']
##
###files = ['challenge_video_output0239.jpg']
##
##show_all=False
##for file in files:
##    file_in=file_dir+"/"+file
##    file_out1=file_dir_out+"/color_binary_"+file
##    file_out2=file_dir_out+"/combined_binary_"+file
##    
##    image = mpimg.imread(file_in)
##    
##    # distortion correct the image
##    image = undistort(image)
##    
##    if show_all:
##        show_image(image, title=file)
##    
##    # threshold the image
##    color_binary, combined_binary = pipeline(image, show_all=show_all)
##    
##    plot_orig_and_changed_image(image1=image , description1=file,
##                               image2=color_binary, description2='color binary',
##                               file_out=file_out1)
##    
##    plot_orig_and_changed_image(image1=image , description1=file,
##                               image2=combined_binary, description2='thresholded binary',
##                               file_out=file_out2)


# # 4. Perspective transform

# In[9]:

# Interactively find the 4 corner points that will be used as the source points
file_in="test_images/straight_lines1.jpg"   
img = mpimg.imread(file_in)

image = np.copy(img)
# distortion correct the image
image = undistort(image)

# source points
p1 = ( 723,            475) # top right
p2 = (1110, image.shape[0]) # bottom right
p3 = ( 204, image.shape[0]) # bottom left
p4 = ( 562,            475) # top left

#p1 = ( 685,            450) # top right
#p2 = (1110, image.shape[0]) # bottom right
#p3 = ( 204, image.shape[0]) # bottom left
#p4 = ( 597,            450) # top left

# draw source lines on the image in red
src_lines = [[p1,p2], [p2,p3], [p3,p4], [p4,p1]]
draw_lines_on_image(image, src_lines, color=[255, 0, 0], thickness=2)

# destination points
correctX = 100
dp1 = ( p2[0]-correctX,              0) # top right
dp2 = ( p2[0]-correctX, image.shape[0]) # bottom right
dp3 = ( p3[0]+correctX, image.shape[0]) # bottom left
dp4 = ( p3[0]+correctX,              0) # top left

# draw destination lines on the image in green
dst_lines = [[dp1,dp2], [dp2,dp3], [dp3,dp4], [dp4,dp1]]
draw_lines_on_image(image, dst_lines, color=[0, 255, 0], thickness=2)

fig = plt.figure(figsize=(15,15))
ax = fig.gca()
ax.set_xticks(np.arange(0, image.shape[1], 50))
ax.set_yticks(np.arange(0, image.shape[0], 50))
plt.imshow(image)
plt.grid(True)

# ------------------------------------------------------------------------
# Set perspective transform and un-transform matrices
img_size = (image.shape[1], image.shape[0])

# four source coordinates
src = np.float32([p1,p2,p3,p4])

# four desired coordinates
dst = np.float32([dp1,dp2,dp3,dp4])

# compute the perspective transform, M
M = cv2.getPerspectiveTransform(src, dst)

# compute the inverse perspective transform, Minv
Minv = cv2.getPerspectiveTransform(dst, src)

# Define a function that does the actual warping (pass in M), and unwarping (pass in Minv)
def warp(img, M):
    img_size = (img.shape[1], img.shape[0])
    # create warped image, using linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)    
    return warped

# As a test, warp the image, show the grid
image = np.copy(img)
# draw source lines on the image in red
draw_lines_on_image(image, src_lines, color=[255, 0, 0], thickness=2)
# now warp it
warped = warp(image,M)
# draw destination lines on the warped image in green
draw_lines_on_image(warped, dst_lines, color=[0, 255, 0], thickness=2)
fig = plt.figure(figsize=(15,15))
ax = fig.gca()
ax.set_xticks(np.arange(0, warped.shape[1], 50))
ax.set_yticks(np.arange(0, warped.shape[0], 50))
plt.imshow(warped)
plt.grid(True)


# # 5. Detect lane lines

# In[14]:

# Sliding Windows Search by applying a convolution
# -------------------------------------------------
# This sliding window method applies a convolution, which maximizes the number of "hot" 
# pixels in each window. A convolution is the summation of the product of two separate 
# signals, in our case the window template and the vertical slice of the pixel image.
#
# We slide a window template across the image from left to right and any overlapping 
# values are summed together, creating the convolved signal. The peak of the convolved 
# signal is where there was the highest overlap of pixels and the most likely position 
# for the lane marker.

WINDOW_WIDTH = 80 
WINDOW_HEIGHT = 40
MARGIN = 80 # How much to slide left and right for searching

CROP_LEFT = 20
CROP_RIGHT = 1260
CROP_BOT = 710
CROP_TOP = 0

MIN_CONVSIGNAL = 100  # if window of convolution has signal lower than this value, it will be rejected
#MIN_CONVSIGNAL = 25  # if window of convolution has signal lower than this value, it will be rejected

MAX_REJECTED = 100 # if this number of windows are rejected in sequence, reject all following windows
                   # this avoids picking up wrong blobs in strongly curved lanes

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, verbose=False):
    
    # window_height must be equal to hood section we skip !
    window_width = WINDOW_WIDTH
    window_height = WINDOW_HEIGHT
    margin = MARGIN
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    if verbose:
        print('initial l_center, r_center = {0}, {1}'.format(l_center, r_center))
    
    midpoint = int(image.shape[1]/2)
    if l_center == 0:
        l_center = 0.5*midpoint
    if r_center == 0:
        r_center = midpoint + 0.5*midpoint
    
    if verbose:
        print('initial midpoint, l_center, r_center = {0}, {1} {2}'.format(midpoint,l_center, r_center))
    
    ## Take a histogram of the bottom half of the image
    #histogram = np.sum(image[image.shape[0]/2:,:], axis=0)
    ## Find the peak of the left and right halves of the histogram
    ## Protect it from l_center = 0
    #midpoint = np.int(histogram.shape[0]/2)
    #l_center = np.argmax(histogram[:midpoint])
    #if l_center == 0:
    #    l_center = 0.5*midpoint
    #r_center = np.argmax(histogram[midpoint:])
    #if r_center == 0:
    #    r_center = 0.5*midpoint
    #r_center = midpoint + r_center
    
    
    # Add what we found for the bottom level
    window_centroids.append((l_center,r_center))
    
    # Keep track if we have found an accepted window
    found_accepted_left = False
    found_accepted_right = False
    
    # Keep track of number of rejected windows in sequence
    l_num_rejected_sequence = 0
    r_num_rejected_sequence = 0
    
    # Go through each level, including 1st level, looking for max pixel locations
    # We go through the bottom level too, to still allow rejection of the window due to to weak a convsignal..
    for level in range(0,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),
                                   :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of 
        # window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin, 0.5*window_width ))
        l_max_index = int(min(l_center+offset+margin, image.shape[1] - 0.5*window_width))
        l_convsignal = np.max(conv_signal[l_min_index:l_max_index])
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0.5*window_width ))
        r_max_index = int(min(r_center+offset+margin,image.shape[1] - 0.5*window_width))
        r_convsignal = np.max(conv_signal[r_min_index:r_max_index])
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        
        
        # If best centroid has too low a convsignal, reject it.
        # Mark this by setting it to negative value of center of previous layer
        # Also, reject any window once certain number have been rejected in sequence
        if l_num_rejected_sequence >= MAX_REJECTED or l_convsignal < MIN_CONVSIGNAL:
            l_center = -abs(window_centroids[-1][0])
            l_num_rejected_sequence += 1
        else:
            l_num_rejected_sequence = 0
            
        if r_num_rejected_sequence >= MAX_REJECTED or r_convsignal < MIN_CONVSIGNAL:
            r_center = -abs(window_centroids[-1][1])
            r_num_rejected_sequence += 1
        else:
            r_num_rejected_sequence = 0
            
        if level==0:
            # set it again, perhaps now in rejected status.
            window_centroids[0] = (l_center, r_center)
        else:
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))
        if verbose:
            print ('level, l_center, r_center, l_convsignal, r_convsignal = {0}, {1}, {2}, {3}, {4}'.format(level, l_center, r_center, l_convsignal, r_convsignal) )
        
        # if this is our first accepted window, correct the ones below to this location
        if found_accepted_left == False and l_center > 0:
            found_accepted_left = True
            for ii in range(len(window_centroids)-1):
                window_centroids[ii] = (-l_center, window_centroids[ii][1])
        if found_accepted_right == False and r_center > 0:
            found_accepted_right = True
            for ii in range(len(window_centroids)-1):
                window_centroids[ii] = (window_centroids[ii][0], -r_center)
    
        # reset it to positive values for next level
        l_center = abs(l_center)
        r_center = abs(r_center)
    
    return window_centroids

# function to extract left and right lane pixel indices inside the windows
def fit_polynomials_trough_pixels_in_lane_windows(binary_img_in, 
                          window_centroids, window_width, window_height,
                          visualize=True,verbose=False):
    
    binary_img = np.copy(binary_img_in)
    
    # If requested, create & return a new image that visualizes the process
    out_img = None
    if visualize:
        out_img = binary_to_rgb(binary_img)
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    dw = 0.5*window_width
    dh = 0.5*window_height 
    yc = binary_img.shape[0]-dh   # y at center of window
    
    #NO...! Activate one pixel at the center of each rejected window
    # This gives mostly a better lane fit in case of lots of rejected windows
    activate_center_of_rejected_windows=False
    if activate_center_of_rejected_windows:
        for centroids in window_centroids: # levels of windows
            # x at center of left & right windows at each level
            xc_left  = centroids[0]
            xc_right = centroids[1]
            #print ('xc_left, xc_right, yc={0}, {1}, {2}'.format(xc_left, xc_right, yc))

            if xc_left < 0:
                binary_img[int(yc), int(abs(xc_left))] = 1
            if xc_right < 0:
                binary_img[int(yc), int(abs(xc_right))] = 1

            # shift level of windows up for next level    
            yc = yc - window_height
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    lines = []
    yc = binary_img.shape[0]-dh   # y at center of window
    for centroids in window_centroids: # levels of windows
        # x at center of left & right windows at each level
        xc_left  = centroids[0]
        xc_right = centroids[1]
        
        # negative values indicated that window was rejected for this level
        # -> pixels in these windows are ignored
        # -> the windows will be colored in red instead of green
        left_accepted, right_accepted = True, True
        color_left, color_rigt = (0,255,0), (0,255,0)
        if xc_left <= 0:
            left_accepted = False
            color_left = (255,0,0)
        if xc_right < 0:
            right_accepted = False
            color_rigt = (255,0,0)
        
        # set it to positive values, so we can draw it...
        xc_left = abs(xc_left)
        xc_right = abs(xc_right)
        
        # left & right window corners
        win_y_low       = int(yc - dh)
        win_y_high      = int(yc + dh)
        win_xleft_low   = int(xc_left  - dw)
        win_xleft_high  = int(xc_left  + dw)
        win_xright_low  = int(xc_right - dw)
        win_xright_high = int(xc_right + dw)
        # Identify the nonzero pixels in x and y within the window & Append these indices to the lists
        #NO...! Include any pixels in the rejected windows too, because logic took care that it is positioned reasonably, and
        #       it improves the polynomial detection.
        if left_accepted:
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
        
        if right_accepted:
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            right_lane_inds.append(good_right_inds)

        # draw the windows if requested
        if visualize:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),color_left,2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),color_rigt,2) 

        
        # shift level of windows up for next level    
        yc = yc - window_height    

    # Concatenate the arrays of indices
    if len(left_lane_inds) > 0:
        left_lane_inds = np.concatenate(left_lane_inds)
    if len(right_lane_inds) > 0:
        right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = None
    right_fit = None
    if len(left_lane_inds) >0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(right_lane_inds) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    
    # ===========================================================
    # Calculate lane curvature in meters
    left_curverad = None
    right_curverad = None
    
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit polynomials to x,y in world space & calculate the radius of curvature
    y_eval_m = np.max(binary_img.shape[0])*ym_per_pix # at bottom of picture, where the car is
    if left_fit is not None:
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_m + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    if right_fit is not None:    
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_m + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    if verbose:
        print('left curvature, right curvatue = {0}m, {1}m'.format(left_curverad,right_curverad))
    
    # ===========================================================
    # Calculate offset in meters
    offset = None
    
    x_lane_center_m = (binary_img.shape[1]/2.0)*xm_per_pix
    
    if left_fit is not None and right_fit is not None:
        x_lane_left_m  = left_fit_cr[0]*y_eval_m**2 + left_fit_cr[1]*y_eval_m + left_fit_cr[2]
        x_lane_right_m = right_fit_cr[0]*y_eval_m**2 + right_fit_cr[1]*y_eval_m + right_fit_cr[2]
        x_car_center_m = 0.5*(x_lane_left_m + x_lane_right_m)
        offset = x_car_center_m - x_lane_center_m
        
    
    if verbose:
        print('left curvature, right curvatue = {0}m, {1}m'.format(left_curverad,right_curverad))

        
    # ===========================================================
    # If requested, color lane pixels and draw polynomial on image
    if visualize:
        # color left-lane pixels red
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # color right-lane pixels blue
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate x and y values for plotting of polynomials
        ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0] )
        
        if left_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        if right_fit is not None:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
        # Draw as yellow lines on the out_image
        for i in range(len(ploty) - 1):
            # left lane
            if left_fit is not None:
                p1 = (int(left_fitx[i]  ), int(ploty[i]  ))
                p2 = (int(left_fitx[i+1]), int(ploty[i+1]))
                cv2.line(out_img, p1, p2, [255, 255, 0], 2)
            # right lane
            if right_fit is not None:
                p1 = (int(right_fitx[i]  ), int(ploty[i]  ))
                p2 = (int(right_fitx[i+1]), int(ploty[i+1]))
                cv2.line(out_img, p1, p2, [255, 255, 0], 2)
    
    
    return left_fit, right_fit, left_curverad, right_curverad, offset, out_img

def find_new_fit(warped, left_fit, right_fit,
                 visualize=True,verbose=False):    
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "warped")
    # It's now much easier to find line pixels!
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
    # Again, extract left and right line pixel positions &
    # Fit a second order polynomial to each
    left_fit = None
    right_fit = None
    if len(left_lane_inds) >0:
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(right_lane_inds) > 0:
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    if left_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    if right_fit is not None:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #
    # If requested, create & return a new image that visualizes the process
    result_img = None
    if visualize:
        out_img = binary_to_rgb(warped)
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        if left_fit is not None:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        if right_fit is not None:
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        if left_fit is not None:
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
        if right_fit is not None:
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane search area onto the warped blank image
        if left_fit is not None:
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        if right_fit is not None:
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        
        result_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Draw the fitted polynomial as yellow lines on the result_img
        for i in range(len(ploty) - 1):
            # left lane
            if left_fit is not None:
                p1 = (int(left_fitx[i]  ), int(ploty[i]  ))
                p2 = (int(left_fitx[i+1]), int(ploty[i+1]))
                cv2.line(result_img, p1, p2, [255, 255, 0], 2)
            # right lane
            if right_fit is not None:
                p1 = (int(right_fitx[i]  ), int(ploty[i]  ))
                p2 = (int(right_fitx[i+1]), int(ploty[i+1]))
                cv2.line(result_img, p1, p2, [255, 255, 0], 2)
        
    return left_fit, right_fit, result_img
    

# using techniques described here: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
def draw_lanes_on_original_image(img,left_fit, right_fit,
                                 left_curverad=None, right_curverad=None, offset=None,
                                 text1=None):
    # create two copies of the original image -- one for
    # the overlay and one for the final output image
    overlay_img = img.copy()
    out_img     = img.copy()

    # -----------------------------------------------------------------------
    # prepare left and right lines as sequence of points
    # Generate x and y values for plotting of polynomials
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    src_pts_left  = []
    src_pts_right = []

    for i in range(len(ploty)):
        src_pts_left.append ( [left_fitx[i] , ploty[i] ] )
        src_pts_right.append( [right_fitx[i], ploty[i] ] )

    # get source point arrays in correct shape for perspectiveTransform
    # see: http://answers.opencv.org/question/252/cv2perspectivetransform-with-python/
    src_pts_left = np.array(src_pts_left, dtype='float32')
    src_pts_right = np.array(src_pts_right, dtype='float32')

    src_pts_left  = np.array([src_pts_left])
    src_pts_right = np.array([src_pts_right])

    # use the inverse perspective transform to get location of these points in
    # the original image
    dst_pts_left  = cv2.perspectiveTransform(src_pts_left , Minv)   # shape = (1, 720, 2)
    dst_pts_right = cv2.perspectiveTransform(src_pts_right, Minv)

    # -----------------------------------------------------------------------
    # draw a filled polygon in between lanes on overlay copy, in green
    # See: http://stackoverflow.com/questions/11270250/what-does-the-python-interface-to-opencv2-fillpoly-want-as-input
    polygon = dst_pts_left[0].tolist() + list(reversed(dst_pts_right[0].tolist()))
    polygon = np.array(polygon, 'int32')
    cv2.fillConvexPoly(overlay_img, polygon, (0, 255, 0), lineType=8, shift=0)

    # draw left lane (red) and right lane (blue) on overlay copy
    pts_left = np.array(dst_pts_left, 'int32')
    pts_left = pts_left.reshape((-1,1,2))
    cv2.polylines(overlay_img,[pts_left],False,(255,0,0),thickness=4)
    
    pts_right = np.array(dst_pts_right, 'int32')
    pts_right = pts_right.reshape((-1,1,2))
    cv2.polylines(overlay_img,[pts_right],False,(0,0,255),thickness=4)    

    # -----------------------------------------------------------------------  
    # apply overlay copy to out_img, with transparency
    alpha=0.5
    cv2.addWeighted(overlay_img, alpha, out_img, 1 - alpha, 0, out_img)

    # -----------------------------------------------------------------------
    # Put Text on out_img

    if text1 is not None:
        cv2.putText(out_img, text1,(10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Put curvature measures as text on overlay

    if left_curverad is not None:
        msg = 'left curvature = {0} m'.format(int(left_curverad))
        cv2.putText(out_img, msg,(10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
    if right_curverad is not None:
        msg = 'right curvature = {0} m'.format(int(right_curverad))
        cv2.putText(out_img, msg,(10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
    if offset is not None:
        if offset < 0.0:
            msg = 'Vehicle is {0:.2f} m left of center'.format(abs(offset))
        else:
            msg = 'Vehicle is {0:.2f} m right of center'.format(abs(offset))
            
        cv2.putText(out_img, msg,(10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)    
        
    # return new image
    return out_img

    


# In[15]:

# put it all in one function...
def process_image(image, left_fit_prev=None, right_fit_prev=None, 
                  show_all=False, verbose=False,
                  show_lanes=False):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # Returns the final output (image with lanes drawn on)
    global frame
    frame += 1
    
    if show_all: 
        print('Frame : ', str(frame),'-', type(image), 'with dimesions:', image.shape)
        show_image(image, 'Original Image')
    
    # distortion correct the image
    image = undistort(image)
    if show_all: show_image(image, 'Distortion Corrected Image')
    
    # threshold the image
    color, binary = pipeline(image, show_all=show_all)
    if show_all: 
        show_image(color , 'Thresholded Image (stacked)')
        show_image(binary, 'Thresholded Image', cmap='gray')

    # warp the image
    warped = warp(binary, M)
    if show_all: show_image(warped, 'Warped Image', cmap='gray')
    
    # crop away hood & outer edges, by turning off pixels
    cropped = crop_binary_image(warped,crop_left=CROP_LEFT, crop_right=CROP_RIGHT,
                                       crop_bot=CROP_BOT, crop_top=CROP_TOP)
    if show_all: show_image(cropped, 'Cropped', cmap='gray')        

    # Find the windows - from scratch
    if left_fit_prev is None:
        window_centroids = find_window_centroids(cropped,verbose=verbose)

        left_fit, right_fit, left_curverad, right_curverad, offset, viz_warped =\
            fit_polynomials_trough_pixels_in_lane_windows(
                                           cropped, 
                                           window_centroids, WINDOW_WIDTH, WINDOW_HEIGHT,
                                           visualize=True,verbose=verbose)
    else:
        # Use this during video processing !
        # Find the windows - in region around left_fit, right_fit of previous image
        left_fit, right_fit, viz_warped = find_new_fit(cropped, 
                                                       left_fit_prev, right_fit_prev,
                                                       visualize=True,verbose=verbose)
    
    if show_all or show_lanes: show_image(viz_warped, 'Warped Image with Lane Windows')
    
    # Draw the lanes on original image
    if left_fit is not None and right_fit is not None:
        final_image = draw_lanes_on_original_image(image,left_fit, right_fit,
                                                   left_curverad, right_curverad, offset,
                                                   text1="frame {0}".format(frame))
    else:
        final_image = image
        
    if show_all or show_lanes: show_image(final_image, 'Original Image with Lanes')
    
    return final_image, left_fit, right_fit, left_curverad, right_curverad, offset


# In[16]:

print('Testing the lane finding using process_image function')

# Make a list of test images

use_prev_fit = False
file_dir = "test_images"
file_dir_out = "test_images_output"
files = os.listdir(file_dir)
files = ['project_video_output0611.jpg']
#files = ['challenge_video_output0239.jpg']
#files = ['challenge_video_output0001.jpg']
#files = ['challenge_video_output0025.jpg']

#======================================================
# Project Video
use_prev_fit = False
file_dir = "test_images_project_video"
file_dir_out = "test_images_project_video_output"
files = os.listdir(file_dir)

#use_prev_fit=True
#files = ['project_video_output0001.jpg',
#         'project_video_output0002.jpg',
#         'project_video_output0003.jpg',
#         'project_video_output0004.jpg',
#         'project_video_output0005.jpg',
#         'project_video_output0006.jpg']

#use_prev_fit=False
#files = ['project_video_output0626.jpg',
#         'project_video_output0547.jpg',
#         'project_video_output0577.jpg',
#         'project_video_output0994.jpg',
#         'project_video_output1003.jpg',
#         'project_video_output1047.jpg']

#======================================================
# Challenge Video
#use_prev_fit = False
#file_dir = "test_images_challenge_video"
#file_dir_out = "test_images_challenge_video_output"
#files = os.listdir(file_dir)

#use_prev_fit=True
#files = ['challenge_video_output0001.jpg',
#         'challenge_video_output0002.jpg',
#         'challenge_video_output0003.jpg',
#         'challenge_video_output0004.jpg',
#         'challenge_video_output0005.jpg',
#         'challenge_video_output0006.jpg']
#files = ['challenge_video_output0484.jpg']

#use_prev_fit=False
#files = ['challenge_video_output0239.jpg']


# test process_image function...
frame=0
left_fit_prev=None
right_fit_prev=None
show_all   = False
verbose    = False
show_lanes = False
show_final = False
for file in tqdm(files):
    file_in=file_dir+"/"+file
    file_out=file_dir_out+"/"+file
    file_out2=file_dir_out+"/unwarped_with_lanes_"+file
    
    image = mpimg.imread(file_in)
    
    if show_lanes:
        show_image(image,title=file)
    
    final_image, left_fit, right_fit, left_curverad, right_curverad, offset = \
        process_image(image, 
                      left_fit_prev, right_fit_prev,
                      show_all=show_all, verbose=verbose,
                      show_lanes=show_lanes)
    plt.imsave(file_out,final_image)
    
    if use_prev_fit:
        left_fit_prev = left_fit
        right_fit_prev = right_fit
    
    if show_final:
        show_image(final_image,title=file)


# # Test on Videos
# 
# We can test our solution on three provided videos:
# - project_video.mp4
# - challenge_video.mp4
# - harder_challenge_video.mp4

# In[ ]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[ ]:

frame = 0
project_video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
project_video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time project_video_clip.write_videofile(project_video_output, audio=False)')


# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_video_output))


# # ... old stuff...

# In[ ]:

#reading in an image
image = mpimg.imread('test_images/straight_lines1.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[ ]:



    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# ## Test on Images
# 
# Now you should build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[ ]:

os.listdir("test_images/")


# run your solution on all test_images and make copies into the test_images directory).

# In[ ]:

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
def draw_lanes_on_image(file_in, file_out, show_all=True, show_final=True):
    ####################################################################
    #reading in an image
    image = mpimg.imread(file_in)
    print ("******************************************************************")
    print('Processing image from file: '+file_in)
    combo = process_image(image, show_all, show_final)
    
    plt.imshow(combo)
    plt.savefig(file_out)
    print('Processed image saved as: ' + file_out)
   
def process_image(image, show_all=False, show_final=False):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    global frame
    frame += 1
    
    print('Frame : ', str(frame),'-', type(image), 'with dimesions:', image.shape)
    if show_all: show_image(image, 'Original Image')
    
    ####################################################################
    # Make it gray scale
    gray = grayscale(image)
    if show_all: show_image(gray, 'Grayscale Image', cmap='gray')

    ####################################################################
    # Next we'll create a masked image with only region of interest (roi)
    roi_vertices = calculate_region_of_interest(image, show_all)   
    masked_gray = region_of_interest(gray, roi_vertices)
    if show_all: show_image(masked_gray, 'Grayscale Region of Interest', cmap='Greys_r')
    
    ####################################################################
    # "Equalize" the dark gray levels in region of interest, so the lighter
    # lines that represent the lanes are more pronounced, and the canny
    # edge detection will be more precise.
    # 
    # I did this to get challenge_frame_110 to work properly:
    # (-) The road is concrete (light)
    # (-) There are black tire skid marks that are detected by canny and
    #      mess it all up without this step
    #
    equalize_gray_level_in_roi(masked_gray)
    if show_all: show_image(masked_gray, 'Equalized gray level', cmap='Greys_r')
    
    ####################################################################
    edges = find_edges_with_canny(masked_gray)
    if show_all: show_image(edges, 'Canny Edges', cmap='Greys_r')
    
    ####################################################################
    # Find the lanes, using hough transform
    #
    # Define the Hough transform parameters
    rho             = 2          # distance resolution in pixels of the Hough grid
    theta           = np.pi/180  # angular resolution in radians of the Hough grid
    threshold       = 15         # minimum number of votes (intersections in Hough grid cell)
    min_line_len    = 10         # minimum number of pixels making up a line
    max_line_gap    = 5          # maximum gap in pixels between connectable line segments
    
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None :
        print ('ERROR: No hough lines found')
        return image
    
    if show_all:
        line_img = np.zeros((edges.shape[0], edges.shape[1], 3), 
                            dtype=np.uint8)
        draw_lines(line_img, lines)
        combo = weighted_img(line_img, image, α=0.8, β=1., λ=0.)
        show_image(combo, 'Hough Lines on Image')

    ####################################################################
    # Draw the lanes on the original image    
    line_img = np.zeros((image.shape[0], image.shape[1], 3), 
                        dtype=np.uint8)
    create_lanes_from_hough_lines(line_img, roi_vertices, lines, thickness=10)
    combo = weighted_img(line_img, image, α=0.8, β=1., λ=0.)
    if show_all or show_final: show_image(combo, 'Final Image with Lanes')

    return combo

def show_image(image, title, cmap=None ):
    plt.title(title)
    if cmap:
        plt.imshow(image, cmap=cmap) # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    else:
        plt.imshow(image)  
    plt.show()
    
def calculate_region_of_interest(image, show_all):
    imshape = image.shape
    
    # height of roi region
    if roi_height == 0:
        height = int( 0.4*imshape[0] )
    else:
        height = roi_height
        
    # bottom width of roi region
    if roi_width_bot == 0:
        width_bot = int( 0.80 * imshape[1] )
    else:
        width_bot = roi_width_bot
        
    # top width of roi region
    if roi_width_top == 0:
        width_top = int( 0.20 * imshape[1] )
    else:
        width_top = roi_width_top
      
    x_bot_min = roi_shift_bot + int( 0.5 * (imshape[1] - width_bot) )
    x_bot_max = roi_shift_bot + int( 0.5 * (imshape[1] + width_bot) )
    x_top_min = roi_shift_top + int( 0.5 * (imshape[1] - width_top) )
    x_top_max = roi_shift_top + int( 0.5 * (imshape[1] + width_top) )
    
    # eliminate car hood from picture
    y_bot = imshape[0] - hood_size
    y_top = y_bot - height
    
    roi_vertices = np.array([[(x_bot_min,y_bot),
                              (x_top_min, y_top), 
                              (x_top_max, y_top), 
                              (x_bot_max,y_bot)]], dtype=np.int32)
    
    if show_all:
        plt.title('Region of interest on Image. Wb,Wt,H,Sb,St,Sv='+
                  str(x_bot_max-x_bot_min)+','+
                  str(x_top_max-x_top_min)+','+
                  str(y_top-y_bot)+','+
                  str(roi_shift_bot)+','+
                  str(roi_shift_top)+','+
                  str(hood_size))
        line_img = np.zeros((image.shape[0], image.shape[1], 3), 
                            dtype=np.uint8)
        p=roi_vertices[0]
        roi_lines = np.array([[[ p[0, 0], p[0, 1], p[1, 0], p[1, 1] ]],
                              [[ p[1, 0], p[1, 1], p[2, 0], p[2, 1] ]],
                              [[ p[2, 0], p[2, 1], p[3, 0], p[3, 1] ]],
                              [[ p[3, 0], p[3, 1], p[0, 0], p[0, 1] ]]], 
                             dtype=np.int32)
        draw_lines(line_img, roi_lines)
        combo = weighted_img(line_img, image, α=0.8, β=1., λ=0.)
        plt.imshow(combo)
        plt.show()
        
    return roi_vertices

def equalize_gray_level_in_roi(masked_gray): 
    # calculate the average gray level in the region of interest   
    gray_average = int( np.average(masked_gray[np.nonzero(masked_gray)]) )
    
    # assign this average gray level to all regions that are darker than this
    # average value
    masked_gray[ masked_gray < gray_average ] = gray_average

def find_edges_with_canny(masked_gray):    
    # See: http://homepages.inf.ed.ac.uk/rbf/HIPR2/canny.htm
    #
    # Define a kernel size for Gaussian smoothing / blurring
    # Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
    kernel_size = 5
    gaussian_blur(masked_gray, kernel_size)
    
    # Define parameters for Canny and run it
    # thresholds between 0 and 255
    low_threshold = 50
    high_threshold = 150
    edges = canny(masked_gray, low_threshold, high_threshold)
    
    return edges 
   
def create_lanes_from_hough_lines(img, roi_vertices, lines, color=[255, 0, 0], thickness=None):
    """
    This routine attempts to extract the left and right lane from the provided
    hough lines, and draws them on the image.
    
    The method used is as follows:
    (-) each line is extrapolated to the top and bottom of the region of interest
    (-) it will be discarded if the intersection at the top and bottom is outside
        the region of interest.
    (-) the remaining lines are assigned to the left lane if the angle is negative,
        else they are assigned to the right lane
    (-) the locations of the line intersections with the top and bottom of 
        the region of interest are averaged to provide the lane end points
        
    NOTE: This logic ONLY works if the lanes within the region of interest are
          straight.
    """
    #
    # extrapolate start and end points for each line to boundary of region of interest
    #
    y_top_roi    = roi_vertices[0, 1, 1]
    y_bot_roi    = roi_vertices[0, 0, 1]
    
    x_top_roi_min   = roi_vertices[0, 1, 0]
    x_top_roi_max   = roi_vertices[0, 2, 0]
    x_bot_roi_min   = roi_vertices[0, 0, 0]
    x_bot_roi_max   = roi_vertices[0, 3, 0]
    
    x_tops_left   = []
    x_bots_left   = []
    
    x_tops_right   = []
    x_bots_right   = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
        
            # fit a line (y=Ax+B) through this line
            # np.polyfit() returns the coefficients [A, B] of the fit
            fit_line = np.polyfit((x1, x2), (y1, y2), 1)
            A, B = fit_line
            
            # skip lines with very small slope (A)
            if abs(A) < 0.1:
                continue
            
            # calculate intersection with top & bottom boundary of region of interest
            x_top = (y_top_roi - B) / A
            x_bot = (y_bot_roi - B) / A
            
            # skip this line if either the top or bottom points are outside region of interest
            if x_top < x_top_roi_min or x_top > x_top_roi_max:
                continue
            if x_bot < x_bot_roi_min or x_bot > x_bot_roi_max:
                continue
            
            # negative angle line --> left lane
            if A < 0:
                x_tops_left.append(x_top)
                x_bots_left.append(x_bot)
            else:
                x_tops_right.append(x_top)
                x_bots_right.append(x_bot)
    
    # if we didn't find a top or bottom, then skip this image
    if (len(x_tops_left) == 0 or
        len(x_bots_left) == 0 or
        len(x_tops_right) == 0 or
        len(x_bots_right) == 0 ):
        print ('Not all end points found --> skipping this image !!')
        return
    
    x_top_left_average  = int( np.array(x_tops_left).mean()  )
    x_top_right_average = int( np.array(x_tops_right).mean() )
    x_bot_left_average  = int( np.array(x_bots_left).mean()  )
    x_bot_right_average = int( np.array(x_bots_right).mean() )
    
    x_top_left_width  = max(x_tops_left)  - min(x_tops_left)
    x_top_right_width = max(x_tops_right) - min(x_tops_right)
    x_bot_left_width  = max(x_bots_left)  - min(x_bots_left)
    x_bot_right_width = max(x_bots_right) - min(x_bots_right)
    
    if thickness == None:
        thickness = int( 0.5*sum( [x_top_left_width ,
                                   x_top_right_width,
                                   x_bot_left_width ,
                                   x_bot_right_width] ) / 4 )
    
    # plot left lane                 
    cv2.line(img, (x_bot_left_average, y_bot_roi), 
                  (x_top_left_average, y_top_roi), color, thickness)
    
    # plot right lane                 
    cv2.line(img, (x_bot_right_average, y_bot_roi), 
                  (x_top_right_average, y_top_roi), color, thickness)

if __name__ == '__main__':
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML
    
    ################################################################
    # process some test files
    #
    file_dir = "test_images"
    file_dir_out = "test_images_output"
    files = os.listdir(file_dir)
 
    frame = 0
    hood_size=45
    roi_height=200
    roi_width_bot=1200
    roi_width_top=500
    roi_shift_bot=0
    roi_shift_top=0
    for file in files:
        frame = 0
        draw_lanes_on_image(file_in=file_dir+"/"+file, 
                            file_out=file_dir_out+"/with_lanes_"+file,
                            show_all=False, show_final=True)
          
    


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on three provided videos:
# 
# `project_video.mp4`
# 
# `challenge_video.mp4`
# 
# `harder_challenge_video.mp4`
# 
# **Note: if you get an `import error` when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt. Also, check out [this forum post](https://carnd-forums.udacity.com/questions/22677062/answers/22677109) for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://carnd-forums.udacity.com/display/CAR/questions/26218840/import-videofileclip-error) for more troubleshooting tips across operating systems.**

# In[ ]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[ ]:

# def process_image is coded up above


# Let's try the project_video.mp4 first ...

# In[ ]:

frame = 0
hood_size=45
roi_height=200
roi_width_bot=1200
roi_width_top=500
roi_shift_bot=0
roi_shift_top=0
project_video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
project_video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time project_video_clip.write_videofile(project_video_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_video_output))


# **At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[ ]:

frame = 0
hood_size=0
roi_height=200
roi_width_bot=810
roi_width_top=150
roi_shift_bot=45
roi_shift_top=15
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# HTML("""
# <video width="960" height="540" controls>
#   <source src="{0}">
# </video>
# """.format(yellow_output))

# # Reflections
# 
# Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail?
# 
# Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!
# 
# ---
# My pipeline was largely created based on frame 110 of the challenge video:
# <figure>
#  <img src="challenge_frame_110_image-0.jpg" width="380" alt="Original" />
# </figure>
# 
# The steps take are best explained by this sequence of pictures:
# 
# <figure>
#  <img src="challenge_frame_110_image-1.jpg" width="380" alt="Grayscale" />
# </figure>
# 
# <figure>
#  <img src="challenge_frame_110_image-2.jpg" width="380" alt="ROI" />
# </figure>
# 
# <figure>
#  <img src="challenge_frame_110_image-3.jpg" width="380" alt="Gray Scale ROI" />
# </figure>
# 
# <figure>
#  <img src="challenge_frame_110_image-4.jpg" width="380" alt="Equalized" />
# </figure>
# 
# <figure>
#  <img src="challenge_frame_110_image-5.jpg" width="380" alt="Canny Edges" />
# </figure>
# 
# <figure>
#  <img src="challenge_frame_110_image-6.jpg" width="380" alt="Hough Lines" />
# </figure>
# 
# <figure>
#  <img src="challenge_frame_110_image-7.jpg" width="380" alt="Final Image" />
# </figure>
#  
# Some key decisions I made along the way:
# 
# 1. I parameterized the inputs for the region of interest (roi):
#         hood_size
#         roi_height
#         roi_width_bot
#         roi_width_top
#         roi_shift_bot
#         roi_shift_top
#     
#     My logic  strongly depends on having a well defined region of interest.
# 
# 2. I 'equalize' the grey level in the region of interest.
# ...This greatly improves the visibility of the lanes in the image.
# ...See function: equalize_gray_level_in_roi
#     
# 3. All images and videos have straight lanes in the region of interest. I make use of this fact, in two ways:
# ..* To select what hough lines belong to the lanes, I extrapolate them to the top and bottom of the region of interest. If they intersect outside of the region of interest then they obviously are not part of the lanes.
# 
# ..* To determine the end points of the lanes, I simply average the intersection points of the selected lines with the top and bottom of the region of interest. The lanes are then drawn with a fixed thickness.
# 
# ## Weaknesses
# 
# The weakest aspects of my approach are:
# 
# * The requirement to manually set the region of interest. If the car drifts, I can see that we lose the lanes very easily.
# 
# * The 'equalizing' of gray level in the region of interest does not seem so robust. It works ok for all tests, but I can imagine it will fail on other cases.
# 
# * The logic only works on straight lines.
# 

# ## Submission
# 
# If you're satisfied with your video outputs it's time to submit!  Submit this ipython notebook for review.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[ ]:

frame = 0
hood_size=55
roi_height=200
roi_width_bot=840
roi_width_top=180
roi_shift_bot=35
roi_shift_top=25
challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# In[ ]:



