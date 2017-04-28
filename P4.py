
# coding: utf-8

# # **Advanced Finding of Lane Lines on the Road** 
# ***
# In this project, I am applying the tools learned to identify lane lines on the road, using following steps: 
# 1. Camera calibration
# 2. Distortion correction
# 3. Color & Gradient threshold
# 4. Perspective transform
# 5. Detect lane lines
# 6. Determine the lane curvature

# In[1]:

I_AM_IN_JUPYTER = True
SCRATCH_IMAGE_DIR = 'C:\\Work\\ScratchImages'  # only used when exporting into .py, and setting I_AM_IN_JUPYTER=False
SCRATCH_IMAGE_NUM = 0

if I_AM_IN_JUPYTER:
    get_ipython().magic('matplotlib inline')
else:
    # use non-interactive back-end to avoid images from popping up
    # See: http://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib-so-it-can-be
    from matplotlib import use
    use('Agg') 
        
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
from collections import deque
import glob


# In[2]:

# some global utility functions, mainly for visualization

# function to show a plot or write it to disk, depending if I am running in a jupyter notebook or not
def my_plt_show():
    global I_AM_IN_JUPYTER, SCRATCH_IMAGE_NUM, f_html, f_url
    plt.show()
    if I_AM_IN_JUPYTER == False:
        # at start
        if SCRATCH_IMAGE_NUM == 0:
            # clean out the scratch image dir
            files = glob.glob(SCRATCH_IMAGE_DIR+'\\*')
            for f in files:
                os.remove(f)  
            # open 'all.html' that displays all the images written
            f_html = open(SCRATCH_IMAGE_DIR+'\\all.html', 'w')
            f_url  = 'file:///'+SCRATCH_IMAGE_DIR+'\\all.html'
            f_html.write('<html>\n')
            # webbrowser.open_new(f_url) # open it in new window of default web-browser
            
        # save all images to a scratch dir
        fname = 'img_{:04d}.jpg'.format(SCRATCH_IMAGE_NUM)
        plt.savefig(SCRATCH_IMAGE_DIR+'\\'+fname)
        fig = plt.gcf() # get reference to the current figure
        plt.close(fig)  # and close it
        f_html.write('<img src="'+fname+'" /> <br />\n') 
        f_html.flush() # flush it directly to disk, for debug purposes.    
        # webbrowser.open(f_url, new=0) # refresh the page        
        SCRATCH_IMAGE_NUM += 1
    plt.gcf().clear() # clear the fig

# function to show an image with title
def show_image(image, title, cmap=None ):
    plt.gcf().clear() # clear the fig
    if I_AM_IN_JUPYTER:
        fig, ax = plt.subplots(1, 1, figsize=(24, 10))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.title(title)
    if cmap:
        plt.imshow(image, cmap=cmap) # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    else:
        plt.imshow(image)  
    my_plt_show()


# function to plot a number of histograms at y-locations
def show_histogram(img, y_histograms = [475,600], img_name=' '):
    for y_histogram in y_histograms:
        histogram = img[y_histogram,:]
        plt.gcf().clear() # clear the fig
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))        
        plt.plot(histogram) 
        plt.title('histogram at y='+str(y_histogram)+' -'+img_name)
        my_plt_show()
        
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
    my_plt_show()
    
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

print('Testing the Calibrated Camera')

# Make a list of test images
file_dir = "camera_cal"
file_dir_out = "camera_cal_output"
files = os.listdir(file_dir)

for file in files:
    file_in=file_dir+"/"+file
    file_out=file_dir_out+"/Distortion_Correction_"+file
    
    image = mpimg.imread(file_in)
    
    dst = undistort(image)
    
    plot_orig_and_changed_image(image1=image , description1=file,
                               image2=dst, description2='Distortion Corrected',
                               file_out=file_out)


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

def find_edges_with_canny(img, thresholds=(0,1)):    
    # See: http://homepages.inf.ed.ac.uk/rbf/HIPR2/canny.htm
    #
    # Define a kernel size for Gaussian smoothing / blurring
    # Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
    #kernel_size = 5
    #masked_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Define parameters for Canny and run it
    # thresholds between 0 and 1 for a binary image !
    edges = cv2.Canny(img, thresholds[0], thresholds[1])
    
    # if binary, scale it back to binary
    if thresholds[1] ==1:
        edges = (edges/255).astype(np.int32)
    
    return edges 

# function to Adaptive Gaussian Thresholding
# See - http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
def adaptive_gaussian_threshold(gray, thresholds=(0, 255)):
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray,(7,7),0)
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
    
    if show_all:
        show_image(scaled_sobel,      title='sobel_thresh: '+orient, cmap='gray')
        if show_histograms:
            show_histogram(scaled_sobel, y_histograms=[475, 600], img_name='sobel_thresh') 
    
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return binary_output

def mag_thresh(img, ksize=5, thresh=(0, 255), rgb_in=True):
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
    if show_all:
        show_image(gradmag,      title='mag_thresh: ', cmap='gray')
        if show_histograms:
            show_histogram(gradmag, y_histograms=[475, 600], img_name='mag_thresh') 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

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

def blacken_or_whiten_MxN_squares(gray, target='blacken', MxN=(1,1), EQFACT=1.0, method='average' ): 
    # MxN = (M rows,N Columns)
    #   ----> j (N Columns)
    #  |
    #  |
    # \ /
    #  i (M Rows)
    #
    # target = 'blacken', 'whiten' or 'average_below' or 'average_above'
    # method = 'average', 'max'
    debug=False
    eql = np.copy(gray)
    M = MxN[0] # number or rows within image
    N = MxN[1] # number of columns within image
    # calculate the average gray in N squares 
    x_h = int(gray.shape[1]/N)
    y_h = int(gray.shape[0]/M)
    x_f = int(gray.shape[1]  )
    y_f = int(gray.shape[0]  )
    for i in range(M+1):        # do 1 extra, to handle left over pixels
        y_start = i*y_h
        y_end   = min( (i+1)*y_h, y_f )
        # if there is a perfect fit of pixels, the left over loop is not needed
        if y_start >= y_end:
            # we're done....
            continue
        for j in range(N+1):    # do 1 extra, to handle left over pixels
            x_start = j*x_h
            x_end   = min( (j+1)*x_h, x_f )
            
            #if i==M-1:
            #    print('i='+str(i))
            #    print('j='+str(j))
            #    print('x_start='+str(x_start))
            #    print('x_end  ='+str(x_end  ))
            #    print('y_start='+str(y_start))
            #    print('y_end  ='+str(y_end  ))
            
            # if there is a perfect fit of pixels, the left over loop is not needed
            if x_start >= x_end:
                # we're done....
                continue
            
            gij = np.array(gray[y_start:y_end, x_start:x_end])
            nonzeros = gij[np.nonzero(gij)]
            if len(nonzeros)>0:
                gij_average = int( np.mean(nonzeros) ) # average over the non-zeros
            else:
                gij_average = 0
                
            
            gij_eql = np.copy(gij)
            
            if debug and i==0 and j==0:
                show_image(gij_eql,      title='MxN '+str(M)+'x'+str(N), cmap='gray')
                #histogram = gij_eql[300,:]
                #plt.plot(histogram) 
                #my_plt_show()             
    
            # 
            if method=='max': 
                # assign new level to all regions that are darker than factor*max_level
                gij_max = int( np.max(gij) )
                if target == 'average_below':
                    gij_eql[ gij < EQFACT*gij_max ] = gij_average
                elif target == 'average_above':
                    gij_eql[ gij > EQFACT*gij_max ] = gij_average
                elif target == 'blacken':   
                    gij_eql[ gij < EQFACT*gij_max ] = 0
                elif target == 'whiten':   
                    gij_eql[ gij > EQFACT*gij_max ] = 255
            else:
                # method='average'
                # assign new level to all regions that are darker than factor*average
                if target == 'average_below':
                    gij_eql[ gij < EQFACT*gij_average ] = gij_average
                elif target == 'average_above':
                    gij_eql[ gij > EQFACT*gij_average ] = gij_average
                elif target == 'blacken':   
                    gij_eql[ gij < EQFACT*gij_average ] = 0
                elif target == 'whiten':   
                    gij_eql[ gij > EQFACT*gij_average ] = 255
                
            if debug and i==0 and j==0:
                show_image(gij_eql,      title='NxN ', cmap='gray')
                #histogram = eql22[300,:]
                #plt.plot(histogram) 
                #my_plt_show() 
    
            eql[y_start:y_end, x_start:x_end] = gij_eql
            
    if debug:
        show_image(eql,      title='Equalize ', cmap='gray')
        histogram = eql[600,:]
        plt.plot(histogram) 
        my_plt_show()                
    
    return eql

def pipeline(img, show_all=False):
    img = np.copy(img)
    
    # ============================================================================
    # Combine some gradient thresholds
    
    # grayscale the unwarped image
    eql=grayscale(img)
    if show_histograms:
        show_histogram(eql, y_histograms=[475, 600], img_name='Grayscale ')
    
    # Remove black lines of road repairs from image
    eql_BL = eql
    
    NUM_BLACKEN = 3
    M_START = 2
    N_START = 2
    M_END   = 2
    N_END   = 2
    BLACK_FACT = 1.0
    BLACK_METHOD = 'average'
    
    NUM_GRAY_GRAD_BLACKEN = NUM_BLACKEN
    for i in range(NUM_GRAY_GRAD_BLACKEN):
        # blacken on progressively larger windows
        # note: not used here.., since M & N stay constant, but it was added as a interesting trial
        M=max(int(M_START/(2**i)),M_END)
        N=max(int(N_START/(2**i)),N_END)
        eql_BL = blacken_or_whiten_MxN_squares(np.array(eql_BL).astype(np.uint8), 
                                               target='blacken', 
                                               MxN=(M,N), EQFACT=BLACK_FACT, method=BLACK_METHOD)
        if show_all:
            show_image(eql_BL, title='Blackened Grayscale - MxN= '+str(M)+'x'+str(N), cmap='gray')
        if show_histograms:
            show_histogram(eql_BL, y_histograms = [475,600], img_name='Blackened Grayscale')
    
    # Apply each of the thresholding functions
    # Choose a Sobel kernel size, used by xgrad, ygrad and mag
    # Choose a larger odd number to smooth gradient measurements
    ksize = 31
    xgrad_thresh=(50  , 100)
    ygrad_thresh=(50  , 100)
    mag__thresh =(50  , 100)
    dir_ksize   = 15
    dir_thresh  =(0.7 , 1.3)
    
    gradx = abs_sobel_thresh(eql, orient='x', ksize=ksize, thresh=xgrad_thresh, rgb_in=False)
    grady = abs_sobel_thresh(eql, orient='y', ksize=ksize, thresh=ygrad_thresh, rgb_in=False)
    mag_binary = mag_thresh(eql, ksize=ksize, thresh=mag__thresh, rgb_in=False)
    # Note: threshold is angle: 0 = horizontal; +/- np.pi/2 = vertical
    dir_binary = dir_threshold(eql, ksize=dir_ksize, thresh=dir_thresh, rgb_in=False)

    if show_all:
        show_image(gradx,      title='Thresholded X-Gradient', cmap='gray')
        show_image(grady,      title='Thresholded Y-Gradient', cmap='gray')
        show_image(mag_binary, title='Thresholded Magnitude' , cmap='gray')
        show_image(dir_binary, title='Thresholded Grad. Dir.', cmap='gray')

    # Combined thresholds
    grad_binary_1 = np.zeros_like(gradx)
    grad_binary_1[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
  
    grad_binary = np.zeros_like(gradx)
    grad_binary[(( eql_BL != 0) & (grad_binary_1 == 1))] = 1

    if show_all:
        show_image(grad_binary_1, title='combined Grad Binary', cmap='gray')
        show_image(grad_binary, title='combined Grad Binary exluding blackened', cmap='gray')
    
    #grad_binary = adaptive_gaussian_threshold(grad_binary, thresholds=(0, 1))
    # Alternative use morpohologyEx instead -- this is more aggressive. Removes more
    # See: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    kernel = np.ones((3,3),np.uint8)
    grad_binary = cv2.morphologyEx(grad_binary, cv2.MORPH_OPEN, kernel)
    if show_all:
        show_image(grad_binary, title='de-noised final Grad Binary', cmap='gray')
      
    # ============================================================================
    # Threshold the HSV space
    h_thresh=(  0,  90)
    s_thresh=(  1, 255) # rely on blackening of s-channel
    v_thresh=(215, 255)
       
    # Convert to HSV color space and separate the S channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    h_channel = hsv[:,:,0]   # [0-179]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]

    if show_histograms:
        show_image(h_channel,      title='h-hsv_channel - checking histogram at 600 ', cmap='gray')
        show_histogram(h_channel, y_histograms=[475, 600], img_name='h-hsv_channel ')
        show_image(s_channel,      title='s-hsv_channel - checking histogram at 600 ', cmap='gray')
        show_histogram(s_channel, y_histograms=[475, 600], img_name='s-hsv_channel ')
        show_image(v_channel,      title='v-hsv_channel - checking histogram at 600 ', cmap='gray')
        show_histogram(v_channel, y_histograms=[475, 600], img_name='v-hsv_channel ')
        
    # blacken
    NUM_BLACKEN = 3
    M_START = 1
    N_START = 1
    M_END   = 1
    N_END   = 1
    BLACK_FACT = 1.0
    BLACK_METHOD = 'average'
    
    NUM_S_HSV_BLACKEN = NUM_BLACKEN  
    for i in range(NUM_S_HSV_BLACKEN):
        M=max(int(M_START/(2**i)),M_END) 
        N=max(int(N_START/(2**i)),N_END)
        s_channel = blacken_or_whiten_MxN_squares(np.array(s_channel).astype(np.uint8), 
                                                  target='blacken', 
                                                  MxN=(M,N), EQFACT=BLACK_FACT, method=BLACK_METHOD)
        if show_histograms:
            show_image(s_channel, title='Blackened s-hsv-channel - MxN= '+str(M)+'x'+str(N), cmap='gray')
            show_histogram(s_channel, y_histograms=[475, 600], img_name='Blackened s-hsv_channel ')

    CLAHE_CLIP_LIMIT = 0.01
    CLAHE_TILE_GRIDSIZE = (16,16)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRIDSIZE)
    
    eql = np.array(v_channel,dtype='uint8')
    eql = clahe.apply(eql)
    if show_all:
        show_image(eql, title='CLAHEd v-hsv-channel ', cmap='gray')
        histogram = eql[600,:]
        plt.plot(histogram) 
        my_plt_show()
    v_channel = eql
    
    # Threshold color channels
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
    if show_all:
        show_image(v_binary,   title='Thresholded s-hsv channel', cmap='gray')
        show_image(v_binary,   title='Thresholded v-hsv channel', cmap='gray')
    
    # get rid of large smudges, eg. 3-0403
    if USE_CANNY:
        v_binary = find_edges_with_canny(v_binary)
        if show_all:
            show_image(v_binary,   title='Canny edges of v-hsv channel', cmap='gray')
        if show_histograms:
            show_histogram(v_binary, y_histograms = [475,600], img_name='Canny edges of v-hsv channel') 
    
    hsv_binary = np.zeros_like(v_channel)
    hsv_binary[((s_binary == 1) | (v_binary == 1)) & (h_binary == 1)] = 1
    
    if show_all:
        show_image(hsv_binary, title='Final hsv binary', cmap='gray')
    
    # ============================================================================
    # Threshold HLS space
    h_thresh=(  0,  90)
    l_thresh=(100, 255)
    s_thresh=(  1, 255) # rely on blackening of s-channel
    
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]   # [0-179]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    if show_histograms:
        show_image(h_channel,      title='h-hls_channel - checking histogram at 600 ', cmap='gray')
        show_histogram(h_channel, y_histograms=[475, 600], img_name='h-hls_channel ')
        show_image(l_channel,      title='l-hls_channel - checking histogram at 600 ', cmap='gray')
        show_histogram(l_channel, y_histograms=[475, 600], img_name='l-hls_channel ')
        show_image(s_channel,      title='s-hls_channel - checking histogram at 600 ', cmap='gray')
        show_histogram(s_channel, y_histograms=[475, 600], img_name='s-hls_channel ')
    
    NUM_BLACKEN = 3
    M_START = 1
    N_START = 1
    M_END   = 1
    N_END   = 1
    BLACK_FACT = 1.0
    BLACK_METHOD = 'average'
    
    NUM_S_HLS_BLACKEN = NUM_BLACKEN
    for i in range(NUM_S_HLS_BLACKEN):
        M=max(int(M_START/(2**i)),M_END) 
        N=max(int(N_START/(2**i)),N_END)                
        s_channel = blacken_or_whiten_MxN_squares(np.array(s_channel).astype(np.uint8), 
                                                  target='blacken', 
                                                  MxN=(M,N), EQFACT=BLACK_FACT, method=BLACK_METHOD)
        if show_all:
            show_image(s_channel, title='Blackened s-hls-channel - MxN= '+str(M)+'x'+str(N), cmap='gray')
        if show_histograms:
            show_histogram(s_channel, y_histograms = [475,600], img_name='Blackened s-hls-channel')
    
    # Threshold color channels
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    if show_all:
        show_image(s_binary,   title='Thresholded s-hls channel', cmap='gray')
    
    # get rid of large smudges, eg. 3-0403
    if USE_CANNY:
        s_binary = find_edges_with_canny(s_binary)
        if show_all:
            show_image(s_binary,   title='Canny edges of s-hls channel', cmap='gray')
        if show_histograms:
            show_histogram(s_binary, y_histograms = [475,600], img_name='Canny edges of s-hls channel')        
    
    hls_binary = np.zeros_like(h_channel)
    hls_binary[((h_binary == 1) & (s_binary == 1)) & (l_binary == 1)] = 1
    
    if show_all:
        show_image(hls_binary, title='Final hls channel', cmap='gray')
        show_image(hls_binary, title='de-noised final HLS Binary', cmap='gray')
                
    # ============================================================================d
    # Stack each channel
    color_binary = np.dstack((grad_binary, hsv_binary, hls_binary))
    if show_all:
        show_image(color_binary,  title='stacked binaries in color')
    
    # Combine the binary thresholds
    combined_binary = np.zeros_like(grad_binary)
    combined_binary[(grad_binary == 1) | (hsv_binary == 1) | (hls_binary == 1) ] = 1  
    if show_all:
        show_image(combined_binary,  title='combined_binary', cmap='gray')
        
    return color_binary, combined_binary


# # 4. Perspective transform

# In[8]:

# Width of target Image for perspective transform
IWP = 2 # Image Width Perspective (IWP) factor

# Interactively find the 4 corner points that will be used as the source points
file_in="test_images/straight_lines1.jpg"   
#file_in="test_images/challenge_video_output0001.jpg"
#file_in="test_images/harder_challenge_video_output0001.jpg"
#file_in="test_images/harder_challenge_video_output0193.jpg"
img = mpimg.imread(file_in)

image = np.copy(img)
# distortion correct the image
image = undistort(image)

# source points
p1 = ( 723,            475) # top right
p2 = (1110, image.shape[0]) # bottom right
p3 = ( 204, image.shape[0]) # bottom left
p4 = ( 562,            475) # top left

# draw source lines on the image in red
src_lines = [[p1,p2], [p2,p3], [p3,p4], [p4,p1]]
draw_lines_on_image(image, src_lines, color=[255, 0, 0], thickness=2)

# destination points
correctX = 100 # To get straight line straight...
dp1 = (p2[0]-correctX,              0) # top right
dp2 = (p2[0]-correctX, image.shape[0]) # bottom right
dp3 = (p3[0]+correctX, image.shape[0]) # bottom left
dp4 = (p3[0]+correctX,              0) # top left

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
if IWP == 1:
    # image size of perspective the same
    shiftX = 0
else:
    # image size of perspective is IWP factor wider 
    # shift destination lines to the center of target image
    shiftX = int(0.25*IWP*img.shape[1])
dp1 = (dp1[0]+shiftX, dp1[1])
dp2 = (dp2[0]+shiftX, dp2[1])
dp3 = (dp3[0]+shiftX, dp3[1])
dp4 = (dp4[0]+shiftX, dp4[1])
    
dst = np.float32([dp1,dp2,dp3,dp4])


# compute the perspective transform, M
M = cv2.getPerspectiveTransform(src, dst)

# compute the inverse perspective transform, Minv
Minv = cv2.getPerspectiveTransform(dst, src)

# Define a function that does the actual warping
def warp(img):
    if IWP == 1:
        # image size of perspective the same
        img_size = (img.shape[1], img.shape[0])
    else:
        # image size of perspective is IWP factor wider 
        img_size = (IWP*img.shape[1], img.shape[0])
        
    # create warped image, using linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)    
    return warped

# Define a function that does the actual unwarping
def unwarp(img):
    if IWP == 1:
        # image size of perspective the same
        img_size = (img.shape[1], img.shape[0])
    else:
        # image size of perspective is IWP factor wider, so scale it back during unwarp
        img_size = (int(img.shape[1]/IWP), img.shape[0])
        
    # create warped image, using linear interpolation
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)    
    return unwarped

# As a test, warp the image, show the grid
image = np.copy(img)
# draw source lines on the image in red
draw_lines_on_image(image, src_lines, color=[255, 0, 0], thickness=2)
# now warp it
warped = warp(image)
# draw destination lines on the warped image in green
# shift dst_lines to center of target image to take care of different sizes (IWP)
for i in range(len(dst_lines)):
    p1=dst_lines[i][0]
    p2=dst_lines[i][1]
    p1 = (p1[0]+shiftX, p1[1]) # shift first point of line in X direction
    p2 = (p2[0]+shiftX, p2[1]) # shift second point of line in X direction
    dst_lines[i][0] = p1
    dst_lines[i][1] = p2
draw_lines_on_image(warped, dst_lines, color=[0, 255, 0], thickness=2)
fig = plt.figure(figsize=(15,15))
ax = fig.gca()
ax.set_xticks(np.arange(0, warped.shape[1], 100))
ax.set_yticks(np.arange(0, warped.shape[0], 50))
plt.imshow(warped)
plt.grid(True)


# # 5. Detect lane lines

# In[9]:

# Parameters that control the sliding windows convolution
WINDOW_WIDTH = 80 
WINDOW_HEIGHT = 40
MARGIN = 100 # How much to slide left and right for searching

# acceptance criteria for windows during search over pixels in warped image.
MIN_CONVSIGNAL = 10  # if window of convolution has signal lower than this value, it will be rejected


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, verbose=False):
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
    
    # window_height must be equal to hood section we skip !
    window_width = WINDOW_WIDTH
    window_height = WINDOW_HEIGHT
    margin = MARGIN
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    #l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    #l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    #r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    #r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Keep it simpler
    l_center = LEFT_START 
    r_center = RIGHT_START
    
    if verbose:
        print('initial l_center, r_center = {0}, {1}'.format(l_center, r_center))
    
    midpoint = int(image.shape[1]/2)
    if l_center == 0:
        l_center = 0.5*midpoint
    if r_center == 0:
        r_center = midpoint + 0.5*midpoint
    
    if verbose:
        print('initial midpoint, l_center, r_center = {0}, {1} {2}'.format(midpoint,l_center, r_center))
        
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
    #patch for level in range(0,(int)(image.shape[0]/window_height)):
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
        if l_convsignal < MIN_CONVSIGNAL:
            l_center = -abs(window_centroids[-1][0])
            l_num_rejected_sequence += 1
        else:
            l_num_rejected_sequence = 0
            
        if r_convsignal < MIN_CONVSIGNAL:
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
def find_pixels_in_lane_windows(binary_img_in, 
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

    return left_lane_inds, right_lane_inds, out_img

def find_pixels_around_best_fit(warped,
                 visualize=True,verbose=False): 
    global l_Line, r_Line
    
    left_lane_inds  = []
    right_lane_inds = []

    # activate best fit pixels in warped image
    if ACTIVATE_BEST_FIT_PIXELS:
        if l_Line.best_fit is not None:
            warped[l_Line.best_yfit, l_Line.best_xfit] = 1
        if r_Line.best_fit is not None:
            warped[r_Line.best_yfit, r_Line.best_xfit] = 1
    
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = MARGIN
    if l_Line.best_fit is not None:
        left_lane_inds = ((nonzerox > (l_Line.best_fit[0]*(nonzeroy**2) + l_Line.best_fit[1]*nonzeroy + l_Line.best_fit[2] - margin)) & (nonzerox < (l_Line.best_fit[0]*(nonzeroy**2) + l_Line.best_fit[1]*nonzeroy + l_Line.best_fit[2] + margin))) 
    if r_Line.best_fit is not None:
        right_lane_inds = ((nonzerox > (r_Line.best_fit[0]*(nonzeroy**2) + r_Line.best_fit[1]*nonzeroy + r_Line.best_fit[2] - margin)) & (nonzerox < (r_Line.best_fit[0]*(nonzeroy**2) + r_Line.best_fit[1]*nonzeroy + r_Line.best_fit[2] + margin)))  
    
    if visualize:   
        # If requested, create & return a new image that visualizes the process
        result_img = None
        
        # Again, extract left and right line pixel positions &
        left_fit = None
        right_fit = None
        if len(left_lane_inds) >0:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            if len(leftx)>0 and len(lefty)>0:
                left_fit = np.polyfit(lefty, leftx, 2)
        if len(right_lane_inds) > 0:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            if len(rightx)>0 and len(righty)>0:
                right_fit = np.polyfit(righty, rightx, 2)
        
        ploty, left_fitx, right_fitx = generate_xy_data_for_plotting_polynomials(warped,
                                                                                 left_fit,
                                                                                 right_fit)
        
        out_img = binary_to_rgb(warped)
        window_img = np.zeros_like(out_img)

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
    else:
        result_img = warped
        
    return left_lane_inds, right_lane_inds, result_img

def map_on_th_image(image, show_all=False): 
    '''
    create new image to be used for thresholding a line with best fit available
    - the line and its search band is straightened, vertical in warped image.
    - bottom stays where it is
    '''
    global l_Line, r_Line
    global debug_lanes
    
    if show_all:
        show_image(image, title='Input to map_on_th_image')    
    
    # warp the image
    warped = warp(image)
    if show_all or debug_lanes:
        # Draw best fit on this image as a red line
        warped_with_best_fit = np.copy(warped)
        draw_fit(warped_with_best_fit,  
                 l_Line.best_yfit, l_Line.best_xfit, [255, 0, 0],
                 r_Line.best_yfit, r_Line.best_xfit, [255, 0, 0])
        show_image(warped_with_best_fit, title='map_on_th_image: warped')    
    
    # start with a black screen (in rgb like the input image)
    th_image = np.zeros_like(warped)

    # x and y values of best fit
    for line in (l_Line, r_Line):
        if line.best_fit is not None:
            best_fit = line.best_fit
            ploty = line.best_yfit
            fitx  = line.best_xfit
        
            # morph search band around best fit into a vertical straightened band
            width  = warped.shape[1]
            
            fitx   = fitx.astype(np.int32)
            ploty  = ploty.astype(np.int32)
            y_bot  = ploty[-1]
            x_bot  = fitx[-1]
            
            for i in range(len(ploty) - 1):
                y     = ploty[i]
                x1    = fitx[i]-MARGIN
                x2    = fitx[i]+MARGIN
                th_x1 = x_bot-MARGIN
                th_x2 = x_bot+MARGIN                 
                
                # line or straightened line is off the picture--> skip this y-level
                if (x1    > width-1 or x2    < 0 or
                    th_x1 > width-1 or th_x2 < 0):
                    continue
                
                x_off_left  = abs( min( 0, x1          , th_x1)          )
                x_off_right =      max( 0, x2-(width-1), th_x2-(width-1) )
                
                x1    = x1    + x_off_left
                x2    = x2    - x_off_right
                th_x1 = th_x1 + x_off_left
                th_x2 = th_x2 - x_off_right                
                    
                th_image[ y, th_x1:th_x2, :] = warped[ y, x1:x2, :]

    if show_all:
            show_image(th_image, title='map_on_th_image: warped & morphed')  
    
    # unwarp the image, so we can send it into the thresholding pipeline
    th_image = unwarp(th_image)
    if show_all:
        show_image(th_image, title='map_on_th_image: unwarped & morphed')     

    return th_image

def map_from_th_image(image, show_all=False): 
    '''
    reverse of map_on_th_image
    NOTE: image is gray scaled & warped...
    '''
    global l_Line, r_Line
    
    if show_all:
        show_image(image, title='Input to map_from_th_image', cmap='gray')    
    
    # start with a black screen (in rgb like the input image)
    th_image = np.zeros_like(image)

    # x and y values of best fit
    for line in (l_Line, r_Line):
        if line.best_fit is not None:
            best_fit = line.best_fit
            ploty = line.best_yfit
            fitx  = line.best_xfit
        
            # morph search band around best fit back from vertical straightened band
            width  = image.shape[1]
            
            fitx   = fitx.astype(np.int32)
            ploty  = ploty.astype(np.int32)
            y_bot  = ploty[-1]
            x_bot  = fitx[-1]            
            
            for i in range(len(ploty) - 1):
                y     = ploty[i]
                x1    = fitx[i]-MARGIN
                x2    = fitx[i]+MARGIN            
                th_x1 = x_bot-MARGIN
                th_x2 = x_bot+MARGIN                 
                
                # line or straightened line is off the picture--> skip this y-level
                if (x1    > width-1 or x2    < 0 or
                    th_x1 > width-1 or th_x2 < 0):
                    continue
                
                x_off_left  = abs( min( 0, x1          , th_x1)          )
                x_off_right =      max( 0, x2-(width-1), th_x2-(width-1) )
                
                x1    = x1    + x_off_left
                x2    = x2    - x_off_right
                th_x1 = th_x1 + x_off_left
                th_x2 = th_x2 - x_off_right       
                
                th_image[ y, x1:x2] = image[ y, th_x1:th_x2] # grayscale, no 3rd dimension

    if show_all:
            show_image(th_image, title='map_from_th_image: unmorphed', cmap='gray')  
    
    return th_image

def fit_polynomials_through_pixels(binary_img, out_img,
                                   left_lane_inds, right_lane_inds,
                                   visualize=True, verbose=False):

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Extract left and right line pixel positions
    leftx = []
    lefty = []
    rightx = []
    righty = []
    if len(left_lane_inds) >0:
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
    if len(right_lane_inds) > 0:
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = None
    right_fit = None
    if len(leftx)>0 and len(lefty)>0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx)>0 and len(righty)>0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    ploty, left_fitx, right_fitx = generate_xy_data_for_plotting_polynomials(out_img, left_fit, right_fit)
        
    # If requested, color lane pixels and draw polynomial on image
    if visualize:       
        # color left-lane pixels red
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # color right-lane pixels blue
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        draw_fit(out_img, 
                 ploty, left_fitx , [255, 255, 0],
                 ploty, right_fitx, [255, 255, 0])
        
    left_fity  = ploty
    right_fity = ploty
    
    return (left_fit, right_fit, left_fitx, left_fity, right_fitx, right_fity, 
            out_img)

def generate_xy_data_for_plotting_polynomials(out_img, left_fit, right_fit):
    # Generate x and y values for plotting of polynomials
    left_fitx  = None
    right_fitx = None
    
    ploty = np.linspace(CROP_TOP, CROP_BOT-1, CROP_BOT-CROP_TOP )

    if left_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        left_fitx = np.maximum(left_fitx,CROP_LEFT) # avoid going out of image...
        left_fitx = np.minimum(left_fitx,CROP_RIGHT-1) 
    if right_fit is not None:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        right_fitx = np.maximum(right_fitx,CROP_LEFT)
        right_fitx = np.minimum(right_fitx,CROP_RIGHT-1)
    
    return ploty, left_fitx, right_fitx    


def draw_fit(out_img, 
             left_fity, left_fitx , left_color,
             right_fity, right_fitx, right_color):
    # Draw as lines on the out_image
    
    # left lane
    if left_fitx is not None:
        for i in range(len(left_fity) - 1):
            p1 = (int(left_fitx[i]  ), int(left_fity[i]  ))
            p2 = (int(left_fitx[i+1]), int(left_fity[i+1]))
            cv2.line(out_img, p1, p2, left_color, 3)
    # right lane
    if right_fitx is not None:
        for i in range(len(right_fity) - 1):
            p1 = (int(right_fitx[i]  ), int(right_fity[i]  ))
            p2 = (int(right_fitx[i+1]), int(right_fity[i+1]))
            cv2.line(out_img, p1, p2, right_color, 3)
            
# using techniques described here: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
def sanity_check_and_draw_lanes(img,left_fit, right_fit, left_fitx, left_fity, right_fitx, right_fity,
                                 text1=None):
    global l_Line, r_Line
    global debug_lanes
        
    if (left_fit is None and right_fit is None and
        l_Line.best_fit is None and r_Line.best_fit is None):
        return img # nothing to show yet.
    
    y_eval_m     = CROP_BOT*ym_per_pix # at bottom of picture, where the car is
    y_eval_m_top = CROP_TOP*ym_per_pix # at top of picture, away from the car
    
    # Fit polynomials to x,y in world space & calculate the radius of curvature
    if left_fit is not None:
        left_fit_cr = np.polyfit(left_fity*ym_per_pix, left_fitx*xm_per_pix, 2)
    if right_fit is not None:    
        right_fit_cr = np.polyfit(right_fity*ym_per_pix, right_fitx*xm_per_pix, 2)
        
    # ===========================================================
    # Calculate lane curvature in meters 
    left_curverad = None
    right_curverad = None
    
    # at the bottom (away from car)
    left_curverad_top = None
    right_curverad_top = None

    if left_fit is not None:
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_m + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        left_curverad_top = ((1 + (2*left_fit_cr[0]*y_eval_m_top + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    if right_fit is not None:    
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_m + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        right_curverad_top = ((1 + (2*right_fit_cr[0]*y_eval_m_top + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    if verbose:
        print('left curvature, right curvature = {0}m, {1}m'.format(left_curverad,right_curverad))
        print('left curvature_top, right curvature_top = {0}m, {1}m'.format(left_curverad_top,right_curverad_top))
            
    # ===========================================================
    # Calculate line-to-car-center distances in meters
    left_car_dist = None     # >0 if inside the lane
    right_car_dist = None    # >0 if inside the lane
    
    x_car_center_m = (WARPED_IMAGE_WIDTH/2.0)*xm_per_pix
    
    if left_fit is not None:
        x_lane_left_m  = left_fit_cr[0]*y_eval_m**2 + left_fit_cr[1]*y_eval_m + left_fit_cr[2]
        left_car_dist = x_car_center_m - x_lane_left_m
        
    if right_fit is not None:
        x_lane_right_m = right_fit_cr[0]*y_eval_m**2 + right_fit_cr[1]*y_eval_m + right_fit_cr[2]
        right_car_dist = x_lane_right_m - x_car_center_m
    
    if (verbose and
        left_car_dist is not None and
        right_car_dist is not None):
        offset = right_car_dist - left_car_dist
        print('offset, left_car_dist, right_car_dist = {0}m, {1}m, {2}m'.format(
               offset, left_car_dist, right_car_dist))
    
    # ===========================================================
    # Calculate lane angles
    left_car_angle      = 0   # at car
    left_car_angle_top  = 0   # at top
    right_car_angle     = 0  
    right_car_angle_top = 0 
    
    if left_fit is not None:
        left_angle_rad      = 2*left_fit_cr[0]*y_eval_m + left_fit_cr[1]
        left_angle_rad_top  = 2*left_fit_cr[0]*y_eval_m_top + left_fit_cr[1]
        left_car_angle      = math.degrees(left_angle_rad)
        left_car_angle_top  = math.degrees(left_angle_rad_top)
        
    if right_fit is not None:
        right_angle_rad      = 2*right_fit_cr[0]*y_eval_m + right_fit_cr[1]
        right_angle_rad_top  = 2*right_fit_cr[0]*y_eval_m_top + right_fit_cr[1]
        right_car_angle      = math.degrees(right_angle_rad)
        right_car_angle_top  = math.degrees(right_angle_rad_top)
    
    if verbose:
        print('left_car_angle, right_car_angle = {0}deg, {1}deg'.format(
               left_car_angle, right_car_angle))
        print('left_car_angle_top, right_car_angle_top = {0}deg, {1}deg'.format(
                left_car_angle_top, right_car_angle_top))        
        
    # -----------------------------------------------------------------------
    # Calculate changes of detected lanes compared to best fit,
    # and decide to accept this lane or not.
    if left_curverad is not None:
        left_curverad_change = left_curverad
    else:
        left_curverad_change = 0
    
    if left_curverad_top is not None:
        left_curverad_change_top = left_curverad_top
    else:
        left_curverad_change_top = 0
    
    left_car_dist_change     = 0
    left_car_angle_change     = 0
    left_fit_change          = np.array([0, 0, 0])
    left_curverad_change_p   = 0
    left_curverad_change_p_top   = 0
    left_car_dist_change_p   = 0
    left_car_angle_change_p   = 0
    left_fit_change_p        = np.array([0, 0, 0])
    left_avrg_xfit_change    = 0 # in m: measure for shifting of lane
    left_max_xfit_change     = 0 # measure for weird local shifting of lane
    if right_curverad is not None:
        right_curverad_change= right_curverad
    else:
        right_curverad_change=0
    
    if right_curverad_top is not None:
        right_curverad_change_top= right_curverad_top
    else:
        right_curverad_change_top=0
        
    right_car_dist_change    = 0
    right_car_angle_change    = 0
    right_fit_change         = np.array([0, 0, 0])
    right_curverad_change_p  = 0
    right_curverad_change_p_top  = 0
    right_car_dist_change_p  = 0
    right_car_angle_change_p   = 0
    right_fit_change_p       = np.array([0, 0, 0])  
    right_avrg_xfit_change    = 0 # measure for shifting of lane
    right_max_xfit_change     = 0 # measure for weird local shifting of lane    
        
    if left_fit is not None and l_Line.best_fit is not None:
        left_curverad_change  = left_curverad-l_Line.best_rad
        left_curverad_change_top  = left_curverad-l_Line.best_rad_top
        left_car_dist_change  = left_car_dist-l_Line.best_car_dist
        left_car_angle_change  = left_car_angle-l_Line.best_car_angle
        left_fit_change       = left_fit-l_Line.best_fit
        left_avrg_xfit_change = np.mean(np.abs(left_fitx-l_Line.best_xfit))*xm_per_pix
        left_max_xfit_change  = np.max(np.abs(left_fitx-l_Line.best_xfit))*xm_per_pix
        
        left_curverad_change_p  = 100*left_curverad_change/l_Line.best_rad
        left_curverad_change_p_top  = 100*left_curverad_change_top/l_Line.best_rad_top
        left_car_dist_change_p  = 100*left_car_dist_change/l_Line.best_car_dist
        left_car_angle_change_p  = 100*left_car_angle_change/l_Line.best_car_angle
        left_fit_change_p       = 100*left_fit_change/l_Line.best_fit

    if right_fit is not None and r_Line.best_fit is not None:
        right_curverad_change  = right_curverad-r_Line.best_rad
        right_curverad_change_top  = right_curverad-r_Line.best_rad_top
        right_car_dist_change  = right_car_dist-r_Line.best_car_dist
        right_car_angle_change  = right_car_angle-r_Line.best_car_angle
        right_fit_change       = right_fit-r_Line.best_fit
        right_avrg_xfit_change = np.mean(np.abs(right_fitx-r_Line.best_xfit))*xm_per_pix
        right_max_xfit_change  = np.max(np.abs(right_fitx-r_Line.best_xfit))*xm_per_pix        
        
        right_curverad_change_p  = 100*right_curverad_change/r_Line.best_rad
        right_curverad_change_p_top  = 100*right_curverad_change_top/r_Line.best_rad_top
        right_car_dist_change_p  = 100*right_car_dist_change/r_Line.best_car_dist
        right_car_angle_change_p  = 100*right_car_angle_change/r_Line.best_car_angle
        right_fit_change_p       = 100*right_fit_change/r_Line.best_fit
    
    left_lane_accepted = True
    right_lane_accepted = True
    
    if (left_fit is None or 
        abs(left_car_dist)           > MAX_CAR_DIST    or 
        left_max_xfit_change         > MAX_XFIT_CHANGE or
        abs(left_curverad)           < MIN_CURVATURE   or
        abs(left_curverad_top)       < MIN_CURVATURE   ): 
        left_lane_accepted = False
    
    if (right_fit is None or 
        abs(right_car_dist)          > MAX_CAR_DIST    or
        right_max_xfit_change        > MAX_XFIT_CHANGE or
        abs(right_curverad)          < MIN_CURVATURE   or
        abs(right_curverad_top)      < MIN_CURVATURE   ): 
        right_lane_accepted = False
    
    # also check:
    # - average distance between the lanes (NOT DONE !)
    # - minimum distance between the lanes (This is a good measure...)
    # This is only a good test at bottom. See eg. 3-0064, where distance at top = 3.86m
    avrg_fitx_diff_between_lanes = 0
    min_fitx_diff_between_lanes  = 0
    max_fitx_diff_between_lanes  = 0
    if left_fitx is not None and right_fitx is not None: 
        fitx_diff_between_lanes = right_fitx - left_fitx
        avrg_fitx_diff_between_lanes = np.mean(np.abs(fitx_diff_between_lanes))*xm_per_pix
        min_fitx_diff_between_lanes  = np.min(fitx_diff_between_lanes)*xm_per_pix
        max_fitx_diff_between_lanes  = np.max(np.abs(fitx_diff_between_lanes))*xm_per_pix
        if (min_fitx_diff_between_lanes < MIN_DIST_BETWEEN_LANES or
            max_fitx_diff_between_lanes > MAX_DIST_BETWEEN_LANES or
            abs(max_fitx_diff_between_lanes - min_fitx_diff_between_lanes) > MAX_DIFF_OF_MAX_AND_MIN_DIST_BETWEEN_LANES or
            abs(left_car_angle - right_car_angle) > MAX_DIFF_ANGLES_BETWEEN_LANES or
            abs(left_car_angle_top - right_car_angle_top) > MAX_DIFF_ANGLES_BETWEEN_LANES or 
            abs(left_car_dist) + abs(right_car_dist) > MAX_CAR_DIST_SUM):
            left_lane_accepted = False
            right_lane_accepted = False
    
    # if both are wrong, reset the history and start over...
    if (left_lane_accepted is False and right_lane_accepted is False and
        l_Line.num_undetected+1 > MAX_FAILURES_BOTH and 
        l_Line.num_undetected+1 > MAX_FAILURES_BOTH):
        l_Line.reset()
        r_Line.reset()
        
    # write info to a file for debug purposes
    write_changes_to_file(frame, left_lane_accepted, right_lane_accepted, 
                          left_curverad_change,left_curverad_change_top, left_car_dist_change, left_fit_change,
                          left_curverad_change_p,left_curverad_change_p_top, left_car_dist_change_p, left_fit_change_p,
                          right_curverad_change,right_curverad_change_top, right_car_dist_change, right_fit_change,
                          right_curverad_change_p,right_curverad_change_p_top, right_car_dist_change_p, right_fit_change_p,
                          left_avrg_xfit_change,left_max_xfit_change,
                          right_avrg_xfit_change,right_max_xfit_change,
                          avrg_fitx_diff_between_lanes,min_fitx_diff_between_lanes,
                          max_fitx_diff_between_lanes,
                          left_car_angle, right_car_angle,
                          left_car_angle_top, right_car_angle_top)    
    
    # store left line
    line = l_Line
    detected = left_lane_accepted
    xfit = left_fitx
    yfit = left_fity
    fit = left_fit
    curverad = left_curverad
    curverad_top = left_curverad_top
    car_dist = left_car_dist
    update_line(line, detected, xfit, yfit, fit, curverad, curverad_top, car_dist, 
                left_car_angle, left_car_angle_top)
    
    # store right line
    line = r_Line
    detected = right_lane_accepted
    xfit = right_fitx
    yfit = right_fity
    fit = right_fit
    curverad = right_curverad
    curverad_top = right_curverad_top
    car_dist = right_car_dist
    update_line(line, detected, xfit, yfit, fit, curverad, curverad_top, car_dist, 
                right_car_angle, right_car_angle_top)
    
    # -----------------------------------------------------------------------
    # draw best_fit lanes with fill in between 
    # (left=yellow, right=blue, fill=green)
    out_img = draw_lanes(img, l_Line.best_fit, r_Line.best_fit, fill=True,
                           color_left=(255,255,0),color_right=(0,0,255),color_fill=(0,255,0))
    if debug_lanes:
        #  draw detected lanes without fill in between 
        # (left=red, right=red)
        out_img = draw_lanes(out_img, left_fit, right_fit, fill=False,
                             color_left=(255,0,0),color_right=(255,0,0))
    
    # -----------------------------------------------------------------------
    # Put some text on the image
    color_txt = (255, 255, 255)

    if text1 is not None:
        cv2.putText(out_img, text1,(10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_txt, 2)
    
    # Put curvature measures as text on out_img
    
    if l_Line.best_rad is not None:
        msg = 'left curvature = {0} m'.format(int(l_Line.best_rad))
        cv2.putText(out_img, msg,(10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_txt, 2)
        
    if r_Line.best_rad is not None:
        msg = 'right curvature = {0} m'.format(int(r_Line.best_rad))
        cv2.putText(out_img, msg,(10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_txt, 2)
    
    if l_Line.best_car_dist is not None and r_Line.best_car_dist is not None:
        offset = r_Line.best_car_dist - l_Line.best_car_dist
        if offset > 0.0:
            msg = 'Vehicle is {0:.2f} m left of center'.format(abs(offset))
        else:
            msg = 'Vehicle is {0:.2f} m right of center'.format(abs(offset))
            
        cv2.putText(out_img, msg,(10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_txt, 2)    
    
    if debug_lanes:
        # put additional debug info on image in debug mode
        txt_yloc   = 15
        y_shift    = 20
        fmt_d = '{0:6.2f}m,  {1:6.2f}m , {2:6.2f}m '
        fmt = '{0}, {1}, {2:.2f}, {3:.2f}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}'
        msg=[]
        msg.append('lane distances: average, min, max')
        msg.append(fmt_d.format(avrg_fitx_diff_between_lanes, min_fitx_diff_between_lanes, max_fitx_diff_between_lanes ))
        msg.append('_______________________________')
        msg.append('for each lane: curvature, curvature_top car_dist, angle, angle_top fit')
        msg.append('_______________________________')
        msg.append('left accepted :  '+str(left_lane_accepted))
        if left_fit is not None:
            msg.append('detected: '+fmt.format(int(left_curverad), int(left_curverad_top),                                               left_car_dist, left_car_angle, left_car_angle_top, *left_fit))
        else:
            msg.append('detection failed !')
            
        for r,r_top,d,a,at, f in zip(l_Line.rads,l_Line.rads_top, l_Line.car_dists,
                                 l_Line.car_angles,l_Line.car_angles_top, l_Line.fits):
            msg.append('history  : '+fmt.format(int(r),int(r_top),d,a,at,*f))        
        msg.append('_______________________________')
        msg.append('right accepted : '+str(right_lane_accepted))
        if right_fit is not None:
            msg.append('detected: '+fmt.format(int(right_curverad), int(right_curverad_top),                                               right_car_dist, right_car_angle, right_car_angle_top, *right_fit))    
        else:
            msg.append('detection failed !')
          
        for r, r_top,d,a,at,f in zip(r_Line.rads, r_Line.rads_top, r_Line.car_dists,
                                  r_Line.car_angles, r_Line.car_angles_top, r_Line.fits):
            msg.append('history  : '+fmt.format(int(r),int(r_top),d,a,at,*f))
            
        
        for txt in msg:
            cv2.putText(out_img, txt,(600, txt_yloc), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_txt, 2)
            txt_yloc += y_shift
    
    
    # return new image
    return out_img

# write changes to a file for debug purposes
def write_changes_to_file(frame, left_lane_accepted, right_lane_accepted, 
                          left_curverad_change,left_curverad_change_top, left_car_dist_change, left_fit_change,
                          left_curverad_change_p,left_curverad_change_p_top, left_car_dist_change_p, left_fit_change_p,
                          right_curverad_change,right_curverad_change_top, right_car_dist_change, right_fit_change,
                          right_curverad_change_p,right_curverad_change_p_top, right_car_dist_change_p, right_fit_change_p,
                          left_avrg_xfit_change,left_max_xfit_change,
                          right_avrg_xfit_change,right_max_xfit_change,
                          avrg_fitx_diff_between_lanes,min_fitx_diff_between_lanes,
                          max_fitx_diff_between_lanes,
                          left_car_angle, right_car_angle,
                          left_car_angle_top, right_car_angle_top):
    global f_changes
    
    msg=[]
    if f_changes is None:
        f_changes = open(fname_changes, 'w')

        msg.append('-------------------------------------------------|----------------------------------')
        msg.append('     LEFT LANE CHANGES                           |    RIGHT LANE CHANGES            ')
        msg.append('curvature,curvature_top car dist, fit            |curvature, curvature_top car dist, fit          ')
        msg.append('avrg_shift, max_shift, car_angle, top_angle      |avrg_shift, max_shift, car_angle, top_angle           ')
        msg.append('avrg_fitx_diff_between_lanes, min_fitx_diff_between_lanes, max_fitx_diff_between_lanes')
        

    msg.append('-------------------------------------------------|----------------------------------')
    msg.append('frame = '+str(frame))
    msg.append('left_accepted = '+str(left_lane_accepted)+'                             |'+
               'right_accepted = '+str(right_lane_accepted))

    fmt   = '{0:>6d}m, {1:>6d}m,  {2:6.2f}m, {3:10.2e} , {4:6.2f} , {5:6.2f}  | {6:>6d}m, {7:>6d}m,  {8:6.2f}m, {9:10.2e} , {10:6.2f} , {11:6.2f} '
    fmt_p = '{0:>6d}%, {1:>6d}%, {2:6.2f}%, {3:10.2f}%, {4:6.2f}%, {5:6.2f}% | {6:>6d}%, {7:>6d}%,  {8:6.2f}%, {9:10.2f}%, {10:6.2f}%, {11:6.2f}%'
    fmt_s = '{0:6.2f}m,  {1:6.2f}m , {2:6.2f}m, {3:6.2f}m                    | {4:6.2f}m,  {5:6.2f}m, {6:6.2f}m, {7:6.2f}m'
    fmt_d = '{0:6.2f}m,  {1:6.2f}m , {2:6.2f}m '
    
    msg.append(fmt.format(int(left_curverad_change), int(left_curverad_change_top), left_car_dist_change, *left_fit_change,
                          int(right_curverad_change), int(right_curverad_change_top), right_car_dist_change, *right_fit_change))
    
    msg.append(fmt_p.format(int(left_curverad_change_p), int(left_curverad_change_p_top), left_car_dist_change_p, *left_fit_change_p,
                            int(right_curverad_change_p), int(right_curverad_change_p_top), right_car_dist_change_p, *right_fit_change_p))        

    msg.append(fmt_s.format(left_avrg_xfit_change,left_max_xfit_change, left_car_angle, left_car_angle_top,
                            right_avrg_xfit_change,right_max_xfit_change, right_car_angle, right_car_angle_top))
    
    msg.append(fmt_d.format(avrg_fitx_diff_between_lanes, min_fitx_diff_between_lanes, max_fitx_diff_between_lanes ))
        
    for txt in msg:
        f_changes.write(txt+'\n')
                
    f_changes.flush() # flush it directly to disk, for debug purposes.
            
# draw lanes with or without fill in between            
def draw_lanes(img, left_fit, right_fit, fill=True,
               color_left=(255,255,0),color_right=(0,0,255),color_fill=(0,255,0)): 
    
    # when asked to fill, only draw lanes when both lanes are there
    if fill and (left_fit is None or right_fit is None):
        return img
    
    # create two copies of the original image -- one for
    # the overlay and one for the final output image
    overlay_img = img.copy()
    out_img     = img.copy()
    
    # -----------------------------------------------------------------------
    # prepare left and right lines as sequence of points
    # Generate x and y values for plotting of polynomials
    # get source point arrays in correct shape for perspectiveTransform
    # see: http://answers.opencv.org/question/252/cv2perspectivetransform-with-python/
    # use the inverse perspective transform to get location of these points in
    # the original image

    ploty, left_fitx, right_fitx = generate_xy_data_for_plotting_polynomials(img, 
                                                                             left_fit, 
                                                                             right_fit)
    src_pts_left  = []
    src_pts_right = []

    if left_fitx is not None: 
        for i in range(len(ploty)):
            src_pts_left.append ( [left_fitx[i] , ploty[i] ] )        
        src_pts_left = np.array(src_pts_left, dtype='float32')
        src_pts_left  = np.array([src_pts_left])  # shape = (1, 720, 2)
        dst_pts_left  = cv2.perspectiveTransform(src_pts_left , Minv)

    if right_fitx is not None:
        for i in range(len(ploty)):
            src_pts_right.append( [right_fitx[i], ploty[i] ] )        
        src_pts_right = np.array(src_pts_right, dtype='float32')
        src_pts_right = np.array([src_pts_right])       
        dst_pts_right = cv2.perspectiveTransform(src_pts_right, Minv)

    # -----------------------------------------------------------------------
    # draw a filled polygon in between lanes on overlay copy, in green
    # See: http://stackoverflow.com/questions/11270250/what-does-the-python-interface-to-opencv2-fillpoly-want-as-input
    if fill:
        if left_fit is not None and right_fit is not None:
            polygon = dst_pts_left[0].tolist() + list(reversed(dst_pts_right[0].tolist()))
            polygon = np.array(polygon, 'int32')
            cv2.fillConvexPoly(overlay_img, polygon, color_fill, lineType=8, shift=0)

    # draw left lane and right lane (blue) on overlay copy
    if left_fit is not None:
        pts_left = np.array(dst_pts_left, 'int32')
        pts_left = pts_left.reshape((-1,1,2))
        cv2.polylines(overlay_img,[pts_left],False,color_left,thickness=4)
    if right_fit is not None:    
        pts_right = np.array(dst_pts_right, 'int32')
        pts_right = pts_right.reshape((-1,1,2))
        cv2.polylines(overlay_img,[pts_right],False,color_right,thickness=4)    

    # -----------------------------------------------------------------------  
    # apply overlay copy to out_img, with transparency
    alpha=0.5
    cv2.addWeighted(overlay_img, alpha, out_img, 1 - alpha, 0, out_img)

    return out_img    
    
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.reset()
        
    def reset(self):
        # 0  if last line was detected
        # >0 number of subsequent undetected lines
        self.num_undetected = 0
        
        # data for each the last n fits of the line:
        self.xfits          = deque([],N_FITS)   # x values
        self.yfits          = deque([],N_FITS)   # y values
        self.fits           = deque([],N_FITS)   # polynomial coefficients
        self.rads           = deque([],N_FITS)   # radius of curvature at bottom
        self.rads_top       = deque([],N_FITS)   # radius of curvature at top
        self.car_dists      = deque([],N_FITS)   # distance in meters of vehicle center from the line
        self.car_angles     = deque([],N_FITS)   # angle at car
        self.car_angles_top = deque([],N_FITS)   # angle at top
        
        # averaged data over the last n fits of the line:
        self.best_xfit          = None           # x location in pixels of best fit
        self.best_yfit          = None           # y location in pixels of best fit
        self.best_fit           = None           # polynomial coefficients    
        self.best_rad           = None           # radius of curvature at car
        self.best_rad_top       = None           # radius of curvature at top
        self.best_car_dist      = None           # distance in meters of vehicle center from the line
        self.best_car_angle     = None           # angle at car
        self.best_car_angle_top = None           # angle at top


# function to keep track of line detection history
def update_line(line, detected, xfit, yfit, fit, curverad, curverad_top, car_dist, 
                car_angle, car_angle_top):
    if detected: 
        line.num_undetected = 0  
        
        # insert last one first. At other end will be pushed out.
        line.xfits.appendleft(xfit) 
        line.yfits.appendleft(yfit)
        line.fits.appendleft(fit)
        line.rads.appendleft(curverad)
        line.rads_top.appendleft(curverad_top)
        line.car_dists.appendleft(car_dist)
        line.car_angles.appendleft(car_angle)
        line.car_angles_top.appendleft(car_angle_top)
        
        line.best_xfit      = np.mean(line.xfits, axis=0, dtype='int32') # store as integers -> pixels 
        line.best_yfit      = np.mean(line.yfits, axis=0, dtype='int32')
        line.best_fit       = np.mean(line.fits, axis=0) 
        line.best_rad       = np.mean(line.rads) 
        line.best_rad_top   = np.mean(line.rads_top)
        line.best_car_dist  = np.mean(line.car_dists)
        line.best_car_angle = np.mean(line.car_angles)
        line.best_car_angle_top = np.mean(line.car_angles_top)
    else:
        line.num_undetected += 1
        # reset best fit if too many failures in a row
        if line.num_undetected > MAX_FAILURES:
            line.reset()


# In[10]:

# put it all in one function that processes a single image
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # Returns the final output (image with lanes drawn on)
    global frame
    global f_changes, fname_changes
    global l_Line, r_Line
    global search_around_best_fit
    global show_all   
    global verbose    
    global show_windows
    global show_final 
    global debug_lanes 
    
    frame += 1
    
    if show_all: 
        print('Frame : ', str(frame),'-', type(image), 'with dimesions:', image.shape)
        show_image(image, 'Original Image')
    
    # distortion correct the image
    image = undistort(image)
    if show_all: show_image(image, 'Distortion Corrected Image')
    
    # threshold the image

    # if one of the lanes does not have a best fit, threshold original image
    binary = None
    if (search_around_best_fit == False or
        l_Line.best_fit is None or
        r_Line.best_fit is None):        
        color, binary = pipeline(image, show_all=show_all)        
    
    # if we have best fit lanes:
    # - threshold on an image that contains filtered & straightened search bands
    binary_fits = None
    if l_Line.best_fit is not None or r_Line.best_fit is not None:
        th_image           = map_on_th_image(image, show_all=show_all)
        color, binary_fits = pipeline(th_image, show_all=show_all)

    
    # warp the thresholded binaries
    if binary is not None:
        warped = warp(binary)
        if show_all: show_image(warped, 'Warped binary', cmap='gray')
    if binary_fits is not None:
        warped_fits = warp(binary_fits)
        if show_all: show_image(warped_fits, 'Warped binary_fits', cmap='gray')
    
    # crop away hood & outer edges, by turning off pixels
    if binary is not None:
        cropped = crop_binary_image(warped,crop_left=CROP_LEFT, crop_right=CROP_RIGHT,
                                crop_bot=CROP_BOT, crop_top=CROP_TOP)
        if show_all: show_image(cropped, 'Cropped binary', cmap='gray') 
    else:
        cropped = np.zeros_like(warped_fits) # make a black canvas for best fit to draw on.
        
    if binary_fits is not None:
        cropped_fits = crop_binary_image(warped_fits,crop_left=CROP_LEFT, crop_right=CROP_RIGHT,
                                        crop_bot=CROP_BOT, crop_top=CROP_TOP)
        if show_all: show_image(cropped_fits, 'Cropped binary_fits', cmap='gray')

    # Find pixels that represent the lane
    # Using windows in regular binary for lines without best fit
    if binary is not None:        
        window_centroids = find_window_centroids(cropped,verbose=verbose)

        left_lane_inds, right_lane_inds, viz_warped = find_pixels_in_lane_windows(cropped, 
                                            window_centroids, WINDOW_WIDTH, WINDOW_HEIGHT,
                                            visualize=True,verbose=verbose)
    # Using best fit if available 
    if binary_fits is not None:
        # map it back from straightened shape
        cropped_fits = map_from_th_image(cropped_fits, show_all=show_all)
        left_lane_inds_fit, right_lane_inds_fit, viz_warped = find_pixels_around_best_fit(cropped_fits, 
                                                visualize=True, verbose=verbose)
        if len(left_lane_inds_fit) >0:
            left_lane_inds = left_lane_inds_fit
        if len(right_lane_inds_fit) >0:
            right_lane_inds = right_lane_inds_fit 
            
        # add these pixels to cropped 
        cropped[(cropped_fits == 1)] = 1
        if show_all: show_image(cropped, 'Cropped + cropped_fits', cmap='gray')
            
            
    # Fit polynomials through left & right lande inds pixels.
    left_fit, right_fit, left_fitx, left_fity, right_fitx, right_fity,        viz_warped = fit_polynomials_through_pixels(cropped, viz_warped,
                                                    left_lane_inds, right_lane_inds,
                                                    visualize=True,verbose=verbose) 
    if show_all or show_windows or debug_lanes: 
        # Draw best fit on this image as a red line
        draw_fit(viz_warped,  
                 l_Line.best_yfit, l_Line.best_xfit, [255, 255, 255],
                 r_Line.best_yfit, r_Line.best_xfit, [255, 255, 255])        
        show_image(viz_warped, 'Image with detected lanes (yellow) & best fit (white)')
            
        
    # Sanity check and then draw best fit lanes on original image
    final_image = sanity_check_and_draw_lanes(image,left_fit, right_fit, left_fitx, left_fity, right_fitx, right_fity,
                                              text1="frame {0}".format(frame))

    if show_all: 
        show_image(final_image, 'Original Image with lanes')
    
    return final_image


# # Test on Videos
# 
# We can test our solution on three provided videos:
# - project_video.mp4
# - challenge_video.mp4
# - harder_challenge_video.mp4

# In[11]:

#----------------------------------------------------------
# global variables used by the processing
N_FITS = 2 # number of detected lines that we save
#N_FITS = 1 # number of detected lines that we save

# activate best fit pixels during polynomial calculation
ACTIVATE_BEST_FIT_PIXELS = True

# use canny to get rid of smudges
USE_CANNY = False

# Parameters to crop out some crappy pixels at the borders
X_CROP = 0

# patchy - hardcoded image sizes, but that is ok for our application
IMAGE_HEIGHT        = 720
IMAGE_WIDTH         = 1280 
WARPED_IMAGE_HEIGHT = IMAGE_HEIGHT
WARPED_IMAGE_WIDTH  = IWP*IMAGE_WIDTH 
CROP_LEFT  = X_CROP
CROP_RIGHT = IMAGE_WIDTH-X_CROP
CROP_BOT   = 695 # crop away hood
CROP_TOP   = 0  
if IWP != 1: # width is changed during perspective transform
    CROP_RIGHT = IWP*IMAGE_WIDTH - X_CROP

# start locations of lines at bottom of graph if no line found yet
LEFT_START = 925
RIGHT_START = 1590        

# acceptance criteria for lane detection compared to best fit
MAX_CAR_DIST               = 2.5   # in m, max allowed lane distance to car center
MAX_CAR_DIST_SUM           = 4.5   # in m, max allowed distance at car between lanes
MAX_XFIT_CHANGE            = 1.0   # in m, max allowed change in horizontal lane shift (max value along line)
DIST_LANES_TARGET          = 3.7   # in m
MIN_DIST_BETWEEN_LANES     = 2.0   # in m
MAX_DIST_BETWEEN_LANES     = 6.0   # in m (in warped space...)
MAX_DIFF_OF_MAX_AND_MIN_DIST_BETWEEN_LANES = 2.5 # in m
#MAX_DIFF_LANES_FROM_TARGET = 3.0   # in m, max allowed difference from target
MIN_CURVATURE              = 25    # in m. Minimum allowed curvature
MAX_DIFF_ANGLES_BETWEEN_LANES = 10  # max difference in angles between left & right lanes, at top & at car

# reset individual lines best fit and start over after a number of subsequent failures
MAX_FAILURES = 15 # eg. it takes 20 frames to get under the bridge in Challenge Video

# reset if both lanes fail a number of times in a row
MAX_FAILURES_BOTH   = 3

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = DIST_LANES_TARGET/700 # meters per pixel in x dimension


l_Line = Line() # to keep track of left line history
r_Line = Line() # to keep track of right line history
frame=0
f_changes = None # file to write debug information about detection changes
fname_changes = 'detection_changes.txt'

# this activates search around best fit. 
# Always leave on, unless you want to debug new window search logic
search_around_best_fit=True        
#----------------------------------------------------------        

# for debugging only:
show_all        = False
show_histograms = False
verbose         = False
show_windows    = False
show_final      = False
debug_lanes     = False

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

project_video_output = 'videoProject.mp4'
clip1 = VideoFileClip("project_video.mp4")
project_video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!



# In[12]:

get_ipython().magic('time project_video_clip.write_videofile(project_video_output, audio=False)')


# In[13]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_video_output))


# # Optional: Challenge Video

# In[14]:

#----------------------------------------------------------
# global variables used by the processing
N_FITS = 2 # number of detected lines that we save
#N_FITS = 1 # number of detected lines that we save

# activate best fit pixels during polynomial calculation
ACTIVATE_BEST_FIT_PIXELS = True

# use canny to get rid of smudges
USE_CANNY = False

# Parameters to crop out some crappy pixels at the borders
X_CROP = 0

# patchy - hardcoded image sizes, but that is ok for our application
IMAGE_HEIGHT        = 720
IMAGE_WIDTH         = 1280 
WARPED_IMAGE_HEIGHT = IMAGE_HEIGHT
WARPED_IMAGE_WIDTH  = IWP*IMAGE_WIDTH 
CROP_LEFT  = X_CROP
CROP_RIGHT = IMAGE_WIDTH-X_CROP
CROP_BOT   = 705 # crop away hood
CROP_TOP   = 20 
if IWP != 1: # width is changed during perspective transform
    CROP_RIGHT = IWP*IMAGE_WIDTH - X_CROP

# start locations of lines at bottom of graph if no line found yet
LEFT_START = 925
RIGHT_START = 1590        

# acceptance criteria for lane detection compared to best fit
MAX_CAR_DIST               = 2.5   # in m, max allowed lane distance to car center
MAX_CAR_DIST_SUM           = 4.0   # in m, max allowed distance at car between lanes
MAX_XFIT_CHANGE            = 1.0   # in m, max allowed change in horizontal lane shift (max value along line)
DIST_LANES_TARGET          = 3.7   # in m
MIN_DIST_BETWEEN_LANES     = 2.0   # in m
MAX_DIST_BETWEEN_LANES     = 6.0   # in m (in warped space...)
MAX_DIFF_OF_MAX_AND_MIN_DIST_BETWEEN_LANES = 2.5 # in m
#MAX_DIFF_LANES_FROM_TARGET = 3.0   # in m, max allowed difference from target
MIN_CURVATURE              = 25    # in m. Minimum allowed curvature
MAX_DIFF_ANGLES_BETWEEN_LANES = 10  # max difference in angles between left & right lanes, at top & at car

# reset individual lines best fit and start over after a number of subsequent failures
MAX_FAILURES = 15 # eg. it takes 20 frames to get under the bridge in Challenge Video

# reset if both lanes fail a number of times in a row
MAX_FAILURES_BOTH   = 3

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = DIST_LANES_TARGET/700 # meters per pixel in x dimension


l_Line = Line() # to keep track of left line history
r_Line = Line() # to keep track of right line history
frame=0
f_changes = None # file to write debug information about detection changes
fname_changes = 'detection_changes.txt'

# this activates search around best fit. 
# Always leave on, unless you want to debug new window search logic
search_around_best_fit=True        
#----------------------------------------------------------

# for debugging only:
show_all        = False
show_histograms = False
verbose         = False
show_windows    = False
show_final      = False
debug_lanes     = False

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

challenge_video_output = 'videoChallenge.mp4'
clip2 = VideoFileClip("challenge_video.mp4")
challenge_video_clip = clip2.fl_image(process_image) #NOTE: this function expects color images!!


# In[15]:

get_ipython().magic('time challenge_video_clip.write_videofile(challenge_video_output, audio=False)')


# In[16]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_video_output))


# # Optional: Harder Challenge Video

# In[17]:

#----------------------------------------------------------
# global variables used by the processing
N_FITS = 2 # number of detected lines that we save
#N_FITS = 1 # number of detected lines that we save

# activate best fit pixels during polynomial calculation
ACTIVATE_BEST_FIT_PIXELS = False

# use canny to get rid of smudges
USE_CANNY = False

# Parameters to crop out some crappy pixels at the borders
X_CROP = 0

# patchy - hardcoded image sizes, but that is ok for our application
IMAGE_HEIGHT        = 720
IMAGE_WIDTH         = 1280 
WARPED_IMAGE_HEIGHT = IMAGE_HEIGHT
WARPED_IMAGE_WIDTH  = IWP*IMAGE_WIDTH 
CROP_LEFT  = X_CROP
CROP_RIGHT = IMAGE_WIDTH-X_CROP
CROP_BOT   = 705 # crop away hood
CROP_TOP   = 200  # crop 200 pixels at the top, else we would need 3rd order polynomial (3-104)
if IWP != 1: # width is changed during perspective transform
    CROP_RIGHT = IWP*IMAGE_WIDTH - X_CROP

# start locations of lines at bottom of graph if no line found yet
LEFT_START = 925
RIGHT_START = 1590        

# acceptance criteria for lane detection compared to best fit
MAX_CAR_DIST               = 2.5   # in m, max allowed lane distance to car center
MAX_CAR_DIST_SUM           = 4.5   # in m, max allowed distance at car between lanes
MAX_XFIT_CHANGE            = 1.0   # in m, max allowed change in horizontal lane shift (max value along line)
DIST_LANES_TARGET          = 3.7   # in m
MIN_DIST_BETWEEN_LANES     = 2.0   # in m
MAX_DIST_BETWEEN_LANES     = 6.0   # in m (in warped space...)
MAX_DIFF_OF_MAX_AND_MIN_DIST_BETWEEN_LANES = 2.5 # in m
#MAX_DIFF_LANES_FROM_TARGET = 3.0   # in m, max allowed difference from target
MIN_CURVATURE              = 25    # in m. Minimum allowed curvature
MAX_DIFF_ANGLES_BETWEEN_LANES = 10  # max difference in angles between left & right lanes, at top & at car

# reset individual lines best fit and start over after a number of subsequent failures
MAX_FAILURES = 15 # eg. it takes 20 frames to get under the bridge in Challenge Video

# reset if both lanes fail a number of times in a row
MAX_FAILURES_BOTH   = 3

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = DIST_LANES_TARGET/700 # meters per pixel in x dimension


l_Line = Line() # to keep track of left line history
r_Line = Line() # to keep track of right line history
frame=0
f_changes = None # file to write debug information about detection changes
fname_changes = 'detection_changes_harder_challenge.txt'

# this activates search around best fit. 
# Always leave on, unless you want to debug new window search logic
search_around_best_fit=True        
#----------------------------------------------------------        

# for debugging only:
show_all        = False
show_histograms = False
verbose         = False
show_windows    = False
show_final      = False
debug_lanes     = False

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

harder_challenge_video_output = 'videoHarderChallenge.mp4'
clip3 = VideoFileClip("harder_challenge_video.mp4")
harder_challenge_video_clip = clip3.fl_image(process_image) #NOTE: this function expects color images!!


# In[18]:

get_ipython().magic('time harder_challenge_video_clip.write_videofile(harder_challenge_video_output, audio=False)')


# In[19]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(harder_challenge_video_output))


# # Next cell is for debugging the logic
# 
# The next cell allows to debug the logic used. The videos were unpacked into individual jpeg files, and the code below allows to read in a certain frame and ask for a lot of intermediate images that demonstrate the thresholding process, pixel search and polynomial fitting.

# In[21]:

#
RUN_THIS_CELL = True

# for debugging only:
show_all        = False
show_histograms = False
verbose         = False
show_windows    = False
show_final      = False
debug_lanes     = True   # draws detected lanes on top of best fit

files_debug_true = None

test_video = 1

if RUN_THIS_CELL:
    print('Testing the lane finding using process_image function')

    #======================================================
    # Project Video
    if test_video==1:
        #----------------------------------------------------------
        # global variables used by the processing
        N_FITS = 2 # number of detected lines that we save
        #N_FITS = 1 # number of detected lines that we save
        
        # activate best fit pixels during polynomial calculation
        ACTIVATE_BEST_FIT_PIXELS = False
        
        # use canny to get rid of smudges
        USE_CANNY = False
        
        # Parameters to crop out some crappy pixels at the borders
        X_CROP = 0
        
        # patchy - hardcoded image sizes, but that is ok for our application
        IMAGE_HEIGHT        = 720
        IMAGE_WIDTH         = 1280 
        WARPED_IMAGE_HEIGHT = IMAGE_HEIGHT
        WARPED_IMAGE_WIDTH  = IWP*IMAGE_WIDTH 
        CROP_LEFT  = X_CROP
        CROP_RIGHT = IMAGE_WIDTH-X_CROP
        CROP_BOT   = 705 # crop away hood
        CROP_TOP   = 200  # crop 200 pixels at the top, else we would need 3rd order polynomial (3-104)
        if IWP != 1: # width is changed during perspective transform
            CROP_RIGHT = IWP*IMAGE_WIDTH - X_CROP
        
        # start locations of lines at bottom of graph if no line found yet
        LEFT_START = 925
        RIGHT_START = 1590        
        
        # acceptance criteria for lane detection compared to best fit
        MAX_CAR_DIST               = 2.5   # in m, max allowed lane distance to car center
        MAX_CAR_DIST_SUM           = 4.5   # in m, max allowed distance at car between lanes
        MAX_XFIT_CHANGE            = 1.0   # in m, max allowed change in horizontal lane shift (max value along line)
        DIST_LANES_TARGET          = 3.7   # in m
        MIN_DIST_BETWEEN_LANES     = 2.0   # in m
        MAX_DIST_BETWEEN_LANES     = 6.0   # in m (in warped space...)
        MAX_DIFF_OF_MAX_AND_MIN_DIST_BETWEEN_LANES = 2.5 # in m
        #MAX_DIFF_LANES_FROM_TARGET = 3.0   # in m, max allowed difference from target
        MIN_CURVATURE              = 25    # in m. Minimum allowed curvature
        MAX_DIFF_ANGLES_BETWEEN_LANES = 10  # max difference in angles between left & right lanes, at top & at car

        # reset individual lines best fit and start over after a number of subsequent failures
        MAX_FAILURES = 15 # eg. it takes 20 frames to get under the bridge in Challenge Video

        # reset if both lanes fail a number of times in a row
        MAX_FAILURES_BOTH   = 3
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = DIST_LANES_TARGET/700 # meters per pixel in x dimension


        l_Line = Line() # to keep track of left line history
        r_Line = Line() # to keep track of right line history
        frame=0
        f_changes = None # file to write debug information about detection changes
        fname_changes = 'detection_changes.txt'

        # this activates search around best fit. 
        # Always leave on, unless you want to debug new window search logic
        search_around_best_fit=True        
        #----------------------------------------------------------        

        file_dir = "test_images_project_video"
        file_dir_out = "test_images_project_video_output"
        files = os.listdir(file_dir)
        
        # set an individual frame
        #files = ['project_video_output0001.jpg']
        
        # startup frames
        files = ['project_video_output0001.jpg',
                 'project_video_output0002.jpg']
        
        # frames for first bright spot in road
        #files = ['project_video_output0579.jpg',
        #         'project_video_output0580.jpg',
        #         'project_video_output0581.jpg']
                
        # frames for second bright spot in road
        #files = ['project_video_output0995.jpg',
        #         'project_video_output0996.jpg',
        #         'project_video_output0997.jpg',
        #         'project_video_output0998.jpg',
        #         'project_video_output0999.jpg',
        #         'project_video_output1000.jpg',
        #         'project_video_output1001.jpg',
        #         'project_video_output1002.jpg',
        #         'project_video_output1003.jpg',
        #         'project_video_output1004.jpg'
        #         ]
        
        # file for which to write all debug info, including all intermediate pipeline images
        files_debug_true = ['project_video_output0001.jpg',
                            'project_video_output0002.jpg']

    #======================================================
    # Challenge Video
    if test_video==2:
        #----------------------------------------------------------
        # global variables used by the processing
        N_FITS = 2 # number of detected lines that we save
        #N_FITS = 1 # number of detected lines that we save
        
        # activate best fit pixels during polynomial calculation
        ACTIVATE_BEST_FIT_PIXELS = True
        
        # use canny to get rid of smudges
        USE_CANNY = False
        
        # Parameters to crop out some crappy pixels at the borders
        X_CROP = 0
        
        # patchy - hardcoded image sizes, but that is ok for our application
        IMAGE_HEIGHT        = 720
        IMAGE_WIDTH         = 1280 
        WARPED_IMAGE_HEIGHT = IMAGE_HEIGHT
        WARPED_IMAGE_WIDTH  = IWP*IMAGE_WIDTH 
        CROP_LEFT  = X_CROP
        CROP_RIGHT = IMAGE_WIDTH-X_CROP
        CROP_BOT   = 705 # crop away hood
        CROP_TOP   = 20 
        if IWP != 1: # width is changed during perspective transform
            CROP_RIGHT = IWP*IMAGE_WIDTH - X_CROP
        
        # start locations of lines at bottom of graph if no line found yet
        LEFT_START = 925
        RIGHT_START = 1590        
        
        # acceptance criteria for lane detection compared to best fit
        MAX_CAR_DIST               = 2.5   # in m, max allowed lane distance to car center
        MAX_CAR_DIST_SUM           = 4.0   # in m, max allowed distance at car between lanes
        MAX_XFIT_CHANGE            = 1.0   # in m, max allowed change in horizontal lane shift (max value along line)
        DIST_LANES_TARGET          = 3.7   # in m
        MIN_DIST_BETWEEN_LANES     = 2.0   # in m
        MAX_DIST_BETWEEN_LANES     = 6.0   # in m (in warped space...)
        MAX_DIFF_OF_MAX_AND_MIN_DIST_BETWEEN_LANES = 2.5 # in m
        #MAX_DIFF_LANES_FROM_TARGET = 3.0   # in m, max allowed difference from target
        MIN_CURVATURE              = 25    # in m. Minimum allowed curvature
        MAX_DIFF_ANGLES_BETWEEN_LANES = 10  # max difference in angles between left & right lanes, at top & at car

        # reset individual lines best fit and start over after a number of subsequent failures
        MAX_FAILURES = 15 # eg. it takes 20 frames to get under the bridge in Challenge Video

        # reset if both lanes fail a number of times in a row
        MAX_FAILURES_BOTH   = 3
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = DIST_LANES_TARGET/700 # meters per pixel in x dimension


        l_Line = Line() # to keep track of left line history
        r_Line = Line() # to keep track of right line history
        frame=0
        f_changes = None # file to write debug information about detection changes
        fname_changes = 'detection_changes.txt'

        # this activates search around best fit. 
        # Always leave on, unless you want to debug new window search logic
        search_around_best_fit=True        
        #----------------------------------------------------------
        file_dir = "test_images_challenge_video"
        file_dir_out = "test_images_challenge_video_output"
        files = os.listdir(file_dir)
                
        #files = ['challenge_video_output0030.jpg']
        
        files = ['challenge_video_output0001.jpg',
                 'challenge_video_output0002.jpg',
                 'challenge_video_output0003.jpg',
                 'challenge_video_output0004.jpg',
                 'challenge_video_output0005.jpg',
                 'challenge_video_output0006.jpg',
                 'challenge_video_output0007.jpg',
                 'challenge_video_output0008.jpg',
                 'challenge_video_output0009.jpg']

        
        files_debug_true = ['challenge_video_output0001.jpg',
                            'challenge_video_output0002.jpg']

    #======================================================
    # Harder challenge Video
    if test_video==3:
        #----------------------------------------------------------
        # global variables used by the processing
        N_FITS = 2 # number of detected lines that we save
        #N_FITS = 1 # number of detected lines that we save
        
        # activate best fit pixels during polynomial calculation
        ACTIVATE_BEST_FIT_PIXELS = False
        
        # use canny to get rid of smudges
        USE_CANNY = False
        
        # Parameters to crop out some crappy pixels at the borders
        X_CROP = 0
        
        # patchy - hardcoded image sizes, but that is ok for our application
        IMAGE_HEIGHT        = 720
        IMAGE_WIDTH         = 1280 
        WARPED_IMAGE_HEIGHT = IMAGE_HEIGHT
        WARPED_IMAGE_WIDTH  = IWP*IMAGE_WIDTH 
        CROP_LEFT  = X_CROP
        CROP_RIGHT = IMAGE_WIDTH-X_CROP
        CROP_BOT   = 705 # crop away hood
        CROP_TOP   = 200  # crop 200 pixels at the top, else we would need 3rd order polynomial (3-104)
        if IWP != 1: # width is changed during perspective transform
            CROP_RIGHT = IWP*IMAGE_WIDTH - X_CROP
        
        # start locations of lines at bottom of graph if no line found yet
        LEFT_START = 925
        RIGHT_START = 1590        
        
        # acceptance criteria for lane detection compared to best fit
        MAX_CAR_DIST               = 2.5   # in m, max allowed lane distance to car center
        MAX_CAR_DIST_SUM           = 4.0   # in m, max allowed distance at car between lanes
        MAX_XFIT_CHANGE            = 1.0   # in m, max allowed change in horizontal lane shift (max value along line)
        DIST_LANES_TARGET          = 3.7   # in m
        MIN_DIST_BETWEEN_LANES     = 2.0   # in m
        MAX_DIST_BETWEEN_LANES     = 6.0   # in m (in warped space...)
        MAX_DIFF_OF_MAX_AND_MIN_DIST_BETWEEN_LANES = 2.5 # in m
        #MAX_DIFF_LANES_FROM_TARGET = 3.0   # in m, max allowed difference from target
        MIN_CURVATURE              = 25    # in m. Minimum allowed curvature
        MAX_DIFF_ANGLES_BETWEEN_LANES = 10  # max difference in angles between left & right lanes, at top & at car

        # reset individual lines best fit and start over after a number of subsequent failures
        MAX_FAILURES = 15 # eg. it takes 20 frames to get under the bridge in Challenge Video

        # reset if both lanes fail a number of times in a row
        MAX_FAILURES_BOTH   = 3
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = DIST_LANES_TARGET/700 # meters per pixel in x dimension


        l_Line = Line() # to keep track of left line history
        r_Line = Line() # to keep track of right line history
        frame=0
        f_changes = None # file to write debug information about detection changes
        fname_changes = 'detection_changes.txt'

        # this activates search around best fit. 
        # Always leave on, unless you want to debug new window search logic
        search_around_best_fit=True        
        #----------------------------------------------------------        
        
        file_dir = "test_images_harder_challenge_video"
        file_dir_out = "test_images_harder_challenge_video_output"
        files = os.listdir(file_dir)

        #files = ['harder_challenge_video_output0001.jpg']
        
        #files = ['harder_challenge_video_output0001.jpg',
        #         'harder_challenge_video_output0002.jpg'] 
        
        files_debug_true = ['harder_challenge_video_output0001.jpg',
                            'harder_challenge_video_output0002.jpg']
    
    # to start in later frame:
    #files = files[206:]
    #frame = 206
    
    for file in tqdm(files):
        if files_debug_true is not None:
            if file in files_debug_true :
                show_all        = True
                show_histograms = True
                verbose         = True
                show_windows    = True
                show_final      = True
                debug_lanes     = True
            else:
                show_all        = False
                show_histograms = False
                verbose         = True
                show_windows    = True
                show_final      = True
                debug_lanes     = True
            
        file_in=file_dir+"/"+file
        file_out=file_dir_out+"/"+file
        file_out2=file_dir_out+"/unwarped_with_lanes_"+file

        image = mpimg.imread(file_in)

        if show_windows:
            show_image(image,title=file)

        final_image = process_image(image)

        plt.imsave(file_out,final_image)

        if show_final:
            show_image(final_image,title=file)


# In[ ]:



