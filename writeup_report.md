# Advanced Lane Finding

---

Advanced Lane Finding Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Sanity check and output visual display of lane boundaries together with numerical estimation of lane curvature and vehicle position on the original image

All the steps are explained & documented in this [Jupyter notebook ](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/P4.ipynb). (or it's [HTML version](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/P4.html))


The write up you are reading provides some additional information.

---

# 1. Submission includes all required files

My project includes the following files:

- [<b>P4.ipynb</b> - The Jupyter notebook containing all code with explanations](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/P4.ipynb)
- [<b>P4.html</b> - The executed Jupyter notebook as HTML ](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/P4.html)
- [<b>writeup_report.md</b> - A summary of the project](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/writeup_report.md)
- [<b>videoProject.mp4</b> - The project video with lanes (works quite nice)](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/videoProject.mp4)

    ![track1](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/videoProject.gif?raw=true)

- [<b>videoChallenge.mp4</b> - The challenge video with lanes (works ok, with a few small glitches)](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/videoChallenge.mp4)


    ![track1](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/videoChallenge.gif?raw=true)

-  [<b>videoHarderChallenge.mp4</b> - The harder challenge video with lanes (not so good yet... Shadows are tough!)](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/videoHarderChallenge.mp4)


    ![track1](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/videoHarderChallenge.gif?raw=true)
  
---

# 2. Utility functions

In code cell 2, there are some utility functions to plot images, and a function to draw lines onto an image.

---

# 3. Camera calibration

The calibrating of the camera is done in code cells 3 & 4.

The camera calibration is done by verifying how much a chessboard is being deformed by the camera, using these steps:

<b>Code cell 3</b>

- Define how many inside corners are on the chessboard in x and y direction (9 and 6 respectively in our test samples)
- Use the function cv2.findChessboardCorners to detect the inside corners
- If found, you can draw them to double check it is all OK. For example:
![image](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/images/Corners_on_calibration02.jpg?raw=true)
- If the inside corners are found, the object and image points are stored in arrays for the actual camera calibration that is done next.

<b>Code cell 4</b>

- Once the object and image points of the chessboard images are found, they are used to calibrate the camera.
- The calibration results in a transformation matrix, which is stored to disk in a pickle file.

---

# 4. Distortion correction

A test distortion correct images is done in code cells 5 & 6.

- Once the transformation matrix is determined, images can be undistorted. This is verified for all the chessboard images that were used to calibrate the camera. The result was very good, for example:
![image](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/images/Distortion_Correction_calibration01.jpg?raw=true)

Model Architecture and Training Strategy
---

# 5. Thresholding pipeline

In code cell 7 the thresholding logic is defined in the function: pipeline

To check what this pipeline is doing, it is best to look at the full sequence of images produced by the very last code cell in the Jupyter notebook ([HTML version](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/P4.html))

This function takes an rgb image, and applies 3 thresholding mechanisms:

1. Gradients:
    - grayscale
    - calculate and apply thresholds using x-gradient, y-gradient, magnitude-of-gradient, direction-of-gradient
    - combine them into a binary image for the gradients, but exclude all pixels that end up black during a 'blackening' operation (*)
    - apply morphologyEx, to eliminate noise

    
2. HSV color space:
    - blacken the s-channel (*)
    - clahe the v-channel, to allow a wider v-channel threshold
    - threshold the h-, s- & v-channels
    - combine them into a binary image for the HSV color space
    
3. HLS color space
    - blacken the s-channel (*)
    - threshold the h-, l- & s-channels
    - combine them into a binary image for the HLS color space

4. At the end, these 3 thresholds are combined in a single binary image used to detect the lanes.

(*) The <b>blackening</b> technique is a method where I take the average level in a certain region, and then set to black (0) every pixel that is below this average level. This turned out to be a very efficient method to deal with black stripes & smudges on the road, and was key in getting video 1 & 2 to work. Unfortunately, it is not a good thing to do when there are a lot of shadows, because it tends to blacken away everything that is covered by the shadows. This is why video 3 is not working that great, and I am looking for a better approach that deals with shadows of video 3 and the dark lines and smudges of videos 1 & 2. 

It was an effort of trial-and-error to find a good combination of blackening, thresholding and clahe that works most of the time. 

In the last code cell of the notebook, it is possible to test the lane detection on individual images, with option to display all intermediate steps.
 

# 6. Perspective Transform

In code cell 8 the perspective transform logic is defined.

The way this works is as follows:

- Manually determine a set of 4 source points and 4 destination points, in pixel location.
- The transformation matrix is calculated that will re-position the source points to the destination points
- That transformation matrix can then be applied to the whole image

By correctly selecting the source and destination points, the image can be 'warped' into a top view. 

This is demonstrated on this image:

![image](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/images//Perspective_Transform_1.jpg?raw=true)


One special note is that I used a destination image that is twice as wide as the original image, while using the same height. I did this to assist with cases I found in the harder challenge video, where a lane exits the side of the original size warped image making lane detection error prone:

![image](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/images//Perspective_Transform_2.jpg?raw=true)

# 7. Detect Lanes

In code cell 9 the logic for lane detection is defined.

At the top of this code cell, some tuning parameters are defined for:

- Size of the sliding window and side-ways margin used during initial lane search over thresholded binary image
- Size of the cropped window used during lane search over thresholded binary image
- Number of curves we store to average out the polynomial fit
- Acceptance criteria of the polynomial lines that represent the left and right lanes

The logic of lane detection is as follows:

<b>Sliding window search</b>

A sliding window method applies a convolution, which maximizes the number of "hot"  pixels in each window. A convolution is the summation of the product of two separate signals, in our case the window template and the vertical slice of the pixel image.

We slide a window template across the image from left to right and any overlapping values are summed together, creating the convolved signal. The peak of the convolved signal is where there was the highest overlap of pixels and the most likely position for the lane marker.

We do this at each level (or layer), where we shift the window upwards each time. We stack the windows on top of each other with a certain side-ways shift allowed.

I build in some logic to deal with cases where no good window is found for a certain layer. This happens often for non-continuous lanes. When a layer does not have a good window, we do mark it as such, and leave it at same x-position as the window of the previous level.

In case the initial levels do not have a good window, but then a good window is found, the x-position of the windows on the lower levels is adjusted to match the x-location of the good window.

Once the windows are selected, all pixels inside those windows are selected for polynomial fit calculation.

This process is graphically visualized as:

![image](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/images//video1_frame1_window_search.jpg?raw=true)


<b>Pixel search around previously found lanes</b>

When a fit of the lanes is available from a previous image in the video, we can optimize things.

First of all, after warping the image, we can take a region around the best fit, and focus just on that during thresholding. To further help the thresholding, we straighten the band of pixels around the best fit, so it is vertical in the warped image. After that, we unwarp it, and that is the image that goes into the thresholding pipeline. I refer to it as the 'th_image':

![image](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/images//video1_frame2_best_fit_available_th_image.jpg?raw=true)


Furthermore, after the thresholding is done, and we have a binary image, we do not need to do a sliding window search. We simply select all those pixels of the thresholded binary image that are located inside the band around the previously found best fit. The image below shows both the new polynomial of video 1, frame 2, in yellow, and the best fit in white:

![image](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/images//video1_frame2_best_fit_available_select_pixels.jpg?raw=true)

<b>Fit polynomials through selected pixels</b>

Once the valid pixels are selected from the thresholded binary image, either through windows search or in a region around the previously found lanes, a polynomial is fit through it. One polynomial for the left lane, and one for the right lane.

To avoid jittery behavior, the pixels of the best fit curve of previously determined lanes are also activated. 

This polynomial fit is already shown as yellow lines in the images above of pixel selection with windows search or with best fit band.


# 8. Sanity check and display lanes onto original image

Once the polynomials are calculated, we do a sanity check, and if all is OK, we draw the polynomials on the output image, and color the region in between green.

When we run in 'production' mode, some text is put in the left upper corner, as can be seen in the videos themselves.

When we run in 'debug' mode, some additional information is put in the right upper corner:

![image](https://github.com/ArjaanBuijk/CarND_Advanced_Lane_Lines/blob/master/images//example_output_with_debug_info.jpg?raw=true)

The sanity check on the lanes include a check on the curve radius at top & bottom of line, the distances between the lanes, and the distance between each lane and the center of the car.

If all is OK, the lanes are accepted, and stored. To avoid jittery behavior, the polynomials and some additional information is stored in instances of the class Line. One instance for the left (l_Line), and one for the right (r_Line).

We save the history of 2 images. This is managed using a deque, which automatically pushes out old lanes if a new one is added at the front.

Each time a new curve is stored for a lane, the best fit is re-calculated.

If it is not OK, one or possibly both lane detections are ignored. A counter for each lane keeps track how many subsequent failures occur, and a separate counter keeps track how many subsequent dual failures occur. Once a certain number of subsequent failures happen, it is assumed that the best-fit curve is no longer good, and the history information for one or both lines will be reset.

# 9. Summary

Overall, it was an interesting exercise that forced me to explore a lot of new things. In the end I feel that something basic is still missing in my logic to deal with shadows and very bright spots that obscure the yellow and white lanes. When adding logic to deal with one issue, it tends to break the logic for another, and it feels a bit like playing whack-a-mole.