
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in an Image

# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[13]:


import math

#add global variables to apply a line filter.
global right_top_prev
global right_bottom_prev
global left_top_prev
global left_bottom_prev
right_top_prev = None
right_bottom_prev = None
left_top_prev = None
left_bottom_prev = None

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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

def cal_top_bottom(line_slope,line_center,line_len,top_y,bottom_y,top_prev,bottom_prev):
    """Calculate top and bottom point of the lane"""
    line_cnt = 0
    line_center_mean = (0,0)
    slope_mean = 0

    if len(line_slope) > 0:
        #Calculate average center and slope based on the weight(length of lines)
        for i in range(len(line_slope)):
            line_center_mean = (line_center_mean[0] + line_len[i] * line_center[i][0], line_center_mean[1] + line_len[i] * line_center[i][1])
            slope_mean += line_len[i]*line_slope[i]
            line_cnt += line_len[i]
        line_center_mean = (line_center_mean[0]/line_cnt,line_center_mean[1]/line_cnt)
        slope_mean = slope_mean/line_cnt
        
        top_point = (int(low_pass_filter((top_y-line_center_mean[1])/slope_mean + line_center_mean[0],top_prev)), int(top_y))
        bottom_point = (int(low_pass_filter((bottom_y-line_center_mean[1])/slope_mean + line_center_mean[0],bottom_prev)), int(bottom_y))
    else:
        top_point = (int(top_prev),int(top_y))
        bottom_point = (int(bottom_prev),int(bottom_y))
        
    return top_point,bottom_point
    
def low_pass_filter(x_current,y_prev,a = 0.1):
    """ Smooth the lines difference between frames """
    if y_prev != None:
        y_current = a * x_current + (1-a) * y_prev
    else:
        y_current = x_current
    
    return y_current

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
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
    
    global right_top_prev
    global right_bottom_prev
    global left_top_prev
    global left_bottom_prev
    
    #set top and bottom of lane
    top_y = 0.6 * img.shape[0]
    bottom_y = img.shape[0]
    
    left_slope = []
    left_center = []
    left_len = []
    
    right_slope = []
    right_center = []
    right_len = []
    #calculate slopes/centers/length of all line and separate them to left/right
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y2 - y1)/(x2 - x1)
                center = ((x1+x2)/2,(y1+y2)/2)
                line_length = math.sqrt(math.pow((y2-y1),2) + math.pow((x2-x1),2))
                if slope < 0:
                    # Ignore obvious invalid lines
                    if slope > -.5 or slope < -.8:
                        continue 
                    left_slope.append(slope)
                    left_center.append(center)
                    left_len.append(line_length)
                
                elif slope > 0:
                    if slope < .5 or slope > .8:
                        continue 
                    right_slope.append(slope)
                    right_center.append(center)
                    right_len.append(line_length)
                
    left_top,left_bottom = cal_top_bottom(left_slope,left_center,left_len,top_y,bottom_y,left_top_prev,left_bottom_prev)
    #update prev value
    left_top_prev = left_top[0]
    left_bottom_prev = left_bottom[0]
    
    right_top,right_bottom = cal_top_bottom(right_slope,right_center,right_len,top_y,bottom_y,right_top_prev,right_bottom_prev)
    #update prev value
    right_top_prev = right_top[0]
    right_bottom_prev = right_bottom[0]
            
    cv2.line(img, left_top, left_bottom, color, thickness)
    cv2.line(img, right_top, right_bottom, color, thickness)

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

def color_filter(img):
    "Add a color fiter for white and yellow"
    color_low = np.array([180,180,0])
    color_high = np.array([255,255,255])
    color_mask = cv2.inRange(img, color_low, color_high)
    filtered_img = cv2.bitwise_and(img, img, mask = color_mask)
    return filtered_img
    

def image_processing(img):
    output_route = 'writeup_img/'
    #filter colors
    filtered_img = color_filter(img)
    mpimg.imsave(output_route + 'color_filtered.jpg', filtered_img)
    #img to gray scale
    gray = grayscale(filtered_img)
    mpimg.imsave(output_route + 'gray_img.jpg', gray)
    # blur gray img with kernel_size
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    mpimg.imsave(output_route + 'blur_gray.jpg', blur_gray)
    #detect edges in the img with thresholds
    low_threshold = 50
    high_threshold = 150
    edges =canny(blur_gray,low_threshold,high_threshold)
    mpimg.imsave(output_route + 'edges.jpg', edges)
    #define polygon
    imshape = image.shape 
    vertices = np.array([[(0.05 * imshape[1],imshape[0]),(0.53 * imshape[1], 0.57 * imshape[0]), (0.55 * imshape[1], 0.57 * imshape[0]), (0.95 * imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    mpimg.imsave(output_route + 'masked_edges.jpg', masked_edges)
    
    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25 #minimum number of pixels making up a line
    max_line_gap = 25  # maximum gap in pixels between connectable line segments
    #find lines
    lines_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    mpimg.imsave(output_route + 'lines_image.jpg', lines_image)

    #add lines img with original img
    weighted_image = weighted_img(lines_image, img)
    mpimg.imsave(output_route + 'weighted_image.jpg', weighted_image)
    
    return weighted_image


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:


import os
os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[14]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
#input and output route
input_route = 'test_images/'
output_route = 'test_images_output/'
#read img from test folder
input_imgs = os.listdir(input_route)
    
for img_files in input_imgs:
    img = mpimg.imread(input_route + img_files)
        
    lined_img = image_processing(img)
    
    #reset global variables when one image processing is done
    right_top_prev = None
    right_bottom_prev = None
    left_top_prev = None
    left_bottom_prev = None
    
    plt.imshow(lined_img)
    plt.show()
    
    mpimg.imsave(output_route + 'result_' + img_files, lined_img)
    


## ## Test on Videos
## 
## You know what's cooler than drawing lanes over images? Drawing lanes over video!
## 
## We can test our solution on two provided videos:
## 
## `solidWhiteRight.mp4`
## 
## `solidYellowLeft.mp4`
## 
## **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
## 
## **If you get an error that looks like this:**
## ```
## NeedDownloadError: Need ffmpeg exe. 
## You can download it by calling: 
## imageio.plugins.ffmpeg.download()
## ```
## **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**
#
## In[15]:
#
#
## Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#
#
## In[16]:
#
#
#def process_image(image):
#    # NOTE: The output you return should be a color image (3 channel) for processing video below
#    # TODO: put your pipeline here,
#    # you should return the final output (image where lines are drawn on lanes)
#    result = image_processing(image)
#    return result
#
#
## Let's try the one with the solid white lane on the right first ...
#
## In[17]:
#
#
##reset global variabals
#right_top_prev = None
#right_bottom_prev = None
#left_top_prev = None
#left_bottom_prev = None
#white_output = 'test_videos_output/solidWhiteRight.mp4'
### To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
### To do so add .subclip(start_second,end_second) to the end of the line below
### Where start_second and end_second are integer values representing the start and end of the subclip
### You may also uncomment the following line for a subclip of the first 5 seconds
###clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')
#
#
## Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.
#
## In[18]:
#
#
#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(white_output))
#
#
## ## Improve the draw_lines() function
## 
## **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
## 
## **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**
#
## Now for the one with the solid yellow lane on the left. This one's more tricky!
#
## In[19]:
#
#
##reset global variabals
#right_top_prev = None
#right_bottom_prev = None
#left_top_prev = None
#left_bottom_prev = None
#yellow_output = 'test_videos_output/solidYellowLeft.mp4'
### To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
### To do so add .subclip(start_second,end_second) to the end of the line below
### Where start_second and end_second are integer values representing the start and end of the subclip
### You may also uncomment the following line for a subclip of the first 5 seconds
###clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
#clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
#yellow_clip = clip2.fl_image(process_image)
#get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')
#
#
## In[20]:
#
#
#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(yellow_output))
#
#
## ## Writeup and Submission
## 
## If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
## 
#
## ## Optional Challenge
## 
## Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!
#
## In[21]:
#
#
##reset global variabals
#right_top_prev = None
#right_bottom_prev = None
#left_top_prev = None
#left_bottom_prev = None
#challenge_output = 'test_videos_output/challenge.mp4'
### To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
### To do so add .subclip(start_second,end_second) to the end of the line below
### Where start_second and end_second are integer values representing the start and end of the subclip
### You may also uncomment the following line for a subclip of the first 5 seconds
###clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
#clip3 = VideoFileClip('test_videos/challenge.mp4')
#challenge_clip = clip3.fl_image(process_image)
#get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')
#
#
## In[22]:
#
#
#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(challenge_output))
#
