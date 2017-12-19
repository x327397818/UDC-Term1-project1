# **Finding Lane Lines on the Road** 


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"


### Reflection

### 1. Pipeline

My pipeline consisted of 5 steps. 

#### 1) Color filter
     
Apply a color mask to filter out the yellow and white colors in the image
	 
#### 2) Grayscale image
     
Convert the filtered image to grayscale
	 
#### 3) Blur the image
     
Remove the gray image noise with 'kernel_size = 5'
	 
#### 4) Edges detection
     
Use canny edge detection to find all edges in the image
	 
#### 5) Region of interest
     
Define a polygon as our Region of interest to focus on the lane part
	 
#### 6) Lines detection
     
Use hough transform to find the lines in the prcessed image
	 
#### 7) Draw the line

In this part, in order to draw a single line on the left and right lanes, I modified the draw_lines() function.
     
	 a. Calculate all detected line's slope, center and length
	 
	 b. According to the slope we can assign them to the left group(`slope < 0`) and right group(`slope > 0`). Filter out the lines that is obviously wrong.
	 
	 c. Use the length of filtered lines as weight and add them together to calculate final line's slope and center. Then calculate the final line based on predefined bottom and top y.

     d.(For video use only) Apply a low pass filter on consecution frames line calculation. This make the line change between frames soomthly and handle the frames which has no line detected.
	 
	 e. Draw the lanes	 


If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline
- 1) Use the color filter.
     If the lane is in other very different color(red/black...), system cannot work. But in real life it is usually yellow and white like color.
	 
- 2) When light changes to dark/bright, performance may not good

- 3) Region of interest and Hough transform parameters are tuned based on test image/video. 
     On other source of image/video it may not have good performance
	 

### 3. Suggest possible improvements to your pipeline

- 1) Image processing can get other color model(HSL/HSV...) involved to reduce color and lights impact

- 2) Parameter tuning may work in automatic way.

- 3) Some advanced algorithm(Kalman...) can be used to do soomthing and prediction of lines between frames
