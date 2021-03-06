# **Finding Lane Lines on the Road** 


[//]: # (Image References)

[color_filtered]: ./writeup_img/color_filtered.jpg "color_filtered"
[gray_img]: ./writeup_img/gray_img.jpg "gray_img"
[blur_gray]: ./writeup_img/blur_gray.jpg "blur_gray"
[edges]: ./writeup_img/edges.jpg "edges"
[masked_edges]: ./writeup_img/masked_edges.jpg "masked_edges"
[lines_image]: ./writeup_img/lines_image.jpg "lines_image"
[weighted_image]: ./writeup_img/weighted_image.jpg "weighted_image"


### Reflection

### 1. Pipeline

My pipeline consisted of 5 steps. 

#### 1) Color filter
     
Apply a color mask to filter out the yellow and white colors in the image

```
def color_filter(img):
    "Add a color fiter for white and yellow"
    color_low = np.array([180,180,0])
    color_high = np.array([255,255,255])
    color_mask = cv2.inRange(img, color_low, color_high)
    filtered_img = cv2.bitwise_and(img, img, mask = color_mask)
    return filtered_img
```
![alt text][color_filtered]
#### 2) Grayscale image
     
Convert the filtered image to grayscale
![alt text][gray_img]
	 
#### 3) Blur the image
     
Remove the gray image noise with 'kernel_size = 5'
![alt text][blur_gray]
	 
#### 4) Edges detection
     
Use canny edge detection to find all edges in the image
![alt text][edges]
	 
#### 5) Region of interest
     
Define a polygon as our Region of interest to focus on the lane part
![alt text][masked_edges]
	 
#### 6) Lines detection
     
Use hough transform to find the lines in the prcessed image
	 
#### 7) Draw the line

In this part, in order to draw a single line on the left and right lanes, I modified the draw_lines() function.
     
a. According to the detected lines' slopes we can assign them to the left group(`slope < 0`) and right group(`slope > 0`). Filter out the lines that is obviously wrong.

```
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
```
					
b. Use the length of filtered lines as weight and add them together to calculate final line's slope and center. Then calculate the final line based on predefined bottom and top y.

```
top_y = 0.6 * img.shape[0]
bottom_y = img.shape[0]
```

```
left_top,left_bottom = cal_top_bottom(left_slope,left_center,left_len,top_y,bottom_y,left_top_prev,left_bottom_prev)
```

```
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
```

c.(For video use only) Apply a low pass filter on consecution frames line calculation. This make the line change between frames soomthly and handle the frames which has no line detected.

```
def low_pass_filter(x_current,y_prev,a = 0.1):
    """ Smooth the lines difference between frames """
    if y_prev != None:
        y_current = a * x_current + (1-a) * y_prev
    else:
        y_current = x_current
    
    return y_current
```

	 
d. Draw the lines	 

![alt text][lines_image]

#### 8) Add lined image to original image

![alt text][weighted_image]

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
