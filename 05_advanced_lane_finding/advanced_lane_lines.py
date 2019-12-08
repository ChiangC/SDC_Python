import cv2
import numpy as np
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML



#=========================
def get_objpoints_imgpoints(images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    return objpoints, imgpoints 

#=========================
#Apply a distortion correction to raw images
def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

#========================
#Apply a perspective transform to rectify binary image ("birds-eye view")
def warper(undist):
    offset = 100
    img_size = (undist.shape[1], undist.shape[0])

#     src = np.float32([ [240, 720],[585,460], [695, 460], [1040, 720]])
#     dst = np.float32([[320, 720], [320,0], [960, 0], [960, 720] ])
    
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size)

    return warped, M_inverse

#=========================
#Detect lane pixels and fit to find the lane boundary
def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram

def search_lines_from_prior(binary_warped, left_line, right_line, margin=90):
    if left_line.detected and len(left_line.current_fit) == 3 and right_line.detected and len(right_line.current_fit) == 3:
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_fit = left_line.current_fit
        right_fit = right_line.current_fit
        
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                          & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                           & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        if len(left_lane_inds)>= 450 and len(right_lane_inds) >= 450:
            left_line.detected = True
            right_line.detected = True

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            left_line.allx = leftx
            left_line.ally = lefty

            right_line.allx = rightx
            right_line.ally = righty
#             print("search_lines_from_prior:Success")
        else:
            left_line.detected = False
            right_line.detected = False
            print("search_lines_from_prior:Failed")
        
    return left_line,right_line

def search_by_sliding_window(binary_warped, left_line, right_line):
    print("search_by_sliding_window:")
    histogram = hist(binary_warped)
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
        # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin  
        win_xright_low = rightx_current - margin  
        win_xright_high = rightx_current + margin  

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = (np.where((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)\
                          &(nonzerox >= win_xleft_low)&(nonzerox <= win_xleft_high)))[0]

        good_right_inds = (np.where((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)\
                          &(nonzerox >= win_xright_low)&(nonzerox <= win_xright_high)))[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)


        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if(len(good_left_inds) > minpix ):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if(len(good_right_inds) > minpix):
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print("========Error")

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_line.detected = True
    right_line.detected = True

    left_line.allx = leftx
    left_line.ally = lefty

    right_line.allx = rightx
    right_line.ally = righty
    print("search_by_sliding_window:Done!")
    return left_line, right_line
    
def find_lane_lines(binary_warped, left_line, right_line):
    
    #Search from Prior
    left_line, right_line = search_lines_from_prior(binary_warped, left_line, right_line)
    
    #If lines not detected from prior, search by sliding window
    if(not left_line.detected or not right_line.detected):
        left_line, right_line = search_by_sliding_window(binary_warped, left_line, right_line)
            
    return left_line, right_line        
        
    
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
#     histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    histogram = hist(binary_warped)
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = (np.where((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)\
                          &(nonzerox >= win_xleft_low)&(nonzerox <= win_xleft_high)))[0]

        good_right_inds = (np.where((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)\
                          &(nonzerox >= win_xright_low)&(nonzerox <= win_xright_high)))[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)


        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if(len(good_left_inds) > minpix ):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
#             print("========leftx_current:{}".format(leftx_current))

        if(len(good_right_inds) > minpix):
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
#             print("========rightx_current:{}".format(rightx_current))
        
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

#         print("========left_lane_inds1:")
#         print(left_lane_inds)
        
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print("========Error")

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, out_img

def fit_polynomial_v2(binary_warped, left_line, right_line):
    find_lane_lines(binary_warped, left_line, right_line)
    #second order polynomial work
    left_fit = np.polyfit(left_line.ally,left_line.allx, 2)
    right_fit = np.polyfit(right_line.ally,right_line.allx, 2)
    
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        print("========Error")
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    left_line.recent_xfitted = left_fitx
    right_line.recent_xfitted = right_fitx
    
    leftx = left_line.allx
    lefty = left_line.ally
    
    rightx = right_line.allx
    righty = right_line.ally
    
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    colored_lane_regions = np.zeros_like(out_img)
    colored_lane_regions[lefty, leftx] = [255, 0, 0]
    colored_lane_regions[righty, rightx] = [0, 0, 255]
    
    return left_line, right_line, colored_lane_regions
    
#=========================
#Determine the curvature of the lane and vehicle position with respect to center
def measure_curvature_vehicle_position(binary_warped, left_fit, right_fit, ym_per_pix, xm_per_pix):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    y_eval = np.max(ploty)
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    left_fit_real_world = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_real_world = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    ## Implement the calculation of the left line here
    left_curverad = (1 + (2*left_fit_real_world[0]*y_eval*ym_per_pix + left_fit_real_world[1])**2)**(3/2)/np.abs(2*left_fit_real_world[0])  
    
    ## Implement the calculation of the right line here
    right_curverad = (1 + (2*right_fit_real_world[0]*y_eval*ym_per_pix + right_fit_real_world[1])**2)**(3/2)/np.abs(2*right_fit_real_world[0])  
    
    curvature = (left_curverad + right_curverad)/2
    
    index = binary_warped.shape[0] - 1# For (720*1280) index = 719
    lane_center_x = (((left_fitx[index] + right_fitx[index]) * xm_per_pix) / 2.)
    image_center_x = ((binary_warped.shape[1] * xm_per_pix) / 2.)
    distance_from_center = lane_center_x - image_center_x

    return curvature, distance_from_center

#=========================
#Warp the detected lane boundaries back onto the original image
def warp_lane_boundaries_onto_origin(undistorted, binary_warped, left_line, right_line, M_inverse):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_line.recent_xfitted
    right_fitx = right_line.recent_xfitted
    
    leftx = left_line.allx
    lefty = left_line.ally
    
    rightx = right_line.allx
    righty = right_line.ally
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Colors in the left and right lane regions
    colored_lane_regions = np.zeros_like(np.dstack((binary_warped, binary_warped, binary_warped)))
    colored_lane_regions[lefty, leftx] = [255, 0, 0]
    colored_lane_regions[righty, rightx] = [0, 0, 255]
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inverse, (undistorted.shape[1], undistorted.shape[0])) 
    unwarped_lane_region = cv2.warpPerspective(colored_lane_regions, M_inverse, (undistorted.shape[1], undistorted.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    result = cv2.addWeighted(result, 1, unwarped_lane_region, 0.9, 0)
    return result

#=========================
#Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
def draw_curvature_vehicle_position_info(img, curvature, distance_from_center):
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of Curvature = %sm" % (round(curvature))

    if distance_from_center > 0:
        pos_flag = 'right'
    else:
        pos_flag = 'left'

    cv2.putText(img, radius_text, (36, 36), font, 1.3, (255, 255, 255), 2)
    center_text = "Vehicle is %.3fm %s of center" % (abs(distance_from_center), pos_flag)
    cv2.putText(img, center_text, (36, 72), font, 1.3, (255, 255, 255), 2)
    cv2.putText(img, "chiangchuna@gmail.com", (36, img.shape[0] - 36), font, 1.3, (255, 255, 255), 2)
    return img

#=========================
#Color and Gradient Threshold
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scale_factor = np.max(abs_sobel)/255
    scaled_sobel = np.uint8(abs_sobel/scale_factor)
    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    # 6) Return this mask as your grad_binary image
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    
    #1 Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    #2 Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    
    #3 Calculate the gradient magnitude
    grad_mag = np.sqrt(sobelx**2, sobely**2)
    
    #4 Rescale to 8 bit
    scale_factor = np.max(grad_mag)/255
    grad_mag = (grad_mag/scale_factor).astype(np.uint8)
    
    #5 Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(grad_mag)
    mag_binary[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1
    
    
    return mag_binary

#
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    abs_grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(abs_grad_dir)
    
    # 6) Return this mask as your dir_binary image
    dir_binary[(abs_grad_dir >= thresh[0])&(abs_grad_dir <= thresh[1])] = 1
    
    return dir_binary

#Combine Sobel x,Sobel y, Direction, and Magnitude
def combine_gradient_threshold(image, sobel_kernel=3, abs_sobel_thresh=(0, 255), mag_thresh=(0, 255),
                                    dir_threshold=(0, np.pi/2)):
    #1 Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    
    #Absolute gradient in x and y
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    scaled_sobelx = np.uint8(abs_sobelx*255/np.max(abs_sobelx))
    scaled_sobely = np.uint8(abs_sobely*255/np.max(abs_sobely))

    #Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    abs_grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    #Calculate the gradient magnitude
    grad_mag = np.sqrt(sobelx**2, sobely**2)
    scale_factor = np.max(grad_mag)/255
    grad_mag = (grad_mag/scale_factor).astype(np.uint8)
    
    combined_binary = np.zeros_like(grad_mag)
    
#     combined_binary[((scaled_sobelx >= abs_sobel_thresh[0]) & (scaled_sobelx <= abs_sobel_thresh[1])) ] = 1
    combined_binary[(((scaled_sobelx >= abs_sobel_thresh[0]) & (scaled_sobelx <= abs_sobel_thresh[1])) 
                        & (((scaled_sobely >= abs_sobel_thresh[0]) & (scaled_sobely <= abs_sobel_thresh[1])))) ] = 1
    
    return combined_binary

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


def rgb_select(image, thresh=(0, 255)):
    R = image[:,:,0]
#     G = image[:,:,1]
#     B = image[:,:,2]
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary


#=========================






