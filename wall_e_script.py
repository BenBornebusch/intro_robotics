from robots import *
import time
from coppeliasim_zmqremoteapi_client import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

client = RemoteAPIClient()
sim = client.require("sim")

# HANDLES FOR ACTUATORS AND SENSORS
robot = Robot_OS(sim, DeviceNames.ROBOT_OS)

top_image_sensor = ImageSensor(sim, DeviceNames.TOP_IMAGE_SENSOR_OS)
small_image_sensor = ImageSensor(sim, DeviceNames.SMALL_IMAGE_SENSOR_OS)

left_motor = Motor(sim, DeviceNames.MOTOR_LEFT_OS, Direction.CLOCKWISE)
right_motor = Motor(sim, DeviceNames.MOTOR_RIGHT_OS, Direction.CLOCKWISE)


#CONVERTING THE RAW_IMAGE AND RESOLUTION TO AN 
def convert_image(raw_image, resolution):
    # Convert float list (range 0.0–1.0) to uint8 (range 0–255)
    img = np.array(raw_image, dtype=np.float32)
    img *= 255  #scalling up 
    img = img.astype(np.uint8) # data type used for cv2 and matplotlib

    img = img.reshape((resolution[1], resolution[0], 3)) #reshape 1D array into 3D image with shape

    img = np.flip(img, axis=0)#flipping image cause usually upside down

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)#flipping color for cv2

    return img

def get_masked_image(frame, color):
    func_name = f"find_color_{color}"
    func = globals().get(func_name)
    if callable(func):
        image = func(frame)
        return image
    else:
        print(f"No function found for color: {color}")
        
#HELPER FUNCTION
def show_image(image):
    plt.imshow(image)
    plt.show()

def get_image_top():
    sensor_handle_top = top_image_sensor._handle #the top_image_sensor is the object of higher level 
    
    raw_image_top = sim.getVisionSensorImage(sensor_handle_top)
    resolution_top = sim.getVisionSensorResolution(sensor_handle_top)

    image_top = convert_image(raw_image_top, resolution_top)
    
    return image_top
    
def get_image_bottom():
    sensor_handle_bottom = small_image_sensor._handle
    raw_image_bottom = sim.getVisionSensorImage(sensor_handle_bottom)
    resolution_bottom = sim.getVisionSensorResolution(sensor_handle_bottom)

    image_bottom = convert_image(raw_image_bottom, resolution_bottom)
    
    return image_bottom

def cropping_an_image(image, crop_rows = 10):
    if image.shape[0] > crop_rows:
        cropped_result = image[crop_rows:, :, :]
    else:
        print("not tall enough")
        cropped_result = image
        
    return cropped_result 

def find_color_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    GREEN_LOWER = np.array([40, 50, 50])
    GREEN_UPPER = np.array([80, 255, 255])
    
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    
    result = cv2.bitwise_and(frame, frame, mask = mask)
    
    if np.any(mask > 0):
        return True, result
    else:
        return False, result

def find_color_yellow(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    YELLOW_LOWER = np.array([20, 100, 100])
    YELLOW_UPPER = np.array([30, 255, 255])
    
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    
    result = cv2.bitwise_and(frame, frame, mask = mask)
    
    if np.any(mask > 0):
        return True, result
    else:
        return False, result

def find_color_blue(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    BLUE_LOWER = np.array([90, 200, 100])
    BLUE_UPPER = np.array([105, 255, 180])


    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    result = cv2.bitwise_and(frame, frame, mask=mask)
 
    if np.any(mask > 0):
        return True, result
    else:
        return False, result
'''
def find_color_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    RED_LOWER = np.array([0, 100, 100])
    RED_UPPER = np.array([10, 255, 255])


    mask = cv2.inRange(hsv, RED_LOWER, RED_UPPER)
    result = cv2.bitwise_and(frame, frame, mask=mask)
 
    if np.any(mask > 0):
        return True, result
    else:
        return False, result
'''    

def find_color_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 150, 120])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 150, 120])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    if np.any(mask > 0):
        return True, result
    else:
        return False, result

def find_color_brown(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    '''
    BROWN_LOWER = np.array([10, 80, 50]) #sadly it recognizes the charging thingy
    BROWN_UPPER = np.array([30, 255, 200])
    '''
    
    BROWN_LOWER = np.array([10, 150, 50])
    BROWN_UPPER = np.array([25, 255, 150])

    mask = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER)
    result = cv2.bitwise_and(frame, frame, mask=mask)
 
    if np.any(mask > 0):
        return True, result
    else:
        return False, result
      
def find_color_black(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    BLACK_LOWER = np.array([0, 0, 0])
    BLACK_UPPER = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)

    mask = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)
 
    red_on_black = np.zeros_like(frame)
    
    red_on_black[mask > 0] = (0, 0, 255)  

    result_cropped = cropping_an_image(red_on_black, 30)

    if np.any(mask > 0):
        return True, result_cropped
    else:
        return False, result_cropped
    
def going_to_green_cube():
    bottom_frame = get_image_bottom()
    found_bottom, green_mask_bottom = find_color_green(bottom_frame)
    if not found_bottom:
        left_motor.run(1)
        right_motor.run(-1)
        return
    
    # Find the horizontal (x) positions of yellow pixels
    green_pixels = np.where(cv2.cvtColor(green_mask_bottom, cv2.COLOR_BGR2GRAY) > 0)
    if green_pixels[1].size > 3300:
        print("skibidi")
        for i in range(5):
            left_motor.run(1)
            right_motor.run(1)
        return True
    else:
        # Use the mean x position of yellow pixels to estimate the center
        cx = int(np.mean(green_pixels[1])) # yellow_pixels[1] gives back the x-ccordinates of the yellow pixels, with mean it calculates the average pixel coordinate
        width = green_mask_bottom.shape[1] # the width of the original image
        center_x = width // 2 # center of the image
        tolerance = width // 10  # 10% of image width
        print("green pixels sighted: " + str(green_pixels[1].size))
        if abs(cx - center_x) < tolerance:
            left_motor.run(2)
            right_motor.run(2)
            
        elif cx < center_x:
            left_motor.run(0.5)
            right_motor.run(1)
        else:
            left_motor.run(1)
            right_motor.run(0.5)

        return False

def going_to_red_container():
    bottom_frame = get_image_bottom()
    top_frame = get_image_top()
    found_bottom, red_mask_bottom = find_color_red(bottom_frame)
    found_top, red_mask_top = find_color_red(top_frame)
    
    if not found_top:
        left_motor.run(1)
        right_motor.run(-1)
        return
    
    # Find the horizontal (x) positions of yellow pixels
    red_pixels_bottom = np.where(cv2.cvtColor(red_mask_bottom, cv2.COLOR_BGR2GRAY) > 0)
    red_pixels_top = np.where(cv2.cvtColor(red_mask_top, cv2.COLOR_BGR2GRAY) > 0)
    
    if red_pixels_bottom[1].size > 3000:
        return True
    else:
        # Use the mean x position of yellow pixels to estimate the center
        cx = int(np.mean(red_pixels_top[1])) # yellow_pixels[1] gives back the x-ccordinates of the yellow pixels, with mean it calculates the average pixel coordinate
        width = red_mask_top.shape[1] # the width of the original image
        center_x = width // 2 # center of the image
        tolerance = width // 10  # 10% of image width
        print("red pixels sighted: " + str(red_pixels_top[1].size))
        if abs(cx - center_x) < tolerance:
            left_motor.run(3)
            right_motor.run(3)
            
        elif cx < center_x:
            left_motor.run(0.5)
            right_motor.run(1)
        else:
            left_motor.run(1)
            right_motor.run(0.5)

        return False

    
def going_to_blue_container():
    bottom_frame = get_image_bottom()
    top_frame = get_image_top()
    found_bottom, blue_mask_bottom = find_color_blue(bottom_frame)
    found_top, blue_mask_top = find_color_blue(top_frame)
    if not found_top:
        left_motor.run(1)
        right_motor.run(-1)
        return
    
    # Find the horizontal (x) positions of yellow pixels
    blue_pixels_bottom = np.where(cv2.cvtColor(blue_mask_bottom, cv2.COLOR_BGR2GRAY) > 0)
    blue_pixels_top = np.where(cv2.cvtColor(blue_mask_top, cv2.COLOR_BGR2GRAY) > 0)
    
    if blue_pixels_bottom[1].size > 3300:
        return True
    else:
        # Use the mean x position of yellow pixels to estimate the center
        cx = int(np.mean(blue_pixels_top[1])) # yellow_pixels[1] gives back the x-ccordinates of the yellow pixels, with mean it calculates the average pixel coordinate
        width = blue_mask_bottom.shape[1] # the width of the original image
        center_x = width // 2 # center of the image
        tolerance = width // 10  # 10% of image width
        print("blue pixels sighted: " + str(blue_pixels_top[1].size))
        if abs(cx - center_x) < tolerance:
            left_motor.run(2)
            right_motor.run(2)
            
        elif cx < center_x:
            left_motor.run(0.5)
            right_motor.run(1)
        else:
            left_motor.run(1)
            right_motor.run(0.5)

        return False

def going_to_charching():
    bottom_frame = get_image_top()
    found_bottom, yellow_mask_bottom = find_color_yellow(bottom_frame)
    if not found_bottom:
        left_motor.run(1)
        right_motor.run(-1)
        return
    
    # Find the horizontal (x) positions of yellow pixels
    yellow_pixels = np.where(cv2.cvtColor(yellow_mask_bottom, cv2.COLOR_BGR2GRAY) > 0)
    if yellow_pixels[1].size == 0:
        left_motor.run(1)
        right_motor.run(-1)
        return

    # Use the mean x position of yellow pixels to estimate the center
    cx = int(np.mean(yellow_pixels[1])) # yellow_pixels[1] gives back the x-ccordinates of the yellow pixels, with mean it calculates the average pixel coordinate
    width = yellow_mask_bottom.shape[1] # the width of the original image
    center_x = width // 2 # center of the image
    tolerance = width // 10  # 10% of image width
    print("yellow pixels sighted: " + str(yellow_pixels[1].size))
    if abs(cx - center_x) < tolerance:
        left_motor.run(3)
        right_motor.run(3)
            
    elif cx < center_x:
        left_motor.run(0.5)
        right_motor.run(1)
    else:
        left_motor.run(1)
        right_motor.run(0.5)

def going_to_trash_cube():
    bottom_frame = get_image_bottom()
    bottom_frame = cropping_an_image(bottom_frame)
    found_bottom, brown_mask_bottom = find_color_brown(bottom_frame)
    if not found_bottom:
        left_motor.run(1)
        right_motor.run(-1)
        return
    
    # Find the horizontal (x) positions of yellow pixels
    brown_pixels = np.where(cv2.cvtColor(brown_mask_bottom, cv2.COLOR_BGR2GRAY) > 0)
    if brown_pixels[1].size == 0:
        left_motor.run(1)
        right_motor.run(-1)
        return
    if brown_pixels[1].size > 3400:
        for i in range(10):
            left_motor.run(5)
            right_motor.run(5)
        return True
    else:
        # Use the mean x position of yellow pixels to estimate the center
        cx = int(np.mean(brown_pixels[1])) # yellow_pixels[1] gives back the x-ccordinates of the yellow pixels, with mean it calculates the average pixel coordinate
        width = brown_mask_bottom.shape[1] # the width of the original image
        center_x = width // 2 # center of the image
        tolerance = width // 10  # 10% of image width
        print("brown pixels sighted: " + str(brown_pixels[1].size))
        if abs(cx - center_x) < tolerance:
            left_motor.run(3)
            right_motor.run(3)
            
        elif cx < center_x:
            left_motor.run(0.5)
            right_motor.run(1)
        else:
            left_motor.run(1)
            right_motor.run(0.5)

        return False

def going_to_compressed_cube():
    top_frame = get_image_top()
    bottom_frame = get_image_bottom()
    foubd_top, black_mask_top = find_color_black(top_frame)
    found_bottom, black_mask_bottom = find_color_black(bottom_frame)
    
    if not foubd_top:
        left_motor.run(1)
        right_motor.run(-1)
        return
    
    # Find the horizontal (x) positions of yellow pixels
    black_pixels_top = np.where(cv2.cvtColor(black_mask_top, cv2.COLOR_BGR2GRAY) > 0)
    black_pixels_bottom = np.where(cv2.cvtColor(black_mask_bottom, cv2.COLOR_BGR2GRAY) > 0)
    if black_pixels_top[1].size == 0:
        left_motor.run(1)
        right_motor.run(-1)
        return
    if black_pixels_top[1].size > 500:
        return True
    else:
        if black_pixels_bottom[1].size > 1000:
            return True
        # Use the mean x position of yellow pixels to estimate the center
        cx = int(np.mean(black_pixels_top[1])) # yellow_pixels[1] gives back the x-ccordinates of the yellow pixels, with mean it calculates the average pixel coordinate
        width = black_mask_top.shape[1] # the width of the original image
        center_x = width // 2 # center of the image
        tolerance = width // 10  # 10% of image width
        print("Black pixels sighted: " + str(black_pixels_top[1].size))
        if abs(cx - center_x) < tolerance:
            left_motor.run(1)
            right_motor.run(1)
            
        elif cx < center_x:
            left_motor.run(0.25)
            right_motor.run(1)
        else:
            left_motor.run(1)
            right_motor.run(0.25)

        return False
    
    
def getting_coordinates_relative_to_robot(object):
    handle_ob = sim.getObject(object)
    pos_ob = sim.getObjectPosition(handle_ob, -1)
        
    #position robot
    handle_robot = sim.getObject("/dr12")
    pos_robot = sim.getObjectPosition(handle_robot, -1)
        
    distance = math.sqrt((pos_ob[0]-pos_robot[0])**2 + (pos_ob[1]-pos_robot[1])**2)
    return distance

def get_camera_views(frame_a, frame_b):
    combined = np.hstack((frame_a, frame_b))
    cv2.namedWindow("Combined Camera View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Combined Camera View", 1200, 500)
    cv2.imshow("Combined Camera View", combined)

def turn_of_180():
    for i in range(55):
        left_motor.run(3)
        right_motor.run(-3)

def going_to_small_black_cube():
    bottom_frame = get_image_bottom()
    # Crop to focus on the lower part of the image (avoid sky)
    cropped = cropping_an_image(bottom_frame, crop_rows=bottom_frame.shape[0] // 3)
    found, black_mask = find_color_black(cropped)
    if not found:
        left_motor.run(1)
        right_motor.run(-1)
        return

    black_pixels = np.where(cv2.cvtColor(black_mask, cv2.COLOR_BGR2GRAY) > 0)
    if black_pixels[1].size == 0:
        left_motor.run(1)
        right_motor.run(-1)
        return

    # Thresholds for "small" black cube (adjust as needed)
    if 30 < black_pixels[1].size < 820:
        cx = int(np.mean(black_pixels[1]))
        width = black_mask.shape[1]
        center_x = width // 2
        tolerance = width // 10
        print("Black pixels:", black_pixels[1].size)
        if abs(cx - center_x) < tolerance:
            left_motor.run(1)
            right_motor.run(1)
        elif cx < center_x:
            left_motor.run(0.5)
            right_motor.run(1)
        else:
            left_motor.run(1)
            right_motor.run(0.5)
        return False
    elif black_pixels[1].size >= 820:
        left_motor.run(0)
        right_motor.run(0)
        return True
    else:
        left_motor.run(1)
        right_motor.run(-1)
        return False

# Starts simulation
sim.startSimulation()
time.sleep(0.5)

try:
    print("---Staring the sorting---")
    # MAIN LOOP
    while True:
        if robot.get_battery() >= 0.40:
            bottom_frame = get_image_bottom()
            found_bottom, green_mask_bottom = find_color_green(bottom_frame)
            if not found_bottom:
                if going_to_trash_cube():
                    right_motor.run(0)
                    left_motor.run(0)
                    robot.compress()
                    for i in range(10):
                        right_motor.run(5)
                        left_motor.run(5)
                    turn_of_180()
                    
                    while not going_to_small_black_cube():
                        going_to_small_black_cube()
                        
                    while True:
                            if going_to_red_container():
                                print("compressed cube brought to destination!")
                                for i in range(10):
                                    right_motor.run(-5)
                                    left_motor.run(-5)
                                break
                            else:
                                while not going_to_red_container:
                                    going_to_red_container()    
                else:
                    while not going_to_trash_cube():
                        going_to_trash_cube()
            else:
                while not going_to_green_cube():
                    going_to_green_cube()
                
                if going_to_blue_container():
                    right_motor.run(0)
                    left_motor.run(0)
                    
                else:
                    while not going_to_blue_container():
                        going_to_blue_container()
            
        else:
            while robot.get_battery() != 1.0:
                coordinates_charching = getting_coordinates_relative_to_robot("/Charging_Pad")
                if coordinates_charching > 0.29:
                    going_to_charching()
                else:
                    left_motor.run(0)
                    right_motor.run(0)
                    break
        
except Exception as e:
    print("Exception in main loop:", e)     
    