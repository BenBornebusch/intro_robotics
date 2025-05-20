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

#HELPER FUNCTION
def show_image(image):
    plt.imshow(image)
    plt.show()


# HSV range for detecting brown (trash)


# Starts simulation
sim.startSimulation()
time.sleep(0.5)

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



def find_color_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    GREEN_LOWER = np.array([40, 50, 50])
    GREEN_UPPER = np.array([80, 255, 255])
    
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    
    result = cv2.bitwise_and(frame, frame, mask = mask)
    
    crop_rows = 10
    if result.shape[0] > crop_rows:
        cropped_result = result[crop_rows:, :, :]
    else:
        print("not tall enough")
        cropped_result = result 

    if np.any(mask > 0):
        return True, cropped_result
    else:
        return False, cropped_result

def find_charching_pat(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    YELLOW_LOWER = np.array([20, 100, 100])
    YELLOW_UPPER = np.array([30, 255, 255])
    
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    
    result = cv2.bitwise_and(frame, frame, mask = mask)
    
    crop_rows = 10
    if result.shape[0] > crop_rows:
        cropped_result = result[crop_rows:, :, :]
    else:
        print("not tall enough")
        cropped_result = result 

    if np.any(mask > 0):
        return True, cropped_result
    else:
        return False, cropped_result

def find_color_brown(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    BROWN_LOWER = np.array([10, 150, 50])
    BROWN_UPPER = np.array([25, 255, 150])

    mask = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER)
    result = cv2.bitwise_and(frame, frame, mask=mask)

 
    crop_rows = 10
    if result.shape[0] > crop_rows:
        cropped_result = result[crop_rows:, :, :]
    else:
        print("not tall enough")
        cropped_result = result 

    if np.any(mask > 0):
        return Tru
def find_color_black(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    BLACK_LOWER = np.array([0, 0, 0])
    BLACK_UPPER = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)

    mask = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)
    result = cv2.bitwise_and(frame, frame, mask=mask)
 
    result_cropped = cropping_an_image(result, 30)
    
    if np.any(mask > 0):
        return True, result_cropped
    else:
        return False, result_cropped
e, cropped_result
    else:
        return False, cropped_result

def going_to_charching():
    bottom_frame = get_image_bottom()
    found_bottom, yellow_mask_bottom = find_charching_pat(bottom_frame)
    if not found_bottom:
        left_motor.run(1)
        right_motor.run(-1)
            re    

    # Find the horizontal (x) positions of yellow     ls
    yellow_pixels = np.where(cv2.cvtColor(yellow_mask_bottom, cv2.COLOR_BGR2        urn

    # Use the mean x position of yellow pixels to estimate t    enter
    cx = int(np.mean(yellowif brown_pixels[1].size > 3300:
        return True
    else:_p        s[1])) # yellow_pixels[1] gives back the x-ccordinates of the yello        xels, with mean it calculates the average pixel     dinate
    width = yellow_mask_bottom.shape[1] # the width of the ori    l image
    center_x = width //         center o    e image
    tolerance = width // 10  # 10% of im    width            if abs(cx - center_x) <        ce:
             eft_mot        )
        right_motor        
        print(brown_pixels[1].size)            
    elif cx < c         
                  ot          )
                t_moto      1)
                           left_motor.r                    ight_motor.run(0.5)
            rn


def getting_        dinates            e_to_robot(object:                #position object 

        return False
    
    
def going_to_compressed_cube():
    bottom_frame = get_image_bottom()
    found_bottom, black_mask_bottom = find_color_black(bottom_frame)
    if not found_bottom:
        left_motor.run(1)
        right_motor.run(-1)
        return
    
    # Find the horizontal (x) positions of yellow pixels
    black_pixels = np.where(cv2.cvtColor(black_mask_bottom, cv2.COLOR_BGR2GRAY) > 0)
    if black_pixels[1].size == 0:
        left_motor.run(1)
        right_motor.run(-1)
        return
    if black_pixels[1].size > 3300:
        return True
    else:
        # Use the mean x position of yellow pixels to estimate the center
        cx = int(np.mean(black_pixels[1])) # yellow_pixels[1] gives back the x-ccordinates of the yellow pixels, with mean it calculates the average pixel coordinate
        width = black_mask_bottom.shape[1] # the width of the original image
        center_x = width // 2 # center of the image
        tolerance = width // 10  # 10% of image width
        print(black_pixels[1].size)
        if abs(cx - center_x) < tolerance:
            left_motor.run(1)
            right_motor.run(2)
            
        elif cx < center_x:
            left_motor.run(0.5)
            right_motor.run(1)
        else:
            left_motor.run(1)
            right_motor.run(0.5)

        return False
    sim.getObject(object)
    pos_ob = sim.getObjectPosition(handle_ob, -1)
    
    #position robot
    handle_robot = sim.getObject("/dr12")
    pos_robot = sim.getObjectPosition(handle_robot, -1)
    
    distance = math.sqrt((pos_ob[0]-pos_robot[0])**2 + (pos_ob[1]-pos_robot[1])**2)
    return distance



# MAIN LOOP
while True:
    # Get battery level
    print(f"Battery: {robot.get_battery():.2f}")

    image_top = get_image_top()
    image_bottom = get_image_bottom()
    
    green_mask_bottom = find_color_brown(image_bottom)
    
    hallo, brown_mask_top =find_color_brown(image_top)
    hallo, brown_mask_bottom =find_charching_pat(image_bottom)


    image_bottom_cropped = image_bottom[10:, :, :]
    image_top_cropped = image_top[10:, :, :]
    
    
    combined = np.hstack((image_top_cropped, brown_mask_bottom))

    cv2.namedWindow("Combined Camera View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Combined Camera View", 1200, 500)
    cv2.imshow("Combined Camera View", combined)

    if cv2.waitKey(1) == ord("q"):
        break

    cv2.waitKey(1)
    
    coordinates = getting_coordinates_relative_to_robot("/Charging_Pad")
    
    if coordinates > 0.29:
        print(coordinates)
        going_to_charching()
        
    else:
        left_motor.run(0)
        right_motor.run(0)
        
       






if :
        goinggoing_to_trash_cube())
        bolbol = goinggoing_to_trash_cube())
    if bnot bol:bol == FalsTrue:
        lefleft_motor.run()0
        righright_motor.rinun()0
        else:
        goinggoing_to_trash_cube())
        robot.ccompress())if robot.getget_battery()) >= 20:
                                                                                
                else: 
        whileWhile while robot.getget_battery()) != 100:
            coordinates_trash = getting_coordinates_relative_to_robot("/Trash_Can/Trash_Can_Floor")
    '''
    if coordinates_trash > 0.29:
        print(coordinates_trash)
        going_to_trash()
    else:
        left_motor.run(0)
        right_motor.run(0)                                                                        
                                                                                                    Charching_Padgcharchingcoordinates_charchingcoordinates_charching5chgoing_to_charching
            
    #main  Main function process 55image_bottom_croppedimage_top_cropped
    robot_battery = robot.getget_battery():.2f
    print()roborobot_battery0.2.0
    
    2020553030
            trah_detectedctiontrah_detectionif trcompressed_trash_detection = goinggoing_to_compressed_cube())
            if comprcompressed_trash_detection == True:
                goinggoing_to_trash())
                else:
                goinggoing_to_compressed_cube())
            while compcompileresscompressed_trash_detection == False:                                                                        
                    compressed_trash_detection = going_to_compressed_cube()black
    imageimage_bottomrimage_brown = cropcropping_an_image()imaimageimage_brown, 30image_brownq