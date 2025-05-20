from robots import *
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require("sim")

# HANDLES FOR ACTUATORS AND SENSORS
left_motor = Motor(sim, DeviceNames.MOTOR_LEFT_LINE, Direction.CLOCKWISE)
right_motor = Motor(sim, DeviceNames.MOTOR_RIGHT_LINE, Direction.CLOCKWISE)
color_sensor = ImageSensor(sim, DeviceNames.IMAGE_SENSOR_LINE)

def is_red_detected(color_sensor):
    """
    Calculates the relative intensity of the red channel compared to
    other channels
    """
    red_ratio_threshold = 1.5
    red, green, blue = color_sensor.rgb()
    print(red, green, blue)
    red_intensity = red / (green + blue)

    return red_intensity > red_ratio_threshold


def is_blue_detected(color_sensor):
    """
       Calculates the relative intensity of the blue channel compared to
       other channels
       """
    blue_ratio_threshold = 1.5
    red, green, blue = color_sensor.rgb()
    blue_intensity = blue / (red + green)

    return blue_intensity > blue_ratio_threshold

prev_error = 0

def follow_line():
    global prev_error
    
    color_sensor._update_image()  # Updates internal image
    reflection = color_sensor.reflection()  # Gets reflection from image
    print(reflection)
    
    error = reflection - 50
    K = 0.2
    D = 0.15  
    base_speed = 6
    corner_adjust = max(2, min(5, abs(error) * 0.2))  
    
    derivative = error - prev_error
    prev_error = error
    
    correction = K * (error + D * derivative)
    
    if error < 0:
        right_motor.run(speed=base_speed + correction)
        left_motor.run(speed=base_speed - corner_adjust)  
    elif error > 0:
        right_motor.run(speed=base_speed + correction)
        left_motor.run(speed=base_speed - corner_adjust)
    


          

# Start simulation if not already running
sim.startSimulation()

while True:
    follow_line()