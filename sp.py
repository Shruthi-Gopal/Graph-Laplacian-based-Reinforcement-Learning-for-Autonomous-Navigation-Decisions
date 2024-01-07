import glob
import os
import sys

import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from numpy import random

def main():

    # Connect to the CARLA simulator
    client = carla.Client("localhost", 2000)
    world = client.get_world()
    world.tick()
    spawn_points = world.get_map().get_spawn_points()
    print(spawn_points)
    spawn_point = spawn_points[1]

    # define the blueprint for the vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(vehicle)

    # Set the desired throttle and PID parameters
    desired_throttle = 0.5
    Kp_throttle = 0.1
    Ki_throttle = 0.01
    Kd_throttle = 0.001

    # Set the desired steering angle and PID parameters
    desired_angle = 0  # radians
    Kp_angle = 0.1
    Ki_angle = 0.01
    Kd_angle = 0.001

    # Initialize the PID controllers
    error_throttle = 0
    integral_throttle = 0
    derivative_throttle = 0
    previous_error_throttle = 0

    error_angle = 0
    integral_angle = 0
    derivative_angle = 0
    previous_error_angle = 0

    # Loop over the simulation time steps
    while True:
        # Read the current throttle set by the user
        user_throttle = 0.4 # Replace this with your own code to read the user input

        # Calculate the error between the desired throttle and the user throttle
        error_throttle = desired_throttle - user_throttle

        # Update the integral and derivative terms of the throttle PID controller
        integral_throttle += error_throttle
        derivative_throttle = error_throttle - previous_error_throttle
        previous_error_throttle = error_throttle

        # Calculate the throttle command using the throttle PID parameters and error terms
        throttle = Kp_throttle * error_throttle + Ki_throttle * integral_throttle + Kd_throttle * derivative_throttle

        # Read the current steering angle of the vehicle from the CARLA simulator
        current_angle = vehicle.get_control().steer

        # Calculate the error between the desired angle and the current angle
        error_angle = desired_angle - current_angle

        # Update the integral and derivative terms of the steering PID controller
        integral_angle += error_angle
        derivative_angle = error_angle - previous_error_angle
        previous_error_angle = error_angle

        # Calculate the steering command using the steering PID parameters and error terms
        steering = Kp_angle * error_angle + Ki_angle * integral_angle + Kd_angle * derivative_angle

        # Apply the throttle and steering commands to the vehicle's controls
        control = carla.VehicleControl(throttle=throttle, steer=steering)
        vehicle.apply_control(control)



        world.tick()


    
if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
