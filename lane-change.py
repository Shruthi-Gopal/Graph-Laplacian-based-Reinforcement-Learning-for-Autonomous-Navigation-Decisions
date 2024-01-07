import carla
import time
import numpy as np
import sys

#Connecting to CARLA using a client and loading the world
client = carla.Client('localhost',2000)
world = client.get_world()

#Getting the vehicle from Command-Line Argument
n = int(sys.argv[1])
vehicle = world.get_actors(actor_ids = [n])[0]

def main():
    #Finding the current lane of the vehicle
    current_lane_id = world.get_map().get_waypoint(vehicle.get_location()).lane_id
    print(current_lane_id)
    sign = current_lane_id/abs(current_lane_id)
    if current_lane_id == 1:
        desired_lane_id = 2
    else:
        desired_lane_id = 1

    # Define the lane change trajectory (move to the left lane for 3 seconds)
    lane_dif = desired_lane_id - current_lane_id
    #print(vehicle.get_transform().rotation.as_euler('xyz'))
    target_lane_position = vehicle.get_transform().location.y + 2.0*lane_dif 
    duration = 2.75

    #Defining Steer array
    Steer = []
    # Perform the lane change
    start_time = world.get_snapshot().timestamp.elapsed_seconds
    while world.get_snapshot().timestamp.elapsed_seconds - start_time < duration:
        # Calculate the steering angle to move towards the target lane position
        current_position = vehicle.get_transform().location.y
        target_offset = target_lane_position - current_position
        steering_angle = target_offset * 0.1  # Adjust this value to control the steering sensitivity

        Steer.append(steering_angle)
        # Set the steering and throttle/brake input to the vehicle
        control = carla.VehicleControl(throttle=0.5, steer=steering_angle)
        vehicle.apply_control(control)

        # Wait for a small time interval to simulate real-time control
        world.wait_for_tick()

    # Steer in opposite direction to orient the car parallel to the road
    size = len(Steer)
    i = 0
    start_time = world.get_snapshot().timestamp.elapsed_seconds
    while world.get_snapshot().timestamp.elapsed_seconds - start_time < duration/2:
        # Set the steering and throttle/brake input to the vehicle
        control = carla.VehicleControl(throttle= 0.5, steer=-Steer[size-1-i])
        vehicle.apply_control(control)

        # Wait for a small time interval to simulate real-time control
        world.wait_for_tick()
        i += 1

    # Stop the vehicle after the lane change
    control = carla.VehicleControl(throttle=0, brake=1.0)
    vehicle.apply_control(control)

    #print(vehicle.get_transform().rotation.as_euler('xyz'))

    while True:
        a = 1

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the vehicle and cleanup
        vehicle.destroy()