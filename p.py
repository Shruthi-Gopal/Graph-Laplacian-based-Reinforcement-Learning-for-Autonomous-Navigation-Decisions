import glob
import os
import sys
import math
import carla
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from carla import VehicleLightState as vls
from numpy import random

n = int(sys.argv[1])
print(n)
client = carla.Client("localhost", 2000)

# Get a sample world
world = client.get_world()
ego_vehicle = world.get_actors(actor_ids=[n])[0]

import math
import time

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world and map
world = client.get_world()
carla_map = world.get_map()

# Get the ego vehicle's current location and orientation
ego_transform = ego_vehicle.get_transform()
ego_location = ego_transform.location
ego_rotation = ego_transform.rotation

# Get the road map
map_data = carla_map.get_spawn_points()
waypoints = carla_map.generate_waypoints(1.0)

## Determine the closest waypoint to the ego vehicle
#closest_waypoint = None
#closest_distance = float('inf')
#for waypoint in waypoints:
#    distance = waypoint.transform.location.distance(ego_location)
#    if distance < closest_distance:
#        closest_distance = distance
#        closest_waypoint = waypoint
        
#print(closest_distance)

#DETECT WAYPOINTS

# define a threshold angle for detecting turns
threshold_angle = 30.0

# iterate over the waypoints to find the nearest turn
nearest_turn = None
dist=0

control = carla.VehicleControl()
control.throttle = 0.5
ego_vehicle.apply_control(control)

while(True):

    # get the current waypoint of the vehicle
    curr_waypoint = world.get_map().get_waypoint(ego_vehicle.get_location())

    # get the next waypoint on the route
    next_waypoint = curr_waypoint.next(10)[0]

    # calculate the angle between the orientations of the two waypoints
    orientation1 = curr_waypoint.transform.rotation.yaw
    orientation2 = next_waypoint.transform.rotation.yaw
    angle_between_waypoints = abs(orientation1 - orientation2)

    velocity = ego_vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    #print("Angle between waypoints:", angle_between_waypoints)
    if(angle_between_waypoints>1):
        location = ego_vehicle.get_transform().location
        distance = location.distance(next_waypoint.transform.location)
        time.sleep(distance/speed+0.8)
        #control.throttle = 0
        control.steer = 73
        ego_vehicle.apply_control(control)
        time.sleep(0.829)
        control.steer = 0
        ego_vehicle.apply_control(control)
        time.sleep(2)



#waypoint = waypoint.next(100.0)
#print(waypoint)
#control = carla.VehicleControl()
#control.throttle = 1.0
#ego_vehicle.apply_control(control)
#time.sleep(0.001)
#ego_transform = ego_vehicle.get_transform()
#ego_location = ego_transform.location
#distance = waypoint.transform.location.distance(ego_location)
#velocity = ego_vehicle.get_velocity()
#print(velocity)
#speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
#print(speed)
#timee=distance/speed
#print(timee)
#time.sleep(timee)
## Calculate the curvature of the road at the closest waypoint
#radius_of_curvature = closest_waypoint.lane_width / math.sin(math.radians(closest_waypoint.lane_width / 2))

#curvature = 1 / radius_of_curvature

## Determine the direction of the road at the closest waypoint
#road_direction = closest_waypoint.transform.rotation.get_forward_vector()
#ego_forward = ego_rotation.get_forward_vector()
#dot_product = road_direction.x * ego_forward.x + road_direction.y * ego_forward.y + road_direction.z * ego_forward.z
#if dot_product < 0:
#    print('The road is turning left with curvature:', curvature)
#    steering_angle = -math.atan((2.5 / 2) / radius_of_curvature) # Example steering angle calculation for left turn
#else:
#    print('The road is turning right with curvature:', curvature)
#    steering_angle = math.atan((2.5 / 2) / radius_of_curvature) # Example steering angle calculation for right turn
#steering_angle*=40    
## Set the vehicle's steering angle to the desired angle
#control = carla.VehicleControl()
#control.steer = steering_angle
#control.throttle = 1.0
#ego_vehicle.apply_control(control)
#time.sleep(0.85)
#control.steer = 0
#ego_vehicle.apply_control(control)