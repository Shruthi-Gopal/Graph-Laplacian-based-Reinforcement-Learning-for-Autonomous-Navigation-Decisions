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
    vehicle_blueprint = blueprint_library.filter('vehicle.audi.etron')[0]

    # spawn the vehicle at the defined spawn point
    vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
    print(vehicle)
    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)


    world.tick()


    
if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
