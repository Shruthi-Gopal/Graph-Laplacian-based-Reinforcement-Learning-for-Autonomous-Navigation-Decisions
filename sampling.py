import glob
import os
import sys
import time
import carla
import random
from sklearn.cluster import KMeans
import numpy as np
import csv
import numpy as np
from fcmeans import FCM
import skfuzzy as fuzz
import math
import numpy as np
from carla import VehicleLightState as vls
from numpy import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

collisioncount=0
# Connect to the CARLA simulator
client = carla.Client("localhost", 2000)

# Get a sample world
world = client.get_world()
world.tick()
n = int(sys.argv[1])
vehicle = world.get_actors(actor_ids=[n])[0]
collision_bp = world.get_blueprint_library().find('sensor.other.collision')
collision_sensor = world.spawn_actor(collision_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
world.tick()

#REWARDS:
def reward_function(vehicle):
    # Reward for staying within the speed limit
    velocity = vehicle.get_velocity()
    speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
    #waypoint = world.get_map().get_waypoint(vehicle.get_transform().location)
    speed_limit= 60 #waypoint.get_speed_limit()
    speed_reward = 0.05*speed if speed <= speed_limit else -0.2

    # Reward for smooth driving
    acceleration = vehicle.get_acceleration().x
    smoothness_reward = -0.01 * abs(acceleration)

    safety_reward = 0.8 if collisioncount==0 else -0.1*collisioncount
    # Compute the total reward
    reward = speed_reward + smoothness_reward + safety_reward

    return [safety_reward,speed_reward,smoothness_reward,reward]

def get_state(world,vehicle):
    # Getting position
    vehicle_location = vehicle.get_location()
    vehicle_x, vehicle_y, vehicle_z = vehicle_location.x, vehicle_location.y, vehicle_location.z
    # Getting velocity
    v = vehicle.get_velocity()
    speed = (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

    neighbouring_vehicles=[]

    location1 = vehicle.get_transform().location

    vehicles = world.get_actors().filter('vehicle.*')
    for car in vehicles:
        if(car.id != vehicle.id):
            location2 = car.get_transform().location
            distance = location1.distance(location2)
            if (distance<10):
                ve = car.get_velocity()
                s = (3.6 * math.sqrt(ve.x**2 + ve.y**2 + ve.z**2))
                neighbouring_vehicles.append([float(distance),float(s)])

    feature_vector = [float(vehicle_x), float(vehicle_y), float(vehicle_z), float(speed), len(neighbouring_vehicles)]
    #print("FS:",feature_vector)
    return feature_vector

def on_collision(event):
    global collisioncount
    collisioncount+=1


def main():

    # Define the possible actions for the agent

    actions = [0,1,2,3,4,5,6,7,8]
    #forward = 0, left = 1,right = 2,forward_left = 3,forward_right = 4,brake = 5,brake_left = 6,brake_right = 7,no_action = 8
    
    #collision_sensor=world.get_actors(actor_ids=[n+1])[0]
    collision_sensor.listen(lambda event: on_collision(event))
    # Initialize the state
    state = get_state(world,vehicle)
    states=[]
    states.append(state)
    global collisioncount
    tuples=[]
    # Start the episode
    for i in range(1000):
        collisioncount=0
        control = vehicle.get_control()
        t=control.throttle
        b=control.brake
        s=control.steer
        action = 0
        if(t>0 and s==0):
            action=0
        elif(t>0.1 and s<0):
            action=3
        elif(t>0.1 and s>0):
            action=4
        elif(b>0.1 and s<0):
            action=6
        elif(b>0.1 and s>0):
            action=7
        elif(b>0.1):
            action=5
        elif(s<0):
            action=1
        elif(s>0):
            action=2
        else:
            action=0

        time.sleep(1)
        
        # Get the next state
        next_state = get_state(world,vehicle)

        # Get the reward
        reward = reward_function(vehicle)

        # Store the s, a, r, s' tuple
        tuples.append([state, action, reward, next_state])
        print([state, action, reward, next_state])
        print('\n')
        # Update the state
        state = next_state
        states.append(state)

    #print(tuples)
    fields=["state","action","reward","next state"]
    with open("sample.csv","w") as f:
        write=csv.writer(f)
        write.writerow(fields)
        write.writerows(tuples)

    states1 = np.array(states, dtype=object)  # Convert the nested list to a 2D numpy array
    print("states =",states1)
    states1 = states1.astype(float)

    my_model = FCM(n_clusters=50) # we use two cluster as an example
    my_model.fit(states1) ## X, numpy array. rows:samples columns:features   

    centers = my_model.centers

    for item in tuples:
        item[0]=my_model.predict(np.array(item[0]))
        item[3]=my_model.predict(np.array(item[3]))

    with open("sample_clustered.csv","w") as f:
        write=csv.writer(f)
        write.writerow(fields)
        write.writerows(tuples)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
