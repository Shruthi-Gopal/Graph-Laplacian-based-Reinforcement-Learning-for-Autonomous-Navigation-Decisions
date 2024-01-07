import csv
from sklearn.cluster import KMeans
import math
from sklearn.decomposition import PCA
import numpy as np
import re
from scipy.sparse import csr_matrix, csgraph
from scipy.linalg import eig
from sklearn.metrics.pairwise import cosine_similarity
from fcmeans import FCM
import carla
import sys
import time

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

def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# Connect to the CARLA simulator
client = carla.Client("localhost", 2000)

# Get a sample world
world = client.get_world()
world.tick()
n = int(sys.argv[1])
vehicle = world.get_actors(actor_ids=[n])[0]

#Reading from the sample file
csv_file = open('final_sample.csv', 'r')
csv_reader = csv.reader(csv_file)

tuples = []
tup=[]
states = set()
i = 0
for row in csv_reader:
    if i == 0:
        i += 1
        continue
    if len(row) != 0:
        row[0] = [float(number.strip()) for number in row[0].replace("[", "").replace("]", "").split(',')]
        row[3] = [float(number.strip()) for number in row[3].replace("[", "").replace("]", "").split(',')]
        row[1] = float(row[1])
        row[2] = [float(number.strip()) for number in row[2].replace("[", "").replace("]", "").split(',')]
        tuples.append(flatten(row))
        states.add(tuple(row[0]))
        states.add(tuple(row[3]))
        tup.append(row)
states = np.array(list(states))
nc=100
# Initialize the c-means algorithm
my_model = FCM(n_clusters=nc) # we use two cluster as an example
my_model.fit(states)

# predict the cluster labels for the data
labels = my_model.predict(states)
#print(labels)
subsample = []
for i in range(my_model.n_clusters):
    cluster_indices = np.where(labels == i)[0]
    index = list(cluster_indices)[0]
    cluster_data = list(states[index])
    for j in tup:
        if cluster_data == list(j[0]):
            subsample.append(j)
            break

# Print the subsample
print(subsample)

#GRAPH FEATURE LEARNING:
graph = np.zeros([nc, nc], dtype = float)
for i in range(nc):
    node=subsample[i][0]
    for j in range(nc):
        if(i==j):
            continue
        node2=subsample[j][0]
        # Convert the tuples to NumPy arrays
        arr1 = np.array(node).reshape(1, -1)
        arr2 = np.array(node2).reshape(1, -1)

        # Calculate the cosine similarity between the tuples
        similarity = cosine_similarity(arr1, arr2)

        # Print the similarity
        #print(similarity)
        if(similarity[0][0]>=0.7):
            graph[(i,j)]=float(similarity[0][0])
            #print(graph[(i,j)])

#print(graph)
np.save('graph.npy', graph)
#print("saved")
degree_matrix = np.diag(np.sum(graph, axis=1))
laplacian_matrix = degree_matrix - graph
#print(laplacian_matrix)
eigenvalues, eigenvectors = eig(laplacian_matrix)
feature_matrix = np.real(eigenvectors[:, :3])
print("Feature Matrix:\n",feature_matrix,'\n')

# Define the maximum number of iterations
max_iterations = 1000
actions = [0,1,2,3,4,5,6]

def get_feature_vector(state,action):
    fv=feature_matrix[my_model.predict(np.array(state))].tolist()[0]

    #print('FV: ',fv)
    #actions=[0,0,0,0,0,0,0]
    #actions[action]=1
    #return (fv+actions)
    return fv

#get rewards
def get_reward(state,action,objective):#objective is 0 for safety, 1 for speed, 2 for smoothness, 3 for total
    most_sim_reward=0
    most_sim=0
    for s in tup:
        if s[1]==action:
            state1 = flatten((np.array(state).reshape(-1, 1)).tolist())
            S = flatten(np.array(s[0]).reshape(-1,1).tolist())
            # compute the cosine similarity between the two vectors
            cosine_similarity = np.dot(state1, S) / (np.linalg.norm(state1) * np.linalg.norm(S))
            #print(cosine_similarity)
            if(cosine_similarity > most_sim):
                most_sim=cosine_similarity
                most_sim_reward=s[2][objective]
    #print('Most Sim for state' + str(state) +': '+ str(most_sim))
    return most_sim_reward

## argmax thing, calc new Rj to pass to weight updation function:

def policy_improvement(w,objective):
    Rj=[]
    for s in subsample:
        values=[0,0,0,0,0,0,0]
        for action in actions:
            #print(get_feature_vector(s,action))
            w_temp=list(w)#+[1,1,1,1,1,1,1]
            #print(w_temp)
            values[action]=np.dot(np.array(get_feature_vector(s[0],action)), np.array(w_temp))
        a=np.argmax(values)
        Rj.append(get_reward(np.array(s[0]),a,objective))
    #print(Rj)
    return Rj

def policy_evaluation(Ri):
    # Define the discount factor gamma
    gamma = 0.9
    #print("Ri in fn ",Ri)
    # Compute f'
    f_prime = np.roll(feature_matrix, -1, axis=0)
    f_prime[-1] = 0

    # Compute the matrix inside the inverse
    A = np.dot(feature_matrix.T, feature_matrix - gamma * f_prime) #3x3
    # Compute the weight
    #print('fm.T',feature_matrix.T.shape,'Ri',np.array(Ri).shape)
    w = np.dot(np.linalg.inv(A), np.dot(feature_matrix.T, Ri))
    return w

weights=[]
for obj in range(3):
    Ri=[]
    for node in subsample:
        Ri.append(node[2][obj])
        #print("Ri ",Ri)
    #Ri = np.array(Ri)
    #print("ri final",Ri)
    w=policy_evaluation(Ri)
    # Perform policy optimization
    while True:
        old_w = w.copy()
        w = policy_evaluation(policy_improvement(old_w,obj))
        if np.allclose(old_w, w, rtol=1e-6):
            break
    weights.append(w)
    print("WEIGHTS\nObjective ",obj,"  Weight ",w,'\n')

def decision_function(state):
    aj=0
    most_sim=0
    for s in tup:
        state1 = flatten((np.array(state).reshape(-1, 1)).tolist())
        S = flatten(np.array(s[0]).reshape(-1,1).tolist())
            # compute the cosine similarity between the two vectors
        cosine_similarity = np.dot(state1, S) / (np.linalg.norm(state1) * np.linalg.norm(S))
            #print(cosine_similarity)
        if(cosine_similarity > most_sim):
            most_sim=cosine_similarity
            aj=s[1]
    return aj
    
    #obj_wts=[0.7,0.2,0.1]
    #total_reward=0
    #action=0
    #r=0
    #values=[0,0,0,0,0,0,0]
    #for obj in range(3):
    #    values=[0,0,0,0,0,0,0]
    #    for action in actions:
    #        #print(get_feature_vector(state,action),'  weight: ',weights[obj])
    #        values[action]=get_reward(state,action,obj)#np.dot(get_feature_vector(state,action), weights[obj])
    #        #print('VALUE  ',values[action])
    #    aj=np.argmax(values)
    #    print('VALUES:  ',values,'  action: ',aj,'\n')
    #    return aj
    #    #for obj in range(3):
    #    #    reward=get_reward(state,aj,obj)
    #    #    total_reward = total_reward + reward*obj_wts[obj]

    #    #if(total_reward>r):
    #    #    action=aj
    #    #    r=total_reward
    #return aj

while(True):
    #throttle,brake,steer
    a=decision_function(get_state(world,vehicle))
    t=0
    b=0
    s=0
    if(a==0):
        t=1
        b=0
        s=0
    elif(a==1):
        t=1
        b=0
        s=-1
    elif(a==2):
        t=1
        b=0
        s=1
    elif(a==3):
        t=0
        b=1
        s=0
    elif(a==4):
        t=0
        b=1
        s=-1
    elif(a==5):
        t=0
        b=1
        s=1
    elif(a==6):
        t=0
        b=0
        s=0
    print('0')

    #DETECT WAYPOINTS

    # define a threshold angle for detecting turns
    threshold_angle = 30.0

    # iterate over the waypoints to find the nearest turn
    nearest_turn = None
    dist=0

    control = carla.VehicleControl()
    control.throttle = 0.5
    vehicle.apply_control(control)

    while(True):

        # get the current waypoint of the vehicle
        curr_waypoint = world.get_map().get_waypoint(vehicle.get_location())

        # get the next waypoint on the route
        next_waypoint = curr_waypoint.next(10)[0]

        # calculate the angle between the orientations of the two waypoints
        orientation1 = curr_waypoint.transform.rotation.yaw
        orientation2 = next_waypoint.transform.rotation.yaw
        angle_between_waypoints = abs(orientation1 - orientation2)

        velocity = vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        #print("Angle between waypoints:", angle_between_waypoints)
        if(angle_between_waypoints>300):
            print('3')
            location = vehicle.get_transform().location
            distance = location.distance(next_waypoint.transform.location)
            time.sleep(distance/speed+0.55)
            #control.throttle = 0
            control.steer = 72.5
            vehicle.apply_control(control)
            time.sleep(0.831)
            control.steer = 0
            vehicle.apply_control(control)
            time.sleep(1)
            #control.steer = -74
            #vehicle.apply_control(control)
            #time.sleep(0.85)
            #control.steer = 0
            #vehicle.apply_control(control)
            #time.sleep(4)
        else:
            print('0')

    #control = carla.VehicleControl(throttle= t, brake=b,steer=s)