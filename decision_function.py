import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import carla

#Connecting to CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()

def get_feature_vector(state,action):
    fv=feature_matrix[my_model.predict(np.array(state))]
    actions=np.zeros(9)
    actions[action]=1
    fv.append(actions)
    return fv

#get rewards
def get_reward(state,action,objective):#objective is 0 for safety, 1 for speed, 2 for smoothness, 3 for total
    most_sim_reward=0
    most_sim=0
    for s in samples:
        if s[1]==action:
            if(cosine_similarity(state, s[0])>most_sim):
                most_sim=cosine_similarity(state, s[0])
                most_sim_reward=s[2][objective]

    return most_sim_reward

## argmax thing, calc new Rj to pass to weight updation function:

def policy_improvement(w,objective):
    Rj=[]
    for s in states:
        values=[]
        for action in actions:
            values[action]=np.dot(get_feature_vector(s,action), w)
        a=np.argmax(values)
        Rj.append(get_reward(s,a,objective))
    return Rj

def policy_evaluation(Ri):
    # Define the discount factor gamma
    gamma = 0.9

    # Compute f'
    f_prime = np.roll(feature_matrix, -1, axis=0)
    f_prime[-1] = 0

    # Compute the matrix inside the inverse
    A = np.dot(feature_matrix.T, feature_matrix - gamma * f_prime) #3x3
    # Compute the weights
    print('fm.T',feature_matrix.T.shape,'Ri',Ri.shape)
    w = np.dot(np.linalg.inv(A), np.dot(feature_matrix.T, Ri))
    return w

for obj in range(3):
    Ri=[]
    for node in subsample:
        Ri.append(node[2][obj])

    w=policy_evaluation(Ri)
    # Perform policy optimization
    while True:
        old_w = w.copy()
        w = policy_evaluation(policy_improvement(old_w,obj))
        if np.allclose(old_w, w, rtol=1e-6):
            break


#NAVIGATION DECISION FUNCTION:
#let weights=[x,y,z]
def decision_function(state):
    obj_wts=[0.7,0.2,0.1]
    total_reward=0
    action=0
    r=0
    for obj in range(3):
        for action in actions:
            values[action]=np.dot(get_feature_vector(s,action), w[obj])
        aj=np.argmax(values)
        for obj in range(3):
            reward=get_reward(state,aj,obj)
            total_reward = total_reward + reward*obj_wts[obj]

        if(total_reward>r):
            action=aj
            r=total_reward
    return aj


    # a=[]
    # for obj in range(objectives.len):
    #     for action in actions:
    #         values[action]=np.dot(get_feature_vector(s,action), w[obj])
    #     aj=np.argmax(values)
    #     a.append(aj)
    #     sum=sum+objweights[obj]*a
    #
    # outputdecision=math.ceil(sum)


#REWARDS:
def reward_function(vehicle):
    # Reward for staying within the speed limit
    velocity = vehicle.get_velocity()
    speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
    waypoint = world.get_map().get_waypoint(vehicle.get_transform().location)
    speed_limit=waypoint.get_speed_limit()
    speed_reward = 0.05*speed if speed <= speed_limit else -0.2

    # Reward for smooth driving
    acceleration = vehicle.get_acceleration().x
    smoothness_reward = -0.01 * abs(acceleration)

    safety_reward = 0.8 if collisioncount==0 else -0.1*collisioncount
    # Compute the total reward
    reward = speed_reward + smoothness_reward + safety_reward

    return [safety_reward,speed_reward,smoothness_reward,reward]
