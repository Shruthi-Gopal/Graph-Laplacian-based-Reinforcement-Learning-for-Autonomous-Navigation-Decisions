import carla

# Connect to the CARLA simulator
client = carla.Client('localhost', 2000)
world = client.get_world()

# Get a list of all actors in the simulation
actor_list = world.get_actors()

# Iterate over the list and remove any actors that are vehicles
for actor in actor_list:
    if 'vehicle' in actor.type_id:
        actor.destroy()
