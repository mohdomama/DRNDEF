# This is for Stanley
from collections import namedtuple
import carla
import sys
import numpy as np

sys.path.append('../../../')

SCENE_NAME = "scene4"

TRAFFIC = False
SCENE_NUMBER = 4
#####
FILTER_VEHICLES = True
USE_RL = True
PLOT = False
#####

EGOLOC = carla.Transform(carla.Location(x=153.0, y=-107.725906, z=1.581011),
                         carla.Rotation(pitch=0.001953, yaw=90.062695, roll=0.000008))
TOWN = 'Town05'

TR_LOC = []
S = [-2] * len(TR_LOC)

TR_VEL = []
for s in S:
    TR_VEL.append(carla.Vector3D(s, 0, 0))

assert len(TR_VEL) == len(TR_LOC), 'Different Vel and Pos shapes!'

EGO_VEL = carla.Vector3D(0, 3, 0)

LAYERS = [
    # carla.MapLayer.Walls,
    # carla.MapLayer.StreetLights,
    # carla.MapLayer.Props,
    # carla.MapLayer.Foliage,
    # # carla.MapLayer.Buildings,
    # carla.MapLayer.ParkedVehicles,
]

LIDAR_RANGES_LIST = [55, 50, 45]

######
LIDAR_RANGE = LIDAR_RANGES_LIST[2]
######

FPS = 10
LANE_L = 3
LANE_R = 3
RUN_LENGTH = 200
DEFAULT_VEL = 1.8

# for props
ADD_PROPS = True
PROP_TYPES = ['static.prop.streetsign',]
n_props1 = 20
n_props2 = 100
x_R1 = list(np.linspace(145.5, 146, n_props1))
x_R2 = list(np.linspace(143, 144, n_props1))
x_R3 = list(np.linspace(138.5, 139.8, n_props1))
x_R4 = list(np.linspace(145.5, 146, n_props2))
x_R5 = list(np.linspace(146, 146.5, n_props2))
y_R1 = list(np.linspace(-90, -100, n_props1))
y_R2 = list(np.linspace(-105, -109, n_props1))
y_R3 = list(np.linspace(-109, -117, n_props1))
y_R4 = list(np.linspace(-80, -16, n_props2))
y_R5 = list(np.linspace(13, 88, n_props2))

rpy = [-0.008640, 0.014069, 0.153900]

PROPS_X_LIST_SHORT = [x_R1, x_R2, x_R3]
PROPS_Y_LIST_SHORT = [y_R1, y_R2, y_R3]

PROPS_X_LIST_LONG = [x_R4, x_R5]
PROPS_Y_LIST_LONG = [y_R4, y_R5]

# Not being used

def generate_props(client, world, props):
    import random 
    import time
    for prop_side_x, prop_side_y in zip(PROPS_X_LIST_SHORT, PROPS_Y_LIST_SHORT):
        prop_x, prop_y = random.sample(prop_side_x, n_props1), random.sample(prop_side_y, n_props1)
        prop_spawn_points = []

        for x,y in zip(prop_x, prop_y):
            prop_spawn_points.append(carla.Transform(carla.Location(x=x, y=y, z=0.2), carla.Rotation(pitch=rpy[1], yaw=rpy[2], roll=rpy[0])))


        batch = []
        SpawnActor = carla.command.SpawnActor

        prop_bps = random.choices(props, k=n_props1)


        for prop_bp, prop_spawn_point in zip(prop_bps, prop_spawn_points):
            batch.append(SpawnActor(prop_bp, prop_spawn_point))

        client.apply_batch_sync(batch, True)
        world.tick()

    for prop_side_x, prop_side_y in zip(PROPS_X_LIST_LONG, PROPS_Y_LIST_LONG):
        prop_x, prop_y = random.sample(prop_side_x, n_props2), random.sample(prop_side_y, n_props2)
        prop_spawn_points = []

        for x,y in zip(prop_x, prop_y):
            prop_spawn_points.append(carla.Transform(carla.Location(x=x, y=y, z=0.2), carla.Rotation(pitch=rpy[1], yaw=rpy[2], roll=rpy[0])))


        batch = []
        SpawnActor = carla.command.SpawnActor

        prop_bps = random.choices(props, k=n_props2)


        for prop_bp, prop_spawn_point in zip(prop_bps, prop_spawn_points):
            batch.append(SpawnActor(prop_bp, prop_spawn_point))

        client.apply_batch_sync(batch, True)
        world.tick()

    time.sleep(1.0)

# Not being used
def cons_vel(traffic):
    for itr, vehicle in enumerate(traffic):
        # vehicle.set_target_velocity(TR_VEL[itr])
        vehicle.enable_constant_velocity(TR_VEL[itr])


def traffic_runner(traffic):
    for itr, vehicle in enumerate(traffic):
        vehicle.set_target_velocity(TR_VEL[itr])
        # vehicle.enable_constant_velocity(TR_VEL[itr])

FILTER_PCD_CLASSES = [
    # Reference: https://carla.readthedocs.io/en/0.9.11/ref_sensors/
   
    # # 10, # Vehicles
   
    # # The following set collectively represents ground
    # 22,  # Terrain
    # 14,   # Ground
    # 6,  # Road Line
    # 7,    # Road
    # 8,  # Side Walk

    # 0,  # Unlabelled
    # 3,  # Other

    # 11,   # Wall
    # # 17,   # Ground Rail
    # # # 1,  # Building
    # # 9, # Vegetation

]