# This is for Stanley
from collections import namedtuple
import carla
import sys
import numpy as np

sys.path.append('../../../')

SCENE_NAME = "scene4"

TRAFFIC = True
SCENE_NUMBER = 4
#####
FILTER_VEHICLES = True
USE_RL = True
PLOT = False
#####

EGOLOC = carla.Transform(carla.Location(x=400.002968, y=248.089096, z=1.001809), carla.Rotation(pitch=-0.016358, yaw=180.0, roll=0.000000))


TOWN = 'Town06_Opt'

TR_LOC = [
    carla.Transform(carla.Location(x=365.002968, y=245.089096, z=1.001809),
                    carla.Rotation(pitch=-0.016358, yaw=180.0, roll=0.000000)),
    # carla.Transform(carla.Location(x=376.002968, y=241.089096, z=1.001809),
    #                 carla.Rotation(pitch=-0.016358, yaw=180.0, roll=0.000000)),
    carla.Transform(carla.Location(x=360.002968, y=245.089096, z=1.001809),
                    carla.Rotation(pitch=-0.016358, yaw=180.0, roll=0.000000))
]
S = [-1, -1]

TR_VEL = []
for s in S:
    TR_VEL.append(carla.Vector3D(s, 0, 0))

assert len(TR_VEL) == len(TR_LOC), 'Different Vel and Pos shapes!'

EGO_VEL = carla.Vector3D(-1.5, 0, 0)

LAYERS = [
    carla.MapLayer.Walls,
    carla.MapLayer.StreetLights,
    carla.MapLayer.Props,
    carla.MapLayer.Foliage,
    # carla.MapLayer.Buildings,
    carla.MapLayer.ParkedVehicles,
]

LIDAR_RANGES_LIST = [55, 50, 45]

######
LIDAR_RANGE = LIDAR_RANGES_LIST[2]
######

FPS = 10
LANE_L = 6
LANE_R = 6
RUN_LENGTH = 150
DEFAULT_VEL = 1.8

# for props
ADD_PROPS = False
PROP_TYPES = ['static.prop.busstop', ]

x_L1 = [x for x in range(200, 450, 5)]
n_props = len(x_L1)
'''
Ablation:
1) gap 5: np.linspace(450, 205, 50)
2) gap 10: np.linspace(450, 210, 25)
'''

y_L1 = [260] * n_props

PROPS_X_LIST = [x_L1]
PROPS_Y_LIST = [y_L1]

rpy = [0.153900, -0.008640, 0.014069]


def generate_props(client, world, props):
    import random
    import time
    for prop_side_x, prop_side_y in zip(PROPS_X_LIST, PROPS_Y_LIST):
        prop_x, prop_y = random.sample(
            prop_side_x, n_props), random.sample(prop_side_y, n_props)
        prop_spawn_points = []

        for x, y in zip(prop_x, prop_y):
            prop_spawn_points.append(carla.Transform(carla.Location(
                x=x, y=y, z=0.2), carla.Rotation(pitch=rpy[1], yaw=rpy[2], roll=rpy[0])))

        batch = []
        SpawnActor = carla.command.SpawnActor

        prop_bps = random.choices(props, k=n_props)

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

    10,  # Vehicles

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
