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

EGOLOC = carla.Transform(carla.Location(x=300.002968, y=245.089096, z=1.001809), carla.Rotation(pitch=-0.016358, yaw=180.0, roll=0.000000))

TOWN = 'Town06_Opt'

TR_LOC = [
    carla.Transform(carla.Location(x=275.002968, y=243.089096, z=1.001809), carla.Rotation(pitch=-0.016358, yaw=180.0, roll=0.000000)),
    carla.Transform(carla.Location(x=275.002968, y=247.089096, z=1.001809), carla.Rotation(pitch=-0.016358, yaw=180.0, roll=0.000000))
]
S = [-2] * len(TR_LOC)

TR_VEL = []
for s in S:
    TR_VEL.append(carla.Vector3D(s, 0, 0))

assert len(TR_VEL) == len(TR_LOC), 'Different Vel and Pos shapes!'

EGO_VEL = carla.Vector3D(-3, 0, 0)

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
LANE_L = 7
LANE_R = 7
RUN_LENGTH = 200
DEFAULT_VEL = 1.8

# for props
ADD_PROPS = True
n_props = 200
x_L1 = list(np.linspace(213, 108, n_props))
x_R1 = list(np.linspace(94, 21, n_props))
x_L2 = list(np.linspace(-18, -68, n_props))
y_L1 = list(np.linspace(255, 259, n_props))
y_R1 = list(np.linspace(228, 232, n_props))
y_L2 = list(np.linspace(255, 259, n_props))

PROPS_X_LIST = [x_L1, x_R1, x_L2]
PROPS_Y_LIST = [y_L1, y_R1, y_L2]

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