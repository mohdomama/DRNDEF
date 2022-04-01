# This is for Stanley
from collections import namedtuple
import carla
import sys
import numpy as np

sys.path.append('../../../')

SCENE_NAME = "scene1"

TRAFFIC = False
SCENE_NUMBER = 2
#####
FILTER_VEHICLES = True
USE_RL = True
PLOT = False
#####
# x = 227, 580,   np.linspace(227, 580, 20)
# y = 240, 250, 2 np.linspace(240, 250, 5)
# theta = 0, 10, 20, ... np.linspace(0, 350, 36)
CAR_POSES = [
    {
    'x': np.linspace(227, 580, 6),
    'y': np.linspace(240, 250, 3),
    'th': np.linspace(0, 350, 5),
    'lane_bound': np.array([240, 250])
    },
    {
    'x': np.linspace(156, 400, 5), # less features is easily learnable with small data
    'y': np.linspace(-20, -10, 3),
    'th': np.linspace(0, 350, 5),
    'lane_bound': np.array([-22, -12])
    },
    {
    'x': np.linspace(156, 400, 6),
    'y': np.linspace(-20, -10, 3),
    'th': np.linspace(0, 350, 5),
    'lane_bound': np.array([-22, -12])
    }
]
EGOLOC = carla.Transform(carla.Location(x=590.002968, y=250.589096, z=1.001809),
                         carla.Rotation(pitch=-0.016358, yaw=720.0, roll=0.000000))
TOWN = 'Town06_Opt'

# in this road, y=-23 is left lane, y=-10 is right lane, stretch is 154-411 in x
EGOLOC_ALT = carla.Transform(carla.Location(x=154.787628, y=-23.306889, z=1.001679),         
                            carla.Rotation(pitch=-0.000061, yaw=-1.432587, roll=0.000101))

EGO_END = carla.Transform(carla.Location(x=411.380859, y=-10.591185, z=0.001681),
                          carla.Rotation(pitch=-0.000246, yaw=3.182308, roll=0.000014))

TR_LOC = []
S = [2.5] * len(TR_LOC)

TR_VEL = []
for s in S:
    TR_VEL.append(carla.Vector3D(s, 0, 0))

assert len(TR_VEL) == len(TR_LOC), 'Different Vel and Pos shapes!'

EGO_VEL = carla.Vector3D(2.5, 0, 0)

LAYERS = [
    carla.MapLayer.Walls,
    carla.MapLayer.StreetLights,
    carla.MapLayer.Props,
    carla.MapLayer.Foliage,
    carla.MapLayer.Buildings,
    carla.MapLayer.ParkedVehicles
]

LAYERS_TO_TOGGLE = [carla.MapLayer.Buildings]

LIDAR_RANGES_LIST = [55, 50, 45]

######
LIDAR_RANGE = LIDAR_RANGES_LIST[2]
######

FPS = 10
LANE_L = 12
LANE_R = 1
RUN_LENGTH = 100
DEFAULT_VEL = 1.8


# Not being used
def cons_vel(traffic):
    for itr, vehicle in enumerate(traffic):
        # vehicle.set_target_velocity(TR_VEL[itr])
        vehicle.enable_constant_velocity(TR_VEL[itr])


def traffic_runner(traffic):
    for itr, vehicle in enumerate(traffic):
        vehicle.set_target_velocity(TR_VEL[itr])
        # vehicle.enable_constant_velocity(TR_VEL[itr])
