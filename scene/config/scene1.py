# This is for Stanley
from collections import namedtuple
import carla
import sys

sys.path.append('../../../')

SCENE_NAME = "scene2"

TRAFFIC = False
SCENE_NUMBER = 2
#####
FILTER_VEHICLES = True
USE_RL = True
PLOT = False
#####

EGOLOC = carla.Transform(carla.Location(x=227.002968, y=250.589096, z=1.001809),
                         carla.Rotation(pitch=-0.016358, yaw=0.0, roll=0.000000))
TOWN = 'Town06_Opt'

TR_LOC = []
S = [2.5] * len(TR_LOC)

TR_VEL = []
for s in S:
    TR_VEL.append(carla.Vector3D(s, 0, 0))

assert len(TR_VEL) == len(TR_LOC), 'Different Vel and Pos shapes!'

EGO_VEL = carla.Vector3D(2.5, 0, 0)

UNLOADED_LAYERS = [
    carla.MapLayer.Walls,
    carla.MapLayer.StreetLights,
    carla.MapLayer.Props,
    carla.MapLayer.Foliage,
    # carla.MapLayer.Buildings,
    # carla.MapLayer.ParkedVehicles,
]

LIDAR_RANGES_LIST = [55, 50, 45]

######
LIDAR_RANGE = LIDAR_RANGES_LIST[2]
######

FPS = 10
LANE_L = 12
LANE_R = 1
RUN_LENGTH = 200
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
