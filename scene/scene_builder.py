from .config_importer import *
import time
import numpy as np
from util.rosutil import RosCom
import sys
sys.path.append('../')

roscom = RosCom()

def get_static_props_bp(world):
    static_props = [
            bp for bp in world.get_blueprint_library().filter('static')]
    props = []
    for prop in static_props:
        for prop_type in PROP_TYPES:
            if prop.id.startswith(prop_type):
                props.append(prop)
    return props


def add_ring_channel(pcd):
    '''
    Converts XYZI pcd to XYZIR
    Args:
        pcd: XYZI array
    Returns:
        pcd: XYZIR
    '''
    assert pcd.shape[1] == 4, "Expected PCD in XYZI format!"

    # There are lidar dependent params
    fov_up = 15 / 180 * np.pi
    fov_down = -15 / 180 * np.pi
    fov = abs(fov_up) + abs(fov_down)
    proj_H = 16  # Number of channels in Lidar

    depth = np.linalg.norm(pcd[:, :3], 2, axis=1)
    pcd = pcd[(depth > 0)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0)]

    scan_x = pcd[:, 0]
    scan_y = pcd[:, 1]
    scan_z = pcd[:, 2]
    intensity = pcd[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    proj_y = (pitch + abs(fov_down)) / fov   # [0 to 1]
    proj_y *= proj_H

    # Round and clamp for use as index
    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)
    proj_y = proj_y.reshape(-1, 1)

    # Concatenate ring column to PCD
    pcd = np.concatenate((pcd, proj_y), axis=1)

    # Remove nan, inf (if there)
    pcd = pcd[np.logical_not(np.any(np.isnan(pcd), axis=1))]
    pcd = pcd[np.logical_not(np.any(np.isinf(pcd), axis=1))]

    return pcd


def process_point_cloud(point_cloud_carla):
    pcd = np.copy(np.frombuffer(point_cloud_carla.raw_data,
                  dtype=np.dtype("f4, f4, f4, f4, u4, u4")))
    pcd = np.array(pcd.tolist())

    # The 4th column is considered as intensity in ros, hence making it one
    pcd[:, 3] = 1

    # Flipping Y
    pcd[:, 1] = -pcd[:, 1]

    pcd_xyzi = pcd[:, :4]
    pcd_sem = pcd[:, 5].reshape(-1, 1)  # Semantic Information
    pcd_xyzir = add_ring_channel(pcd_xyzi)
    # Append Semantics Column
    pcd_xyzirs = np.concatenate((pcd_xyzir, pcd_sem), axis=1)
    # Save PCD if required
    # noise = np.random.normal(0, 0.012, (len(pcd_xyzirs), 3))
    # pcd_xyzirs[:, :3] += noise


    roscom.publish_points(pcd_xyzirs)


def dummy_function(image):
    pass


def setup_carla():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.load_world(TOWN)
    for i in range(10):
        world.tick()
    for layer in LAYERS:
        world.unload_map_layer(layer)
        time.sleep(1)

    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0/FPS  # FPS = 1/0.05 = 20
    world.apply_settings(settings)
    world.tick()
    time.sleep(1)
    egobp = world.get_blueprint_library().find('vehicle.mini.cooperst')
    egobp.set_attribute('role_name', 'ego')

    trafficbp = world.get_blueprint_library().find('vehicle.volkswagen.t2')
    # trafficbp = world.get_blueprint_library().find('vehicle.audi.a2')
    trafficbp.set_attribute('role_name', 'traffic')

    ego = world.spawn_actor(egobp, EGOLOC)
    traffic = []
    if TRAFFIC:
        for loc in TR_LOC:
            tr = world.spawn_actor(trafficbp, loc)
            traffic.append(tr)

    # Setting Vehicle Physics Off
    for tr in traffic:
        carla.command.SetSimulatePhysics(tr, enabled=False)
    carla.command.SetSimulatePhysics(ego, enabled=False)

    dummy_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    # dummy_transform = carla.Transform(carla.Location(
    #     x=-1, z=31), carla.Rotation(pitch=0.0))
    dummy_transform = carla.Transform(carla.Location(
        x=-5, z=10, y = 0), carla.Rotation(pitch=20.0, yaw=0, roll=0))
    dummy = world.spawn_actor(dummy_bp, dummy_transform, attach_to=ego,
                              attachment_type=carla.AttachmentType.SpringArm)
    dummy.listen(lambda image: dummy_function(image))

    spectator = world.get_spectator()
    spectator.set_transform(dummy.get_transform())

    def setup_ticks():
        for i in range(20):
            world.tick()
            spectator.set_transform(dummy.get_transform())
            ego.apply_control(carla.VehicleControl(
                throttle=0, steer=0, brake=1))
            if TRAFFIC:
                for itr, vehicle in enumerate(traffic):
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=0, steer=0, brake=1))
        # Clearing Brake Control | This is Important
        ego.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=0))
        if TRAFFIC:
            for itr, vehicle in enumerate(traffic):
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0, steer=0, brake=0))

    setup_ticks()  # Ensures that vehicle lands to the ground

    # VLP 16
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('channels', str(16))
    # Set the fps of simulator same as this
    lidar_bp.set_attribute('rotation_frequency', str(FPS))
    lidar_bp.set_attribute('range', str(LIDAR_RANGE))
    lidar_bp.set_attribute('lower_fov', str(-15))
    lidar_bp.set_attribute('upper_fov', str(15))
    lidar_bp.set_attribute('points_per_second', str(300000))
    # noise_stddev
    # lidar_bp.set_attribute('dropoff_general_rate',str(0.0))
    lidar_location = carla.Location(0, 0, 1.75)
    lidar_rotation = carla.Rotation(0, 0, 0)
    lidar_transform = carla.Transform(lidar_location, lidar_rotation)
    lidar_sen = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego)
    lidar_sen.listen(lambda point_cloud: process_point_cloud(point_cloud))

    if ADD_PROPS:
        props = get_static_props_bp(world)
        generate_props(client=client, world=world, props=props)



    setup_ticks()  # Ensures that loam has some data to begin with

    return ego, traffic, world, spectator, dummy, lidar_sen


