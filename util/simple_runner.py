from numpy.linalg.linalg import norm
from scene.scene_builder import *
from util.transforms import se3_to_components, build_se3_transform
from util.ray_trace import generate_rays_features, get_mesh, generate_rays, intersect_vis
from util.misc import timeit
from matplotlib import pyplot as plt


def get_current_speed(EGO):
    vel_ego = EGO.get_velocity()
    speed = np.linalg.norm(np.array([vel_ego.x, vel_ego.y]))
    return speed


def get_gt_state(ego, tf_matrix):
    ego_tf = ego.get_transform().get_matrix()
    ego_tf_odom = np.linalg.pinv(tf_matrix) @ ego_tf
    x, y, z, roll, pitch, yaw = se3_to_components(ego_tf_odom)
    speed = get_current_speed(ego)
    return speed, x, y, yaw


def get_state(ego):
    speed = get_current_speed(ego)
    loamX, loamY, _, loamYaw = roscom.loam_latest
    return speed, loamX, -loamY, loamYaw


def main():
    ego, traffic, world, spectator, dummy, lidar_sen = setup_carla()
    debug = world.debug
    print('Ego ID: ', ego.id)
    time.sleep(3)

    world.tick()

    odom_tf = ego.get_transform().get_matrix()
    while True:
        if TRAFFIC:
            traffic_runner(traffic)
        print('Drift: ', roscom.drift)
        speed, x, y, yaw = get_state(ego)
        _, gtX, gtY, gtYaw = get_gt_state(ego, odom_tf)
        roscom.publish_gt(gtX, -gtY)
        world.tick()
        spectator.set_transform(dummy.get_transform())


if __name__ == '__main__':
    main()
