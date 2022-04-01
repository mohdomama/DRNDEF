"""
    Create and display range images from 3D pointclouds represented as (x,y,z) in the Cartesian
    Coordinate System.

    USAGE:
        python -m util.range_image [--save_images] [--show_range_image] [--show_count_image]
"""

import numpy as np
import math
import os
import matplotlib.pyplot as plt
import random
import argparse

# Folder Creation
FOLDER = "data_collection/test-data/pointclouds/"
range_img_folder = "data_collection/test-data/range_images/"
count_img_folder = "data_collection/test-data/count_images/"


def range_projection(current_vertex, fov_up=15.0, fov_down=-15.0, proj_H=16, proj_W=1800, max_range=100):
    """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds, XYZI
      Returns:
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    current_vertex = current_vertex[(depth > 0) & (
        depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    intensity = current_vertex[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                             dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array(
        [scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices
    proj_intensity[proj_y, proj_x] = intensity

    return proj_range, proj_vertex, proj_intensity, proj_idx


def pcd_to_range_image(points, width=200):
    # Width defines the resolution of 1 channel
    height = 16  # Range image height, number of cannels in lidar
    FOV = 15 * np.pi / 180  # Should be in radian
    theta_up, theta_down = +FOV, -FOV
    range_image = np.zeros((height, width))
    count_image = np.zeros((height, width))
    for point in points:
        r = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        phi = math.atan2(point[1], point[0])
        theta = math.asin(point[2]/r)
        # print('Theta: ', theta)
        u = math.floor((0.5 * (1+(phi/math.pi))) * width)  # column
        v = math.floor(
            ((theta_up - theta)/(theta_up - theta_down)) * (height-1))  # row
        count_image[v, u] += 1

        # The farther it is, the larger the value. Might need to change it later
        if range_image[v, u] != 0 and range_image[v, u] < r:  # only plot closest point
            continue
        else:
            range_image[v, u] = r

        # Might add column image as input along with range image

    # Needed since there is a Y=-Y transformation in carla
    range_image = np.fliplr(range_image)
    count_image = np.fliplr(count_image)
    range_image = range_image
    count_image = count_image
    return range_image, count_image


def pcd_file_to_range_image(points_name, show_range_img=False, show_count_img=False):
    "Convert pointcloud to range image."

    points = np.vstack(np.load(points_name))
    width = 200  # Can be an arbitrary value. It defines resolution of 1 channel
    height = 16  # Range image height, number of cannels in lidar
    FOV = 15 * np.pi / 180  # Should be in radian
    theta_up, theta_down = +FOV, -FOV
    range_image = np.zeros((height, width))
    count_image = np.zeros((height, width))
    for point in points:
        r = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        phi = math.atan2(point[1], point[0])
        theta = math.asin(point[2]/r)
        # print('Theta: ', theta)
        u = math.floor((0.5 * (1+(phi/math.pi))) * width)  # column
        v = math.floor(
            ((theta_up - theta)/(theta_up - theta_down)) * height)  # row
        count_image[v, u] += 1

        # The farther it is, the larger the value. Might need to change it later
        if range_image[v, u] != 0 and range_image[v, u] < r:  # only plot closest point
            continue
        else:
            range_image[v, u] = r

        # Might add column image as input along with range image

    # Needed since there is a Y=-Y transformation in carla
    range_image = np.fliplr(range_image)
    count_image = np.fliplr(count_image)

    # range_image = range_image / np.amax(range_image)
    # count_image = count_image / np.amax(count_image)

    if show_range_img:
        plt.imshow(range_image, cmap='gray')
        plt.title(points_name)
        plt.show()
    if show_count_img:
        plt.imshow(count_image, cmap='gray')
        plt.title(points_name)
        plt.show()
    return range_image, count_image


def main(args):
    if not (os.path.isdir(range_img_folder) and os.path.isdir(count_img_folder)):
        os.makedirs(range_img_folder)
        os.makedirs(count_img_folder)

    if args.save_images:
        for file in os.listdir(FOLDER):
            range_image, count_image = pcd_file_to_range_image(FOLDER + file)
            range_img_name = range_img_folder + file[:-4] + ".png"
            count_img_name = count_img_folder + file[:-4] + ".png"
            plt.imsave(range_img_name, range_image)
            plt.imsave(count_img_name, count_image)
    elif args.show_range_image:
        file = random.choice(os.listdir(FOLDER))
        pcd_file_to_range_image(FOLDER + file, show_range_img=True)
    elif args.show_count_image:
        file = random.choice(os.listdir(FOLDER))
        pcd_file_to_range_image(FOLDER + file, show_count_img=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--save_images',
        default=False,
        action='store_true',
        help='Save range and count images of pointclouds'
    )
    argparser.add_argument(
        '--show_range_image',
        default=False,
        action='store_true',
        help='Shows a random range image'
    )
    argparser.add_argument(
        '--show_count_image',
        default=False,
        action='store_true',
        help='Shows a random count image'
    )
    args = argparser.parse_args()

    main(args)


'''
Range Projection Ref: https://github.com/PRBonn/range-mcl/blob/main/src/utils.py
'''
