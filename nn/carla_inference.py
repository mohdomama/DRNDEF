"""
    Inference PyTorch models on CARLA Simulator.

    USAGE:
        python -W ignore -m nn.carla_inference
"""

from jax._src.api import grad
from numpy.linalg import pinv
from util.transforms import se3_to_components, build_se3_transform
from scene.scene_builder import *
import numpy as np
import torch
from util.range_image import pcd_to_range_image, range_projection
from matplotlib import pyplot as plt
from .dataset import preprocess, DriftDataset
from util.misc import bcolors
from math import ceil, degrees
import argparse
import cv2
from math import ceil
from planner.frenet import FrenetRunner
from planner import stanley
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_current_speed(EGO):
    vel_ego = EGO.get_velocity()
    speed = np.linalg.norm(np.array([vel_ego.x, vel_ego.y]))
    return speed


def get_state(ego):
    speed = get_current_speed(ego)
    loamX, loamY, _, loamYaw = roscom.loam_latest
    return speed, loamX, loamY, loamYaw


def get_gt_state(ego, tf_matrix):
    ego_tf = ego.get_transform().get_matrix()
    ego_tf_odom = np.linalg.pinv(tf_matrix) @ ego_tf
    x, y, z, roll, pitch, yaw = se3_to_components(ego_tf_odom)
    speed = get_current_speed(ego)
    return speed, x, -y, -yaw


INFERENCE_TRANSFORMS = [
    # Surrounded by buildings
    # carla.Transform(carla.Location(x=450.819763, y=195.897217, z=0.194828),
    # carla.Rotation(pitch=-0.005649, yaw=179.550919, roll=-0.000000)),
    # carla.Transform(carla.Location(x=550.463867, y=195.962158, z=0.194814),
    #                 carla.Rotation(pitch=-0.000519, yaw=179.552216, roll=0.000000)),
    # carla.Transform(carla.Location(x=538.718689, y=195.831909, z=0.194831),
    #                 carla.Rotation(pitch=-0.006714, yaw=-179.426773, roll=0.000000)),


    # carla.Transform(carla.Location(x=318.002968, y=144.3, z=1.001809),
    #  carla.Rotation(pitch=-0.016358, yaw=180.0, roll=0.000000))

    # # buildings on left, trees on right
    # carla.Transform(carla.Location(x=289.048157, y=140.0, z=1.042423),
    # carla.Rotation(pitch=-0.003722, yaw=-179.395493, roll=-0.000000)),
    # carla.Transform(carla.Location(x=275.332855, y=136.192200, z=0.042424),
    #                 carla.Rotation(pitch=-0.004064, yaw=-179.394592, roll=0.000000)),

    # # bleh
    # carla.Transform(carla.Location(x=235.323257, y=164.546173, z=0.195008),
    #                 carla.Rotation(pitch=-0.005894, yaw=22.521198, roll=-0.003113)),

    # Building on right, small bushes on left
    # carla.Transform(carla.Location(x=573.455322, y=245.033035, z=0.042428),
    # carla.Rotation(pitch=-0.005464, yaw=179.792618, roll=0.000000)),
    # carla.Transform(carla.Location(x=530.276733, y=246.753952, z=0.042441),
    # carla.Rotation(pitch=-0.009460, yaw=179.403595, roll=-0.000000))

    # stress-test frenet
    # Transform(Location(x=255.301559, y=-18.940542, z=0.042411), Rotation(pitch=0.000348, yaw=179.439804, roll=0.000000))

    carla.Transform(carla.Location(x=319.178772, y=251.330322, z=0.042414),
                    carla.Rotation(pitch=-0.000553, yaw=179.984894, roll=0.000000))

]


def get_features_from_edgepcd(edgepcd):
    assert edgepcd.shape[1] == 4, 'Wrong edgepcd shape!'

    if np.mean(edgepcd[:, 1]) < 0:
        print('[Edge Prediction] Right Better')
        feats = edgepcd[edgepcd[:, 1] < 0]
    else:
        print('[Edge Prediction] Left Better')
        feats = edgepcd[edgepcd[:, 1] > 0]

    feats = feats[0:50,0:2]
    Fx = feats[:, 0]
    Fy = feats[:, 1]

    return Fx, Fy

def get_features_from_grad(ri, gradimg, ri_traffic):
    img_d = cv2.dilate(ri_traffic, np.ones([3,7]), iterations=1)
    mask = 1 - img_d
    gradimg = gradimg * mask
    left = gradimg[:, :900]
    # left = left[left > 0.2]
    right = gradimg[:, 900:]
    # right = right[right > 0.2]
    if np.sum(left) > np.sum(right):
        print('[Grad Prediction] Left Better')
        gradimg[:, 900:] = 0
    else:
        print('[Grad Prediction] Right Better')
        gradimg[:, :900] = 0


    gradimg[ri<=0] = 0
    # gradimg = gradimg*mask

    featsidx = (-gradimg).argsort(axis=None)
    
    # TOP 50
    ids = np.unravel_index(featsidx[:50], gradimg.shape)
    gradimg[ids] = 1

    # All others
    ids = np.unravel_index(featsidx[50:], gradimg.shape)
    gradimg[ids] = 0

    featimg = gradimg * ri

    # plt.imshow(featimg, interpolation='nearest', aspect='auto')
    # plt.show()
    # --- Feat XY calculation
    U, V = np.where(featimg > 0) # U and V are set of image coords
    Vals = featimg[featimg>0]
    Fx, Fy = [], []
    for u, v, val in zip(U, V, Vals):
        dis = (1-val) * 45
        ang = np.deg2rad(180 - (0.2*v))
        feat_x = np.cos(ang) * dis
        feat_y = np.sin(ang) * dis
        Fx.append(feat_x)
        Fy.append(feat_y)

    
    return Fx, Fy


def get_frenet_state(ego, odom_tf):
    odom_rot = np.copy(odom_tf)
    odom_rot[:-1, 3] = 0  # Making the translation part as zero
    vel = ego.get_velocity()
    vel_tf = build_se3_transform([vel.x, vel.y, vel.z, 0, 0, 0])
    vel_tf_odom = np.linalg.pinv(odom_rot) @ vel_tf
    velx, vely, velz, _, _, _ = se3_to_components(vel_tf_odom)

    acc = ego.get_acceleration()
    acc_tf = build_se3_transform([acc.x, acc.y, acc.z, 0, 0, 0])
    acc_tf_odom = np.linalg.pinv(odom_rot) @ acc_tf
    accx, accy, accz, _, _, _ = se3_to_components(acc_tf_odom)

    speed, x, y, yaw = get_state(ego)

    # y in frenet init frame (center of the road)
    y_finit = y + (ROAD_WIDTH/2 - LANE_L)

    return x, y_finit, speed, -vely, -accy


def get_gt(ego, traffic, world, spectator, dummy, rf):
    xrange = 5
    yrange = 5
    x_sample = list(range(-xrange, xrange))
    y_sample = list(range(-yrange, yrange))
    target_tf = np.eye(4)
    ego_tf = np.array(ego.get_transform().get_matrix())
    heat_map = np.ones((len(x_sample), len(y_sample)))
    for itrx, x_target in enumerate(x_sample):
        for itry, y_target in enumerate(y_sample):
            target_tf[0][3] = x_target  # In lidar frame
            target_tf[1][3] = y_target
            ego_target_tf = ego_tf @ target_tf  # In map frame
            x, y, z, roll, pitch, yaw = se3_to_components(ego_target_tf)
            ego_target_loc = carla.Location(x, y, z)
            ego_target_rot = carla.Rotation(pitch, yaw * 180 / np.pi, roll)
            ego.set_transform(carla.Transform(ego_target_loc, ego_target_rot))
            # heat_map[x+xrange][y+yrange] = numlocs
            for i in range(5):
                world.tick()
                spectator.set_transform(dummy.get_transform())

            tf = len(roscom.edge_points) + len(roscom.surface_points)
            heat_map[itrx][itry] = (tf - rf) / rf
            print(itrx, itry)

    heat_map = np.flipud(heat_map)
    return heat_map


def get_traffic_pcd(pcd):
    pcd_traffic = pcd[pcd[:, 5] == 10]
    return pcd_traffic

def filter_ground_plane(pcd):
    pcd = pcd[pcd[:, 2] > - 1.55]
    return pcd

def filter_pcd_classes(pcd):
    for semantic in FILTER_PCD_CLASSES:
        pcd = pcd[pcd[:, 5] != semantic]
    return pcd


def main(args):
    ego, traffic, world, spectator, dummy, lidar_sen = setup_carla()
    model = torch.load(args.model_path).to(device)
    print(f"Ego ID: {ego.id}")
    time.sleep(5)

    # to filter pcd, set this to true
    roscom.filter_pcd = False

    if args.prototype:
        model.eval()
        world.tick()
        spectator.set_transform(dummy.get_transform())
        # ego_tf = ego.get_transform().get_matrix()
        zinit = ego.get_transform().location.z
        target_layers = [model.enc2.residual_0.conv2]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        debug = world.debug
        odom_tf = np.array(ego.get_transform().get_matrix())

        
        pcd = np.copy(roscom.pcd[:, :4])
        ri_tf = np.array(build_se3_transform([0, 0, 0, 0, 0, 0]))
        pcd = (np.linalg.pinv(ri_tf) @ pcd.T).T


        ri, _, _, _ = range_projection(pcd)


        ## Begin
        data = DriftDataset.preproces(
            ri, ri, ri, drift=None, has_label=False)  # dosen't have label
        ri = data['anchor'].unsqueeze(0).to(
            device)  # Adding batch dimension

        targets = [ClassifierOutputTarget(0)]
        grayscale_cam0 = cam(input_tensor=ri, targets=targets)

        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        plt.imshow(ri[0][0].detach().cpu().numpy(),
                    interpolation='nearest', aspect='auto')

        fig.add_subplot(2, 1, 2)
        plt.imshow(grayscale_cam0[0],
                    interpolation='nearest', aspect='auto')

        preds = model(ri).detach().cpu().numpy()
        print('Preds: ', np.array(preds))

        plt.show()
        breakpoint()

    elif args.drive:
        world.tick()
        spectator.set_transform(dummy.get_transform())
        # cv2.namedWindow('Standard Norm', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('Per Image Norm', cv2.WINDOW_NORMAL)
        odom_tf = np.array(ego.get_transform().get_matrix())
        model.eval()

        while True:
            # heat_map = surround_inference(ego, model)
            # heat_map2 = np.copy(heat_map)

            # print('Min Max: ', heat_map.min(), heat_map.max())
            # print('Diff:: ', np.abs(heat_map.min()-heat_map.max()))
            # print('Variane: ', np.var(heat_map))
            # print('Dis from 0: ', np.mean(np.abs(heat_map)))

            # up = 0.45
            # down = -0.25
            # heat_map2 = np.clip(heat_map2, down, up)

            # heat_map = cv2.normalize(
            #     heat_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

            # up_array = np.ones((1, heat_map2.shape[1])) * up
            # down_array = np.ones((1, heat_map2.shape[1])) * down
            # heat_map2 = np.vstack([up_array, heat_map2, down_array])

            # heat_map2 = cv2.normalize(
            #     heat_map2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

            # cv2.imshow('Per Image Norm', heat_map)
            # cv2.imshow('Standard Norm', heat_map2)
            # cv2.waitKey(1)
            if args.drive_infer:
                target_layers = [model.conv5]
                _, gt_x, gt_y, _ = get_gt_state(ego, odom_tf)
                roscom.publish_gt(gt_x, gt_y)
                # print('Drift: ', roscom.drift)
                traffic_pcd = get_traffic_pcd(roscom.pcd)
                pcd = filter_pcd_classes(roscom.pcd)

                ri, _, _, _ = range_projection(pcd[:, :4])
                ri_traffic, _, _, _ = range_projection(traffic_pcd[:, :4])
                ri_traffic[ri_traffic>0] = 3
                ri_traffic[ri_traffic<0] = 0
                ri = ri + ri_traffic
                data = DriftDataset.preproces(
                    ri, ri, ri, drift=None, has_label=False)  # dosen't have label
                ri = data['anchor'].unsqueeze(0).to(
                    device)  # Adding batch dimension

                cam = GradCAM(
                    model=model, target_layers=target_layers, use_cuda=True)

                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                targets = [ClassifierOutputTarget(0)]

                grayscale_cam0 = cam(input_tensor=ri)

                # targets = [ClassifierOutputTarget(1)]
                # grayscale_cam1 = cam(input_tensor=ri, targets=targets)

                preds = model(ri).detach().cpu().numpy()

                print('Preds: ', np.array(preds))

                # side = np.argmax(preds[0])
                # if side == 0:
                #     camimg = grayscale_cam0[0]
                #     camimg[:, 900:] = 0

                # else:
                #     camimg = grayscale_cam1[0]
                #     camimg[:, :900] = 0

                fig = plt.figure()
                fig.add_subplot(2, 1, 1)
                plt.imshow(ri[0][0].detach().cpu().numpy(),
                           interpolation='nearest', aspect='auto')

                fig.add_subplot(2, 1, 2)
                plt.imshow(grayscale_cam0[0],
                           interpolation='nearest', aspect='auto')

                # fig.add_subplot(2, 1, 3)
                # plt.imshow(grayscale_cam1[0],
                #            interpolation='nearest', aspect='auto')
                plt.show()
                breakpoint()
                # plt.imshow(ri[0][0].detach().cpu().numpy(), interpolation='nearest', aspect='auto'); plt.show()
                # cv2.namedWindow('camimg', cv2.WINDOW_NORMAL)
                # camimg = cv2.resize(camimg, (4400, 900))
                # cv2.imshow('camimg', camimg)
                # cv2.waitKey(20)
            world.tick()
            spectator.set_transform(dummy.get_transform())

    elif args.frenet:
        world.tick()
        spectator.set_transform(dummy.get_transform())
        odom_tf = np.array(ego.get_transform().get_matrix())
        wx = np.linspace(0, 150, 20)
        wy = np.full(wx.shape[0], 0)
        ob = []
        frenet = FrenetRunner(wx, wy, ob)
        start_loc = np.array(ego.get_location().x, dtype=np.float32)
        drifts = []
        target_layers = [model.conv1]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        while True:
            curr_loc = np.array(ego.get_location().x, dtype=np.float32)
            dist = np.linalg.norm(start_loc-curr_loc)
            if dist >= RUN_LENGTH:
                print(f'Distance Traversed: {dist:.3f}')
                break

            # Inference

            # print('Drift: ', roscom.drift)
            pcd = roscom.pcd[:, :4]
            _, _, _, yaw = get_state(ego)

            print('Yaw: ', -yaw)
            # TODO: inv not required if we make yaw -ve
            ri_tf = np.array(build_se3_transform([0, 0, 0, 0, 0, yaw]))

            pcd = (np.linalg.pinv(ri_tf) @ pcd.T).T
            ri, _, _, _ = range_projection(pcd[:, :4])
            data = DriftDataset.preproces(
                ri, drift=None, has_label=False)  # dosen't have label
            ri = data['ri'].unsqueeze(0).to(
                device)  # Adding batch dimension

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            targets = [ClassifierOutputTarget(0)]
            grayscale_cam0 = cam(input_tensor=ri, targets=targets)

            targets = [ClassifierOutputTarget(1)]
            grayscale_cam1 = cam(input_tensor=ri, targets=targets)

            preds = model(ri).detach().cpu().numpy()
            print('Preds: ', np.array(preds))

            side = np.argmax(preds[0])
            if side == 0:
                camimg = grayscale_cam0[0]
                camimg[:, 900:] = 0

            else:
                camimg = grayscale_cam1[0]
                camimg[:, :900] = 0

            cv2.namedWindow('camimg', cv2.WINDOW_NORMAL)
            img = cv2.resize(camimg, (5000, 900))
            cv2.imshow('camimg', img)
            cv2.waitKey(20)
            ###########

            fx, fy, fspeed, fvely, faccy = get_frenet_state(ego, odom_tf)
            fx = max(fx, 0)  # Frenet fails otherwise

            paths = frenet.get_paths(fx, fspeed, fy, fvely, faccy, ob)
            # fig = plt.figure()
            # for path in paths:
            #     # px = np.array(path.x).reshape(1, -1)
            #     # py = np.array(path.y).reshape(1, -1) - (ROAD_WIDTH/2 - LANE_L)
            #     # print(px,py)
            #     plt.plot(-np.array(path.y), np.array(path.x))
            # plt.xlabel('y (in m)')
            # plt.ylabel('x (in m)')
            # plt.title('Frenet-Generated Paths')
            # # plt.show()
            # # fig.savefig('frenet_paths.png', dpi=400)
            # pts = np.vstack([roscom.edge_points, roscom.surface_points])
            # # pts = filter_ground_plane(pts)
            # heatmap = surround_inference(ego, model)
            # heatmap = cv2.normalize(
            #     heatmap, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            # # cv2.imwrite('heatmap.png', heatmap)
            # paths_np = None
            # for itr, path in enumerate(paths):
            #     px = np.array(path.x).reshape(1, -1)
            #     # Path in odom frame
            #     py = np.array(path.y).reshape(1, -1) - (ROAD_WIDTH/2 - LANE_L)

            #     pz = np.zeros(len(path.x)).reshape(1, -1)
            #     ones = np.ones(len(path.x)).reshape(1, -1)
            #     all_points_odom = np.vstack([px, py, pz, ones])
            #     base_tf = np.array(ego.get_transform().get_matrix())
            #     # Convering points in base_link
            #     all_points_bl = np.linalg.pinv(
            #         base_tf) @ odom_tf @ all_points_odom

            #     px = all_points_bl[0, :].reshape(-1, 1)
            #     # NN works in left hand coordinate frame
            #     py = -all_points_bl[1, :].reshape(-1, 1)

            #     rf = np.array([len(pts)] * len(path.x)).reshape(-1, 1)
            #     pitr = np.array([itr] * len(path.x)).reshape(-1, 1)
            #     pcomb = np.hstack([px, py, rf, pitr])
            #     if paths_np is None:
            #         paths_np = pcomb
            #     else:
            #         paths_np = np.vstack([paths_np, pcomb])
            #     # Visualisation of LHCS
            #     # plt.plot(-resy, resx)

            # paths_np[:, :2] = paths_np[:, :2] / 10
            # paths_np[:, 2] = paths_np[:, 2] / 50
            # sc_inp = torch.tensor(paths_np[:, :-1]).float().to(device)
            # # print('Sc Inp: ', sc_inp.shape)
            # range_image, count_image = range_projection(pts)
            # range_image, count_image = torch.from_numpy(
            #     np.copy(range_image)), torch.from_numpy(np.copy(count_image))
            # image = torch.stack([range_image, count_image])
            # image = image.float().to(device) / 45
            # image = image.expand([len(paths_np), image.shape[0],
            #                       image.shape[1], image.shape[2]])

            # preds = model(image, sc_inp)
            # # print('Preds: ', preds.shape)

            max_score = -np.inf
            for itr, path in enumerate(paths):
                if side == 0:
                    score = np.mean(np.array(path.y))
                else:
                    score = -np.mean(np.array(path.y))
                if score > max_score:
                    # print(' Best Itr: ', itr)
                    best_path = path
                    max_score = score
            # plt.show()
            # print('Paths NP: ', paths_np.shape)

            path = best_path
            # Converting to odom frame
            path.x = np.array(path.x)
            path.y = np.array(path.y) - (ROAD_WIDTH/2 - LANE_L)
            desired_speed = 4

            # if args.auto_drive:
            #     print('Here!')
            #     path.x = np.array([RUN_LENGTH, RUN_LENGTH])
            #     path.y = np.array([0, 0])
            # throttle_previous, int_val = previous
            previous = (0, 0)
            last_target_index = 0
            # plt.plot(-np.array(path.y), np.array(path.x))
            # plt.show()
            for i in range(5):
                speed, x, y, yaw = get_state(ego)
                _, gt_x, gt_y, _ = get_gt_state(ego, odom_tf)
                roscom.publish_gt(gt_x, gt_y)
                throttle, previous = stanley.speed_control(
                    speed, desired_speed, 1/FPS, previous)
                delta, last_target_index = stanley.stanley_control(
                    path.x[1:], -path.y[1:], path.yaw[1:], x, y, -yaw, speed, last_target_index)

                if args.auto_drive:
                    ego.apply_control(carla.VehicleControl(
                        throttle=throttle, steer=0, brake=0))
                else:
                    ego.apply_control(carla.VehicleControl(
                        throttle=throttle, steer=-delta*0.3, brake=0))

                drifts.append(roscom.drift)
                print(f'Drift: {drifts[-1]:.3f}')
                # print('heya: ', last_target_index, len(path.x))
                world.tick()
                spectator.set_transform(dummy.get_transform())
        print(f'Average Drift: {np.mean(drifts):.3f}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--prototype",
        action="store_true",
        help="To test model inference and prototype quickly"
    )
    argparser.add_argument(
        "--drive",
        action="store_true",
        help="To generate real-time inference while driving"
    )
    argparser.add_argument(
        "--drive_infer",
        action="store_true",
        help="Run inference while driving"
    )
    argparser.add_argument(
        "--auto_drive",
        action="store_true",
        help="To show drift in real-time vanilla Stanley-controlled autonomous driving"
    )
    argparser.add_argument(
        "--frenet",
        action="store_true",
        help="To show drift in real-time Frenet-Generated paths in autonomous driving"
    )
    argparser.add_argument(
        "--model_path",
        default='trained_models/test/1.pth',
        help="Enter the path of the trained model"
    )
    args = argparser.parse_args()
    main(args)
