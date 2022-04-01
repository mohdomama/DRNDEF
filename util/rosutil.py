# ROS Communication Utilities
from scene.config.infer.is12_t_t6 import FILTER_VEHICLES
import rospy
import sensor_msgs
import std_msgs
from sensor_msgs.msg import PointCloud2, PointField, Joy
from std_msgs.msg import Header
from nav_msgs.msg import Path, Odometry
from roslib import message
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from util.transforms import get_yaw_from_odom_orientation
from scene.config_importer import FILTER_PCD_CLASSES

import numpy as np
import ros_numpy

rospy.init_node('Carla')


class RosCom:
    def __init__(self) -> None:
        self.path_publisher = rospy.Publisher(
            'gt_path', Path, queue_size=1, latch=True)
        rospy.Subscriber('/gt_path', Path, self.gt_callback)
        rospy.Subscriber('/aft_mapped_to_init_high_frec',
                         Odometry, self.loam_odom_callback)
        rospy.Subscriber('/laser_odom_to_init', Odometry,
                         self.loam_odom_nomap_callback)
        rospy.Subscriber('/aft_mapped_to_init', Odometry,
                         self.loam_map_callback)
        rospy.Subscriber('/laser_cloud_less_sharp',
                         PointCloud2, self.corners_callback)
        rospy.Subscriber('/laser_cloud_less_flat',
                         PointCloud2, self.surfaces_callback)
        self.points_publisher = rospy.Publisher(
            '/velodyne_points', PointCloud2, queue_size=1)
        rospy.Subscriber("/joy", Joy, self.joystick_callback)

        self.loam_latest = [0, 0, 0, 0]
        self.loam_map_latest = [0, 0, 0, 0]
        self.gt_latest = None
        self.gt_count = 0
        self.seq = None

        self.msg = Path()
        self.msg.header.frame_id = 'camera_init'
        self.msg.header.stamp = rospy.Time.now()
        self.centeroid = None
        self.drift = 0
        self.joy_control = {'steer': 0, 'throttle': 0}
        self.edge_points = None
        self.surface_points = None
        self.edge_heat_map = None
        self.pcd = None
        self.filter_pcd = False

    def joystick_callback(self, msg):
        # mag:  A, B, X, Y, L/R(1/-1), U/D(1/-1)
        # string_to_send = str(msg.buttons[0]) + " " + str(msg.buttons[1]) + " " + str(
        #     msg.buttons[2]) + " " + str(msg.buttons[3]) + " " + str(msg.axes[0]) + " " + str(msg.axes[1])
        throttle = float(msg.buttons[0])*0.5 + float(msg.buttons[1])*0.3
        steer = -float(msg.axes[0]) * 0.5
        self.joy_control['steer'] = steer
        self.joy_control['throttle'] = throttle

    def corners_callback(self, msg):
        seq = msg.header.seq
        pc = ros_numpy.numpify(msg)
        points = np.zeros((pc.shape[0], 3))
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        # print('Number Of Edge Features: ', len(points))
        # print(seq)
        self.edge_points = points
        self.centeroid = points.mean(axis=0)

    def surfaces_callback(self, msg):
        seq = msg.header.seq
        pc = ros_numpy.numpify(msg)
        points = np.zeros((pc.shape[0], 3))
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        # print('Number Of Edge Features: ', len(points))
        # print(seq)
        self.surface_points = points

    def gt_callback(self, msg):
        self.gt_count += 1
        pose = msg.poses[-1]
        position = pose.pose.position
        self.gt_latest = np.array([position.x, position.y, position.z])

    def loam_odom_callback(self, msg):
        self.odom_seq = msg.header.seq
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = get_yaw_from_odom_orientation(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        self.loam_latest = np.array([position.x, position.y, position.z, yaw])
        if self.loam_latest is not None and self.gt_latest is not None:
            drift = np.linalg.norm(self.loam_latest[:2] - self.gt_latest[:2])
        else:
            drift = 0
        self.drift = drift

    def loam_odom_nomap_callback(self, msg):
        self.odom_nomap_seq = msg.header.seq
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = get_yaw_from_odom_orientation(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        self.loam_nomap_latest = np.array([position.x, position.y, position.z, yaw])

    def loam_map_callback(self, msg):
        self.map_seq = msg.header.seq
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = get_yaw_from_odom_orientation(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        self.loam_map_latest = np.array([position.x, position.y, position.z, yaw])
      

    def clear(self):
        self.sequences = []
        self.positions = []

    def pcd_2_point_cloud(self, points, parent_frame, frametime):
        assert points.shape[1] == 5, 'PCD should be in XYZIR format!'
        ros_dtype = sensor_msgs.msg.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        data = points.astype(dtype).tobytes()
        fields = [
            sensor_msgs.msg.PointField(
                name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate(['x', 'y', 'z', 'intensity', 'ring'])
        ]
        header = std_msgs.msg.Header(frame_id=parent_frame, stamp=frametime)

        return sensor_msgs.msg.PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 5),
            row_step=(itemsize * 5 * points.shape[0]),
            data=data
        )

    def publish_points(self, pcd):
        assert pcd.shape[1] == 6, 'PCD should be in XYZIRS format'
        self.pcd = pcd  # Saving PCD with semantics
        # Filter out vehicles from Lidar scan
        if FILTER_VEHICLES:
            for classnum in FILTER_PCD_CLASSES:
                pcd = pcd[pcd[:, 5] != classnum]
        ros_pcd = self.pcd_2_point_cloud(pcd[:, :-1], 'map', rospy.Time.now())
        self.points_publisher.publish(ros_pcd)

    def publish_gt(self, x, y, z=0):
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y  # Not flipping
        pose.pose.position.z = z
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        self.msg.poses.append(pose)

        self.path_publisher.publish(self.msg)


def main():
    roscom = RosCom()
    rospy.spin()


if __name__ == '__main__':
    main()
