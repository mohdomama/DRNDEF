import roslaunch
import rospy
import atexit


class ALOAM:
    def __init__(self) -> None:
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        self.launch = None
        # CARLA_WS='/ssd_scratch/omama/catkin_ws/'
        CARLA_WS = '/home/padfoot7/catkin_ws/'
        self.cli_args = [CARLA_WS +
                         "src/A-LOAM/launch/aloam_carla.launch", "rviz:=True", "output:=log"]
        self.roslaunch_args = self.cli_args[1:]
        self.roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(
            self.cli_args)[0], self.roslaunch_args)]
        atexit.register(self.kill)  # This will run with python exists

    def start(self):
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        self.launch = roslaunch.parent.ROSLaunchParent(
            self.uuid, self.roslaunch_file)
        self.launch.start()
        rospy.loginfo("Started ALOAM for CARLA")

    def kill(self):
        if self.launch is not None:
            self.launch.shutdown()


'''
cli_args = ['/home/mosaic/catkin_ws/src/robot/launch/id.launch','vel:=2.19']
roslaunch_args = cli_args[1:]
roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
'''
