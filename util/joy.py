from sensor_msgs.msg import Joy
import rospy
import os
import sys
from functools import partial
sys.path.append("../")


def callback_joystick(data):

    string_to_send = str(data.buttons[0]) + " " + str(data.buttons[1]) + " " + str(
        data.buttons[2]) + " " + str(data.buttons[3]) + " " + str(data.axes[0]) + " " + str(data.axes[1])
    print("Sending string: ", string_to_send, "\n")


if __name__ == "__main__":

    rospy.init_node('listener_joystick', anonymous=True)
    sub = rospy.Subscriber("/joy", Joy, callback_joystick)
    rospy.spin()
