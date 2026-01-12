#!/usr/bin/env python3
import sys
import time

import rospy
from std_msgs.msg import String

import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import sys

import cv2 as cv
import numpy as np


import rospkg
rospack = rospkg.RosPack()
# get the file path for rospy_tutorials
rospack.get_path('depth_subs')


import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("/tof_a/output",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/tof_a/tof_camera/depth",Image,self.callback)

    args = get_args()

    self.cap_device = args.device
    self.cap_width = args.width
    self.cap_height = args.height

    self.use_static_image_mode = args.use_static_image_mode
    self.min_detection_confidence = args.min_detection_confidence
    self.min_tracking_confidence = args.min_tracking_confidence


    # カメラ準備 ###############################################################
    self.cap = cv.VideoCapture(self.cap_device)
    self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
    self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

    

   

  def callback(self,data):

      # キー処理(ESC：終了) #################################################
    key = 40
    try:
      image = self.bridge.imgmsg_to_cv2(data, "32FC1")
    except CvBridgeError as e:
      print(e)

    image = cv.flip(image, 1)  # ミラー表示
    # print(image)
    # 検出実施 #############################################################
    # image = np.invert(image)
    # image = cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    gray = image/1400
    print(gray)
    # image = cv.merge([((gray/gray.max())*255).astype(np.uint8) , ((gray/gray.max())*255).astype(np.uint8), ((gray/gray.max())*255).astype(np.uint8)])
    image = cv.merge([((gray ) * 255).astype(np.uint8), ((gray ) * 255).astype(np.uint8),
                      ((gray ) * 255).astype(np.uint8)])
    # image = cv.cvtColor(image,  cv.COLOR_BGR2RGB)  # cv.COLOR_BGR2RGB)

    # image = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)#cv.COLOR_BGR2RGB)
    # image = cv.merge([int(gray/140), int(gray/140), int(gray/140)])
    debug_image = copy.deepcopy(image)
    image.flags.writeable = False
    results = self.hands.process(image)
    image.flags.writeable = True

    (rows,cols,channels) = debug_image.shape


    # cv2.imshow("Image window", debug_image)
    # cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main():
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.3)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.2)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
