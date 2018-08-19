#!/usr/bin/env python

import os
import time
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import pickle
import argparse

# yolo Libraries
import darknet as dn

model_num = 0 # 7
thresh = 0.5
hier_thresh = 0.0
nms = 0.45
verbose = True
max_objects = 5

start_time = time.time()
font = cv2.FONT_HERSHEY_DUPLEX

dnPath = os.environ['HOME'] + '/programs/darknet/'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--model_type", required=False, help="model number")
args = vars(ap.parse_args())

if args['model_type'] is not None:
  model_num = int(args['model_type'])

modelType = ''
if model_num == 0:
  modelType = 'yolo'
elif model_num == 1:
  modelType = 'yolov2'
elif model_num == 2:
  modelType = 'yolov2_544'
elif model_num == 3:
  modelType = 'yolov2_608'
elif model_num == 4:
  modelType = 'tiny_yolo'
elif model_num == 5:
  modelType = 'tiny_yolov2'
elif model_num == 6:
  modelType = 'alexnet'
elif model_num == 7:
  modelType = 'tiny-yolo-voc'

else:
  print 'Invalid model_num: ',model_num
  exit(1)

modelPath = dnPath + 'cfg/' + modelType + '.cfg'
weightsPath = dnPath + 'weights/' + modelType + '.weights'
metaPath = dnPath + 'cfg/' + 'temp_coco.data'

if model_num == 7:
  metaPath = dnPath + 'cfg/' + 'voc.data'
# Load model
dn.set_gpu(0)
net = dn.load_net(modelPath, weightsPath, 1)
meta = dn.load_meta(metaPath)

# Convert numpy array to image
def array_to_image(arr):
  arr = arr.transpose(2,0,1)
  c = arr.shape[0]
  h = arr.shape[1]
  w = arr.shape[2]
  arr = (arr/255.0).flatten()
  data = dn.c_array(dn.c_float, arr)
  im = dn.IMAGE(w,h,c,data)
  return im

def detectObjects(net, meta, image, thresh=thresh, hier_thresh=hier_thresh, nms=nms):
  boxes = dn.make_boxes(net)
  probs = dn.make_probs(net)
  num =   dn.num_boxes(net)
  dn.network_detect(net, image, thresh, hier_thresh, nms, boxes, probs)
  res = []
  for j in range(num):
    for i in range(meta.classes):
      if probs[j][i] > 0:
        res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
  res = sorted(res, key=lambda x: -x[1])
  dn.free_ptrs(dn.cast(probs, dn.POINTER(dn.c_void_p)), num)
  return res  

def imageCb(data):

  start_time = time.time()
  try:
    cv_image = CvBridge().imgmsg_to_cv2(data,"bgr8")
  except CvBridgeError as e:
    print e

  im = array_to_image(cv_image)
  dn.rgbgr_image(im)

  objects = detectObjects(net,meta,im)
  obj_ctr = 0
  for obj in objects:
    obj_ctr += 1
    if obj_ctr > max_objects:
      break
    x = int(obj[2][0])
    y = int(obj[2][1])
    w = int(obj[2][2])
    h = int(obj[2][3])

    pt1 = (x-w/2,y-h/2)
    pt2 = (x+w/2,y+h/2)

    cv2.rectangle(cv_image, pt1, pt2, (0,255,0), 3)
    text = "{} ({}%)".format(obj[0],int(obj[1]*100))
    cv2.putText(cv_image, text, (pt1[0],pt1[1]-20), font, 0.6, (0.255,255),1)

  #print r
  print "RunTime:", round(time.time() - start_time,3)

  cv2.imshow('Image Window', cv_image)
  cv2.waitKey(1)
  rospy.sleep(0.01)
  
def read_cam():
  cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
  
  print cap.isOpened()
  if cap.isOpened():
    while not rospy.is_shutdown():
      ret_val, frame = cap.read();
      imageCb(frame)
   
if __name__ == "__main__":
  rospy.init_node('yolo_node')
  image_sub = rospy.Subscriber("camera/rgb/image_color", Image, imageCb, queue_size=1, buff_size=2**24)

  if verbose:
    print ("Loading yolo model took {} seconds.".format(time.time() - start_time))

  try:
    rospy.spin()
    read_cam()
  except KeyboardInterrupt:
    print ("Shutting down")
  cv2.destroyAllWindows()

