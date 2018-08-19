#!/usr/bin/python2.7

import os
import sys
import argparse
import time
import numpy as np

import cv2

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import darknet as dn

dnPath = os.environ['HOME'] + '/programs/darknet'
sys.path.insert(0,dnPath)
sys.path.insert(0,dnPath + '/keras-yolov3')

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False, help="model name")
args = vars(ap.parse_args())

model = 'yolov3'
thresh = 0.5
hier_thresh = 0.5
nms = 0.45
font = cv2.FONT_HERSHEY_DUPLEX

dn_active = True
keras_active = True

if args['model'] is not None:
  model = args['model']

data_file = dnPath + '/' + 'data' + '/' 
cfg_file =  dnPath + '/' + 'cfg' + '/' + model + '.cfg'
weights_file =  dnPath + '/' + 'weights' + '/' + model + '.weights'

if 'voc' in model:
  data_file += 'voc.data'
elif '9000' in model:
  data_file += 'combine9k.data'
else:
  data_file += 'coco.data'

######## Load darknet model ######

net = ''
meta = ''
if dn_active:
  dn.set_gpu(0)
  net = dn.load_net(cfg_file, weights_file, 0)
  meta = dn.load_meta(data_file)

##################################

######### Load Keras model #######

yolo = ''
if keras_active:
  yolo = YOLO(model)

##################################

# Convert numpy array to dn image
# def array_to_image(arr):
#   arr = arr.transpose(2,0,1)
#   c = arr.shape[0]
#   h = arr.shape[1]
#   w = arr.shape[2]
#   arr = (arr/255.0).flatten()
#   data = dn.c_array(dn.c_float, arr)
#   im = dn.IMAGE(w,h,c,data)
#   return im

def imageCb(data):
  global font, net, meta, nms, thres, hier_thresh, yolo
  yolo_im = ''
  dn_im = ''
  try:
    if keras_active: yolo_im = CvBridge().imgmsg_to_cv2(data,"bgr8")
    if dn_active: dn_im = CvBridge().imgmsg_to_cv2(data,"bgr8")
  except CvBridgeError as e:
    rospy.logerr('[YOLO] %s',e)
    return

  if keras_active:
    start_time = time.time()
    names, confs, locs = yolo.detect(yolo_im)

    print 'Keras detection time:', time.time() - start_time

    for name, conf, box in zip(names, confs, locs):

      label = '{} {:.2f}'.format(name, conf)

      l,t,r,b = box

      pt1 = (l,t)
      pt2 = (r,b)

      cv2.rectangle(yolo_im, pt1, pt2, (0,255,0), 3)
      cv2.putText(yolo_im, label, (pt1[0],pt1[1]-20), font, 0.6, (0.255,255),1)

      print pt1, pt2

    cv2.imshow('keras-yolo',yolo_im)

  if dn_active:
    start_time = time.time()
    names, confs, locs = dn.detect(dn_im, net, meta, thresh, hier_thresh, nms)
    print 'Yolo detection time:', time.time() - start_time

    for name, conf, box in zip(names, confs, locs):
      
      label = "{} ({}%)".format(name,int(conf*100))
      
      x, y, w, h = box
      x = int(box[0])
      y = int(box[1])
      w = int(box[2])
      h = int(box[3])

      pt1 = (x-w/2,y-h/2)
      pt2 = (x+w/2,y+h/2)

      cv2.rectangle(dn_im, pt1, pt2, (0,255,0), 3)
      cv2.putText(dn_im, label, (pt1[0],pt1[1]-20), font, 0.6, (0.255,255),1)

    cv2.imshow("yolo",dn_im)
    
  cv2.waitKey(1)

if __name__ == "__main__":
  rospy.init_node('yolo')
  rospy.loginfo("[YOLO] Started!")
  image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, imageCb, queue_size=1, buff_size=2**24)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

  cv2.destroyAllWindows()