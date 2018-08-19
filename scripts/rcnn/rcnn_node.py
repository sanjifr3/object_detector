#!/usr/bin/env python

import time
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import pickle
import argparse

# Fast RCNN Libraries
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import caffe
import os
from utils.timer import Timer
import sys
import scipy.io as sio

model_num = 1 # 7
use_cpu = False
max_objects = 5
verbose = True
gpu_id = 0
thresh = 0.8
nms_thresh = 0.3

# thresh = 0.5
# hier_thresh = 0.0
# nms = 0.45

start_time = time.time()
font = cv2.FONT_HERSHEY_DUPLEX

rcnnPath = '/home/sanjif/programs/py-faster-rcnn/'

#caffePath = '/home/sanjif/programs/py-faster-rcnn/caffe-fast-rcnn/python'
#rcnnPath = '/home/sanjif/programs/py-faster-rcnn/lib'

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

cfg.TEST.HAS_RPN = True # Use RPN for proposals

prototxt = rcnnPath + 'models/' + 'pascal_voc/' + NETS[NETS.keys()[model_num]][0] + '/faster_rcnn_alt_opt/faster_rcnn_test.pt'
caffemodel = rcnnPath + 'data/faster_rcnn_models/' + NETS[NETS.keys()[model_num]][1]

if use_cpu:
  caffe.set_mode_cpu()
else:
  caffe.set_mode_gpu()
  caffe.set_device(gpu_id)
  cfg.GPU_ID = gpu_id

net = caffe.Net(prototxt, caffemodel, caffe.TEST)

print '\n\nLoaded network {:s}'.format(caffemodel)

def detectObjects(im):
  # Detect all object classes and regress object bounds
  timer = Timer()
  timer.tic()
  all_scores, all_boxes = im_detect(net, im)
  timer.toc()
  print ('Detection took {:.3f}s for '
         '{:d} object proposals').format(timer.total_time, all_boxes.shape[0])

  res = []
  dets = np.zeros((len(all_scores),5))
  dets_i = 0

  for obj_i in range(len(all_scores)):
    j = -1
    best_score = 0
    for score_i in range(1,len(all_scores[obj_i])):
      score = all_scores[obj_i][score_i]
      if score > thresh and score > best_score:
        best_score = score
        j = score_i
    
    if j != -1:
      name = CLASSES[j]
      score = best_score
      loc = all_boxes[obj_i][4*j:4*(j+1)]
      res.append((name, score, (loc[0],loc[1],loc[2],loc[3])))
      dets[dets_i] = np.array((loc[0],loc[1],loc[2],loc[3],score))
      dets_i += 1

  dets = dets[0:dets_i]
  dets = dets.astype(np.float32)
  keep = nms(dets, nms_thresh)

  nms_res = []
  for idx in keep:
    nms_res.append(res[idx])

  return nms_res

def imageCb(data):

  start_time = time.time()
  try:
    cv_image = CvBridge().imgmsg_to_cv2(data,"bgr8")
  except CvBridgeError as e:
    print e

  objects = detectObjects(cv_image)

  for obj in objects:
    x1 = int(obj[2][0])
    y1 = int(obj[2][1])
    x2 = int(obj[2][2])
    y2 = int(obj[2][3])

    pt1 = (x1,y1)
    pt2 = (x2,y2)

    cv2.rectangle(cv_image, pt1, pt2, (0,255,0), 3)
    text = "{} ({}%)".format(obj[0],int(obj[1]*100))
    cv2.putText(cv_image, text, (pt1[0],pt1[1]-20), font, 0.6, (0.255,255),1)

  print "RunTime:", round(time.time() - start_time,3)
  
  cv2.imshow('detection_window', cv_image)
  cv2.waitKey(1)
  rospy.sleep(0.01)
   
if __name__ == "__main__":
  rospy.init_node('rcnn_node')
  image_sub = rospy.Subscriber("camera/rgb/image_color", Image, imageCb, queue_size=1, buff_size=2**24)

  if verbose:
    print ("Loading rcnn model took {} seconds.".format(time.time() - start_time))

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print ("Shutting down")
  cv2.destroyAllWindows()

