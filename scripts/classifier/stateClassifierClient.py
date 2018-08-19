#!/usr/bin/env python

import cv2
import rospy
import os
import pandas as pd
import numpy as np
from numpy import random

from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from object_detector.msg import cvRect, classificationInfo
from object_detector.srv import *

types = ['tv','lamp_light','head']

def getClasses(file):
  classes = []
  with open(file,'r') as f:
    for line in f:
      classes.append(line.strip('\n').split(',')[0])
  return classes

def getClsId(cls,all_classes):
  if cls in all_classes:
    return all_classes.index(cls)
  return -1

def parseTrainingFiles(files_path):
  files = []
  with open(files_path + '_train.txt','r') as f:
    for line in f:
      line = line.strip('\n')#.replace('images','labels').split('.')[0] + '.txt'
      files.append(line)

  with open(files_path + '_test.txt','r') as f:
    for line in f:
      line = line.strip('\n')#.replace('images','labels').split('.')[0] + '.txt'
      files.append(line)
  return files

def getImagePaths(class_name, coco_classes, imagenet_classes, script_path):
  files = []
  if class_name in coco_classes:
    files = parseTrainingFiles(script_path + 'coco/coco')
  elif class_name in imagenet_classes:
    files = parseTrainingFiles(script_path + 'imagenet/imagenet')
  return files

def getBBs(file, id):
  bb = []
  file = file.replace('images','labels').split('.')[0] + '.txt'
  with open(file,'r') as f:
    for line in f:
      line = line.strip('\n').split(' ')
      cls_id = int(line[0])
      if cls_id == id:
        bb.append((float(line[1]),float(line[2]),float(line[3]),float(line[4])))
  return bb

def call(im,typs,rects):
  rospy.wait_for_service('ObjectStateClassification')
  ros_im = ''
  try:
    if 'head' in typs:
      ros_im = CvBridge().cv2_to_imgmsg(im, "mono16")
    else:
      ros_im = CvBridge().cv2_to_imgmsg(im, "bgr8")
  except CvBridgeError as e:
    print e

  resp = ''
  try:
    classify = rospy.ServiceProxy('ObjectStateClassification',object_state_classifier)
    resp = classify(ros_im,typs,rects)
  except rospy.ServiceException, e:
    print "Service call failed: %s" % e
  return resp

if __name__ == "__main__":
  rospy.init_node("ObjectStateClassifier_client")


  current_im = cv2.imread('/home/sanjif/Database/Kinect_samples/Kinect_2.tiff',-1)
  old_im = cv2.imread('/home/sanjif/blueberry/src/social_robot/validation/images-merged/Ben_NULL_N_1002.tiff',-1)

  cv2.imshow('current',current_im)
  cv2.imshow('old',old_im)
  cv2.waitKey(0)


  print current_im.max(), current_im.min(), current_im.std(), current_im.mean()
  print old_im.max(), old_im.min(), old_im.std(), old_im.mean()

  exit(1)

  im_type = 'light'

  target_folder = '/home/sanjif/Database/'
  target_folder += 'objects/'
  target_folder += im_type + '/'



  #target_folder = "/home/sanjif/Database/Kinect_samples/"
  #target_folder = '/home/sanjif/Database/objects/head/'
  #target_folder = '/home/sanjif/Database/objects/light/'

  im_files = []

  for file in os.listdir(target_folder):
    im_files.append(file.split(".")[0])

  im_files = sorted(list(set(im_files)))

  for im_file in im_files:
    if im_file[0] in ['0','1','2','3','4','5','6','7','8','9']:
      continue
    print ("Analyzing", target_folder + im_file + ".jpg")
    #im = cv2.imread(target_folder + im_file + ".tiff",-1)
    im = None
    if im_type != 'head':
      im = cv2.imread(target_folder + im_file + '.jpg')
    else:
      im = cv2.imread(target_folder + im_file + '.tiff',-1)

    #df = pd.read_csv(target_folder + im_file + ".csv", sep=',')
    #df[['x','y','w','h']].astype('int32')

    if im is None: 
      continue
    
    req_boxes = []
    req_types = []

    # # print df
    # cv2.imshow("im",im)
    # cv2.waitKey(0)

    # for i, row in df.iterrows():
    #   #cv2.rectangle(im, (row['x'],row['y']), (row['x'] + row['w'], row['y'] + row['h']),2**16-1,3)

    #   box = cvRect()
    #   box.x = row['x']
    #   box.y = row['y']
    #   box.w = row['w']
    #   box.h = row['h']

    #   req_boxes.append(box)
    #   req_types.append('head')

    req_types.append(im_type)
    box = cvRect()
    box.x = 0
    box.y = 0
    if len(im.shape) == 3:
      box.h, box.w, _ = im.shape
    else:
      box.h, bow.w = im.shape

    req_boxes.append(box)

    resp = ''
    try:
      resp = call(im, req_types, req_boxes)
    except KeyboardInterrupt:
      break

    if im_type != 'head':
      im = cv2.imread(target_folder + im_file + '.jpg')
    else:
      im = cv2.imread(target_folder + im_file + '.tiff')

    for obj in resp.results:
      rect = req_boxes[obj.id]
      if obj.confidence >= 0.75:
        cv2.rectangle(im, (rect.x,rect.y), (rect.x+rect.w, rect.y+rect.h), (0,255,0), 2)
      elif obj.confidence >= 0.5:
        cv2.rectangle(im, (rect.x,rect.y), (rect.x+rect.w, rect.y+rect.h), (0,125,125), 2)
      else:
        cv2.rectangle(im, (rect.x,rect.y), (rect.x+rect.w, rect.y+rect.h), (0,0,255), 2)
      print obj.confidence
    
    #cv2.imshow('results',im)
    #cv2.waitKey(1)

    




  exit(1)












if __name__ == '__main__':
  rospy.init_node("ObjectStateClassifier_client")

  script_path = '/home/sanjif/programs/darknet/scripts/prepare/'

  # Get classes
  coco_classes = getClasses(script_path + 'coco/classes.csv')
  imagenet_classes = getClasses(script_path  + 'imagenet/classes.csv')
  all_classes = getClasses(script_path + 'obj.names')

  test_images = ['n04107743_1344','n04107743_1384','n04107743_1454','n04107743_1476']

  lamp_files = getImagePaths('light',coco_classes,imagenet_classes,script_path)
  light_files = getImagePaths('lamp',coco_classes,imagenet_classes,script_path)
  tv_files = getImagePaths('tv',coco_classes,imagenet_classes,script_path)

  files = lamp_files + light_files + tv_files

  file_types = ['lamp']*len(lamp_files) + ['light']*len(light_files) + ['tv']*len(tv_files)

  num_images = 10000

  print num_images
  print len(files)
  print len(file_types)

  for i in range(num_images):
    idx = random.randint(0,len(files)-1)

    file = files[idx]
    typ = file_types[idx]
    cls_id = getClsId(typ,all_classes)

    if typ == 'lamp' or typ == 'light':
      typ = 'lamp_light'
    
    try:
      bbs = getBBs(file,cls_id)
    except IOError:
      continue

    if len(bbs) > 0:
      im = cv2.imread(file)
      req_boxes = []
      req_types = []

      if len(bbs) == 1: 
        continue

      for bb in bbs:      
        box = cvRect()
        box.x = int(bb[0]*im.shape[1])
        box.y = int(bb[1]*im.shape[0])
        box.w = int(bb[2]*im.shape[1])
        box.h = int(bb[3]*im.shape[0])

        req_boxes.append(box)
        req_types.append(typ)

      try:
        call(im,req_types,req_boxes)
      except KeyboardInterrupt:
        break