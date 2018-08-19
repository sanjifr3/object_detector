#!/usr/bin/env python

## Return in the same order passed!! ## 

import cv2
import os
import csv
import argparse
import rospkg
import rospy
import numpy as np
from numpy import random
import pandas as pd
import shutil

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, Model, load_model
from keras.layers import Input, GlobalAveragePooling2D, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras import backend
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from object_detector.msg import cvRect, classificationInfo
from object_detector.srv import *

model_types = ['head','lamp_light','tv'] # ['head','lamp_light','tv'] # ,['lamp_light','tv']
threshold = 0.6

rospy.init_node("ObjectStateClassifier_server")

if rospy.has_param(rospy.get_name() + '/classification_tol'):
  threshold = float(rospy.get_param(rospy.get_name() + '/classification_tol'))

if rospy.has_param(rospy.get_name() + '/classifier_types'):
  print rospy.get_param(rospy.get_name() + '/classifier_types')
  model_types = rospy.get_param(rospy.get_name() + '/classifier_types').split(',')

def get_session(gpu_fraction=0.15):
  '''
  30% assuming that you have 6 GB of ram and want to allocate 2 GB for Keras
  '''

  num_threads = os.environ.get('OMP_NUM_THREADS')
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

  if num_threads:
    return tf.Session(config=tf.ConfigProto(
      gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
  else:
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

rospack = rospkg.RosPack()
packagePath = rospack.get_path('object_detector')
dataDir = packagePath + '/models/keras/'

models = {e1:{'input_size':0} for e1 in model_types}

for typ in model_types:
  df = pd.read_csv(dataDir + typ + '.csv')
  models[typ]['input_size'] = int(df[df['cv_acc_mean'] == df['cv_acc_mean'].max()]['input_size'].values[0])
  models[typ]['model'] = load_model(dataDir + typ + '_3fold.h5')
  if typ == 'head': channels = 1
  else: channels = 3
  models[typ]['model'].predict(np.zeros((1,models[typ]['input_size'],models[typ]['input_size'],channels)))

rospy.loginfo("[ObjectStateClassifier_server] Enabled w/ threshold of %.2f with the following classes:"%threshold)
print "   ", model_types



def imageCb(data, types, rects):
  if len(types[0]) == 1:
    types = [''.join(types)]

  try:
    if 'head' not in types:
      cv_image = CvBridge().imgmsg_to_cv2(data,"bgr8")
    else:
      cv_image = CvBridge().imgmsg_to_cv2(data,"mono16")
  except CvBridgeError as e:
    print e

  if len(rects) == 0:
    rects = [cvRect()]
    rects[0].x = 0
    rects[0].y = 0
    rects[0].w = cv_image.shape[1]
    rects[0].h = cv_image.shape[0]

  X = {e1:{
          "im":[],
          "id":[]
      } for e1 in model_types}

  draw_im = CvBridge().imgmsg_to_cv2(data,"bgr8")

  idx = 0

  #print('cv_image shape:',cv_image.shape)
  for rect, typ in zip(rects,types):
    #print('Rectangle:', rect.y, ":",rect.y+rect.h,",",
    #        rect.x, ":", rect.x + rect.w)

    if typ == "lamp" or typ == "light":
      typ = "lamp_light"
    elif typ == "tvmonitor":
      typ = "tv"
    elif typ == 'heads':
      typ = 'head'

    cv2.imwrite('/home/sanjif/Database/im.tiff',cv_image)

    im = cv_image[rect.y:rect.y+rect.h, rect.x:rect.x+rect.w].copy()
    im = cv2.resize(im, (models[typ]['input_size'],models[typ]['input_size']))

    X[typ]['im'].append(im)
    X[typ]['id'].append(idx)
    idx += 1 

  resp = object_state_classifierResponse()

  for typ, data in X.items():
    if len(data['im']) == 0:
      continue
    x_test = data['im']
    ids = data['id']

    x_test = np.array(x_test)
    x_test = x_test.astype('float32')
    if len(x_test.shape) == 3:
        x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

    max_val = 2**8-1
    if typ == 'head': max_val = 2**16-1

    x_test /= max_val

    #print x_test.max()

    predictions = models[typ]['model'].predict(x_test)

    for (idx,prediction) in zip(ids,predictions):
      obj = classificationInfo()
      obj.confidence = prediction[0]
      obj.state = 1 if obj.confidence > threshold else 0
      obj.id = idx
      resp.results.append(obj)

  return resp 

  # #cv2.imshow("cv image", draw_im)
  # #cv2.waitKey(1)

  #   #   if len(im.shape) == 3:
  #   #   im = im / (2**8-1)
  #   # else:
  #   #   im = im.reshape(im.shape[0],im.shape[1],1) / (2**16-1)


  # for typ, x_test in X.items():
  #   x_test = np.array(x_test)

  #   print x_test.shape
   

  #   im = np.array(im).astype('float32')

  #   # print typ, x_test.shape
  #   if len(x_test.shape) != 4:
  #     continue
  #   predictions = models[typ]['model'].predict(x_test)
  #   for prediction in predictions:
  #     obj = classificationInfo()
  #     obj.confidence = prediction[0]
  #     obj.state = 1 if obj.confidence >= threshold else 0
  #     resp.results.append(obj)

  # for rect, obj in zip(rects,resp.results):
  #   if obj.confidence >= 0.75:
  #     cv2.rectangle(draw_im, (rect.x,rect.y), (rect.x+rect.w, rect.y+rect.h), (0,255,0), 2)
  #   elif obj.confidence >= 0.5:
  #     cv2.rectangle(draw_im, (rect.x,rect.y), (rect.x+rect.w, rect.y+rect.h), (125,125,0), 2)
  #   else:
  #     cv2.rectangle(draw_im, (rect.x,rect.y), (rect.x+rect.w, rect.y+rect.h), (255,0,0), 2)

  # cv2.imwrite('/home/sanjif/Database/detected.jpg',draw_im)

  # return resp

def handle_OC(req):
  resp = object_state_classifierResponse()

  try:
    resp = imageCb(req.im, req.types, req.rects)
  except rospy.ROSInterruptException:
    resp.states = classificationInfo()
    rospy.logerr("[ObjectStateClassifier_server] ROS Interrupt!")

  return resp

def objClassifier_server():
  s = rospy.Service("ObjectStateClassification", object_state_classifier, handle_OC)
  rospy.loginfo("[ObjectStateClassifier_server] Object Classification Server Enabled!")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print rospy.loginfo('[ObjectStateClassifier_server] Shutting down...')

if __name__ == '__main__':
  objClassifier_server()

# Error:
# [ERROR] [1531508527.172922]: Error processing request: /tmp/binarydeb/ros-kinetic-opencv3-3.3.1/modules/imgproc/src/resize.cpp:3939: error: (-215) ssize.width > 0 && ssize.height > 0 in function resize

# ['Traceback (most recent call last):\n', '  File "/opt/ros/kinetic/lib/python2.7/dist-packages/rospy/impl/tcpros_service.py", line 625, in _handle_request\n    response = convert_return_to_response(self.handler(request), self.response_class)\n', '  File "/home/sanjif/mia-robot/mia-vision/src/object_detector/scripts/classifier/stateClassifierServer.py", line 119, in handle_OC\n    resp = imageCb(req.im, req.types, req.rects)\n', '  File "/home/sanjif/mia-robot/mia-vision/src/object_detector/scripts/classifier/stateClassifierServer.py", line 88, in imageCb\n    im = cv2.resize(im, (models[typ][\'input_size\'],models[typ][\'input_size\']))\n', 'error: /tmp/binarydeb/ros-kinetic-opencv3-3.3.1/modules/imgproc/src/resize.cpp:3939: error: (-215) ssize.width > 0 && ssize.height > 0 in function resize\n\n']
# [ERROR] [1531508527.173193282]: Service call failed: service [/ObjectStateClassification] responded with an error: error processing request: /tmp/binarydeb/ros-kinetic-opencv3-3.3.1/modules/imgproc/src/resize.cpp:3939: error: (-215) ssize.width > 0 && ssize.height > 0 in function resize
