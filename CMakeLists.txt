cmake_minimum_required(VERSION 2.8.3)
project(object_detector)

# Set c++11 cmake flags
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# Define path of darknet folder here.
find_path(DARKNET_PATH
  NAMES "README.md"
  #HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../libraries/darknet/")
  #HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../programs/darknet/")
  HINTS "$ENV{HOME}/programs/darknet/")
message("-- Found DARKNET: ${DARKNET_PATH}")
add_definitions(-DDARKNET_FILE_PATH="${DARKNET_PATH}")
add_definitions(-DMDARKNET_FILE_PATH=${DARKNET_PATH}/sr)

# Find X11
MESSAGE ( STATUS "Searching for X11..." )
FIND_PACKAGE ( X11 REQUIRED )
IF ( X11_FOUND )
  INCLUDE_DIRECTORIES ( ${X11_INCLUDE_DIR} )
  LINK_LIBRARIES ( ${X11_LIBRARIES} )
  MESSAGE ( STATUS " X11_INCLUDE_DIR: " ${X11_INCLUDE_DIR} )
  MESSAGE ( STATUS " X11_LIBRARIES: " ${X11_LIBRARIES} )
ENDIF ( X11_FOUND )

# Find CUDA
FIND_PACKAGE(CUDA)
if (CUDA_FOUND)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
  message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
  set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
  )
  add_definitions(-DGPU)
else()
  list(APPEND LIBRARIES "m")
endif()

# Find Boost
find_package(Boost REQUIRED COMPONENTS thread)

# Find OpenCV
find_package (OpenCV 3.4.0 EXACT REQUIRED 
	NO_MODULE  # Should be optional, tells CMake to use config mode
	PATHS /usr/local # Look here
	NO_DEFAULT_PATH # And don't look anywhere else
)
include_directories(${OpenCV_INCLUDE_DIRS})
# Enable OPENCV in darknet
add_definitions(-DOPENCV)
add_definitions(-O4 -g)

# Find catkin packages
find_package(catkin REQUIRED
  COMPONENTS
    cv_bridge
    roscpp
    rospy
    std_msgs
    actionlib
    image_transport
    image_utils
    dynamic_reconfigure
)

add_message_files(
  DIRECTORY msg
  FILES
    cvRect.msg
    classificationInfo.msg
)

add_service_files(
  DIRECTORY srv
  FILES
    object_state_classifier.srv
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    sensor_msgs
    std_msgs
)

generate_dynamic_reconfigure_options(
  cfg/ObjectDetectorRQT.cfg
)

catkin_package(
  INCLUDE_DIRS
    #${DARKNET_PATH}/src
    #${DARKNET_PATH}/include
    include
  LIBRARIES
    darknet_lib yolo_lib
  CATKIN_DEPENDS
    cv_bridge
    roscpp
    actionlib
    rospy
    std_msgs
    image_transport
  DEPENDS
    Boost
)

include_directories(
  #${DARKNET_PATH}/src
  #${DARKNET_PATH}/include
  include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

if (CUDA_FOUND)
  link_directories(/usr/local/cuda/lib64)
  
  cuda_add_library(darknet_lib
    ${DARKNET_PATH}/src/darknet_nomain.c
    ${DARKNET_PATH}/src/image_interface.c
    
    ${DARKNET_PATH}/src/activation_kernels.cu       ${DARKNET_PATH}/src/avgpool_layer_kernels.cu
    ${DARKNET_PATH}/src/blas_kernels.cu             ${DARKNET_PATH}/src/col2im_kernels.cu
    ${DARKNET_PATH}/src/convolutional_kernels.cu    ${DARKNET_PATH}/src/crop_layer_kernels.cu
    ${DARKNET_PATH}/src/deconvolutional_kernels.cu  ${DARKNET_PATH}/src/dropout_layer_kernels.cu
    ${DARKNET_PATH}/src/im2col_kernels.cu           ${DARKNET_PATH}/src/maxpool_layer_kernels.cu
    ${DARKNET_PATH}/src/network_kernels.cu
   
    ${DARKNET_PATH}/src/activation_layer.c          ${DARKNET_PATH}/src/activations.c
    ${DARKNET_PATH}/src/art.c                       ${DARKNET_PATH}/src/avgpool_layer.c
    ${DARKNET_PATH}/src/batchnorm_layer.c           ${DARKNET_PATH}/src/blas.c
    ${DARKNET_PATH}/src/box.c                       ${DARKNET_PATH}/src/captcha.c
    ${DARKNET_PATH}/src/cifar.c                     ${DARKNET_PATH}/src/classifier.c
    ${DARKNET_PATH}/src/coco.c                      ${DARKNET_PATH}/src/col2im.c
    ${DARKNET_PATH}/src/compare.c                   ${DARKNET_PATH}/src/connected_layer.c
    ${DARKNET_PATH}/src/convolutional_layer.c       ${DARKNET_PATH}/src/cost_layer.c
    ${DARKNET_PATH}/src/cpu_gemm.c                  ${DARKNET_PATH}/src/crnn_layer.c
    ${DARKNET_PATH}/src/crop_layer.c                ${DARKNET_PATH}/src/cuda.c
    ${DARKNET_PATH}/src/data.c
    ${DARKNET_PATH}/src/deconvolutional_layer.c     ${DARKNET_PATH}/src/demo.c
    ${DARKNET_PATH}/src/detection_layer.c           ${DARKNET_PATH}/src/detector.c
    ${DARKNET_PATH}/src/dice.c                      ${DARKNET_PATH}/src/dropout_layer.c
    ${DARKNET_PATH}/src/gemm.c                      ${DARKNET_PATH}/src/getopt.c
    #${DARKNET_PATH}/src/gettimeofday.c             
    ${DARKNET_PATH}/src/go.c
    ${DARKNET_PATH}/src/gru_layer.c                 ${DARKNET_PATH}/src/im2col.c
    ${DARKNET_PATH}/src/image.c                     ${DARKNET_PATH}/src/layer.c
    ${DARKNET_PATH}/src/list.c                      ${DARKNET_PATH}/src/local_layer.c
    ${DARKNET_PATH}/src/matrix.c                    ${DARKNET_PATH}/src/maxpool_layer.c
    ${DARKNET_PATH}/src/network.c                   ${DARKNET_PATH}/src/nightmare.c
    ${DARKNET_PATH}/src/normalization_layer.c       ${DARKNET_PATH}/src/option_list.c
    ${DARKNET_PATH}/src/parser.c                    ${DARKNET_PATH}/src/region_layer.c
    ${DARKNET_PATH}/src/reorg_layer.c               ${DARKNET_PATH}/src/reorg_old_layer.c
    ${DARKNET_PATH}/src/rnn.c                       ${DARKNET_PATH}/src/rnn_vid.c
    ${DARKNET_PATH}/src/rnn_layer.c                 ${DARKNET_PATH}/src/route_layer.c
    ${DARKNET_PATH}/src/shortcut_layer.c            ${DARKNET_PATH}/src/softmax_layer.c
    ${DARKNET_PATH}/src/super.c                     ${DARKNET_PATH}/src/swag.c
    ${DARKNET_PATH}/src/tag.c                       ${DARKNET_PATH}/src/tree.c
    ${DARKNET_PATH}/src/upsample_layer.c            ${DARKNET_PATH}/src/utils.c
    ${DARKNET_PATH}/src/voxel.c                     ${DARKNET_PATH}/src/writing.c
    ${DARKNET_PATH}/src/yolo.c                      ${DARKNET_PATH}/src/yolo_layer.c
     
    ${DARKNET_PATH}/src/http_stream.cpp             
    #${DARKNET_PATH}/src/yolo_console_dll.cpp
    ${DARKNET_PATH}/src/yolo_v2_class.cpp
  )
  
  target_link_libraries(darknet_lib
    m
    pthread
    stdc++
    cuda
    cudart
    cublas
    curand
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
  )
  
  ## Libraries
  cuda_add_library(yolo_lib src/libraries/YoloObjectDetector.cpp)
else()
  add_library(darknet_lib
    ${DARKNET_PATH}/src/darknet_nomain.c
    ${DARKNET_PATH}/src/image_interface.c    
    
    ${DARKNET_PATH}/src/activation_layer.c          ${DARKNET_PATH}/src/activations.c
    ${DARKNET_PATH}/src/art.c                       ${DARKNET_PATH}/src/avgpool_layer.c
    ${DARKNET_PATH}/src/batchnorm_layer.c           ${DARKNET_PATH}/src/blas.c
    ${DARKNET_PATH}/src/box.c                       ${DARKNET_PATH}/src/captcha.c
    ${DARKNET_PATH}/src/cifar.c                     ${DARKNET_PATH}/src/classifier.c
    ${DARKNET_PATH}/src/coco.c                      ${DARKNET_PATH}/src/col2im.c
    ${DARKNET_PATH}/src/compare.c                   ${DARKNET_PATH}/src/connected_layer.c
    ${DARKNET_PATH}/src/convolutional_layer.c       ${DARKNET_PATH}/src/cost_layer.c
    ${DARKNET_PATH}/src/cpu_gemm.c                  ${DARKNET_PATH}/src/crnn_layer.c
    ${DARKNET_PATH}/src/crop_layer.c                ${DARKNET_PATH}/src/cuda.c
    ${DARKNET_PATH}/src/data.c
    ${DARKNET_PATH}/src/deconvolutional_layer.c     ${DARKNET_PATH}/src/demo.c
    ${DARKNET_PATH}/src/detection_layer.c           ${DARKNET_PATH}/src/detector.c
    ${DARKNET_PATH}/src/dice.c                      ${DARKNET_PATH}/src/dropout_layer.c
    ${DARKNET_PATH}/src/gemm.c                      ${DARKNET_PATH}/src/getopt.c
    #${DARKNET_PATH}/src/gettimeofday.c             
    ${DARKNET_PATH}/src/go.c
    ${DARKNET_PATH}/src/gru_layer.c                 ${DARKNET_PATH}/src/im2col.c
    ${DARKNET_PATH}/src/image.c                     ${DARKNET_PATH}/src/layer.c
    ${DARKNET_PATH}/src/list.c                      ${DARKNET_PATH}/src/local_layer.c
    ${DARKNET_PATH}/src/matrix.c                    ${DARKNET_PATH}/src/maxpool_layer.c
    ${DARKNET_PATH}/src/network.c                   ${DARKNET_PATH}/src/nightmare.c
    ${DARKNET_PATH}/src/normalization_layer.c       ${DARKNET_PATH}/src/option_list.c
    ${DARKNET_PATH}/src/parser.c                    ${DARKNET_PATH}/src/region_layer.c
    ${DARKNET_PATH}/src/reorg_layer.c               ${DARKNET_PATH}/src/reorg_old_layer.c
    ${DARKNET_PATH}/src/rnn.c                       ${DARKNET_PATH}/src/rnn_vid.c
    ${DARKNET_PATH}/src/rnn_layer.c                 ${DARKNET_PATH}/src/route_layer.c
    ${DARKNET_PATH}/src/shortcut_layer.c            ${DARKNET_PATH}/src/softmax_layer.c
    ${DARKNET_PATH}/src/super.c                     ${DARKNET_PATH}/src/swag.c
    ${DARKNET_PATH}/src/tag.c                       ${DARKNET_PATH}/src/tree.c
    ${DARKNET_PATH}/src/upsample_layer.c            ${DARKNET_PATH}/src/utils.c
    ${DARKNET_PATH}/src/voxel.c                     ${DARKNET_PATH}/src/writing.c
    ${DARKNET_PATH}/src/yolo.c                      ${DARKNET_PATH}/src/yolo_layer.c
     
    ${DARKNET_PATH}/src/http_stream.cpp             
    #${DARKNET_PATH}/src/yolo_console_dll.cpp
    ${DARKNET_PATH}/src/yolo_v2_class.cpp
  )

  target_link_libraries(darknet_lib
    m
    pthread
    stdc++
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
  )

  ## Libraries
  add_library(yolo_lib src/libraries/YoloObjectDetector.cpp)
endif()

target_link_libraries(yolo_lib darknet_lib)
add_dependencies(yolo_lib ${PROJECT_NAME}_gencfg)
