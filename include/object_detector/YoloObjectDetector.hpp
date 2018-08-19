#pragma once

#include <image_utils/Utilities.h>

#include <object_detector/object_state_classifier.h>

#include <dynamic_reconfigure/server.h>
#include <object_detector/ObjectDetectorRQTConfig.h>

// Darknet.
#ifdef GPU
 #include "cuda_runtime.h"
 #include "curand.h"
 #include "cublas_v2.h"
#endif

#define stringify(x) #x
#define FILE2(path, file) stringify(path ## file)
#define FILE(path, file) FILE2(path, file)

extern "C" {
  #include FILE(MDARKNET_FILE_PATH,c/demo_nomain.c)
  #include FILE(MDARKNET_FILE_PATH,c/network.h)
  #include FILE(MDARKNET_FILE_PATH,c/parser.h)
  #include FILE(MDARKNET_FILE_PATH,c/image.h)
  #include FILE(MDARKNET_FILE_PATH,c/utils.h)
  #include FILE(MDARKNET_FILE_PATH,c/box.h)
  #include FILE(MDARKNET_FILE_PATH,c/detection_layer.h)
  #include FILE(MDARKNET_FILE_PATH,c/region_layer.h)
  #include FILE(MDARKNET_FILE_PATH,c/cost_layer.h)  

  #include <sys/time.h>
  #include <time.h>
}

extern "C" image ipl_to_image(IplImage* src);
extern "C" void show_image_cv(image p, const char *name, IplImage *disp);

namespace dn_yolo{
  struct results{
    std::string name = "";
    int class_id = -1;
    double det_conf = 0.0;
    int state = 0;
    double rec_conf = 1.0;
    cv::Rect loc = cv::Rect(0,0,0,0);
  };
}

class YoloObjectDetector{
  public: // Functions
    YoloObjectDetector();
    YoloObjectDetector(std::string model_file, std::string data_file);
    YoloObjectDetector(ros::NodeHandle nh, ros::NodeHandle rqt_nh);
    YoloObjectDetector(ros::NodeHandle nh, ros::NodeHandle rqt_nh, std::string model_file, std::string data_file);
    ~YoloObjectDetector();
    std::vector<RPS::objectInfo> detect(cv::Mat&im);
    std::vector<dn_yolo::results> detect(cv::Mat& im, bool draw);
    void drawResults(cv::Mat& im, std::vector<dn_yolo::results>& results, int wait=1);
    void printResults(std::vector<dn_yolo::results>& results);
    void printFPS();

  private: // Functions
    void init(std::string& model_file, std::string& data_file);
    void getClasses(std::string data_file);
    void setupNetwork(std::string& cfg_file, std::string& weight_file, std::string& name_file);

    double getWallTime();

    IplImage* toIplImage(cv::Mat& im);
    void getImage(cv::Mat& im);
    
    void classify(cv::Mat& im, std::vector<RPS::objectInfo>& objects);
    void classify(cv::Mat& im, std::vector<dn_yolo::results>& objects);    

    void loadROSParams(std::string ns=ros::this_node::getName());
    void rqtCb(object_detector::ObjectDetectorRQTConfig& config, uint32_t level);

  private: // Variables
    ros::NodeHandle nh_;
    
    // Service Clients
    ros::ServiceClient sc_client_;

    // RQT Reconfigure
    dynamic_reconfigure::Server<object_detector::ObjectDetectorRQTConfig> server_;
    dynamic_reconfigure::Server<object_detector::ObjectDetectorRQTConfig>::CallbackType f_;
    char* pHome = std::getenv("HOME");
  
    // Directories
    std::string darknet_dir_ = (std::string)pHome + "/programs/darknet/";
    std::string darknet_dir2_ = (std::string)pHome + "/casper-vision/src/darknet_ros/darknet_ros/yolo_network_config/";
    std::string weights_dir_ = "weights/";
    std::string name_dir_ = "data/";
    std::string cfg_dir_ = "cfg/";
    
    std::string my_weights_dir_ = "my-weights/";
    std::string my_cfg_dir_ = "my-cfg/";

    std::string model_file_ = "yolov2-tiny"; // "yolo","tiny-yolo","tiny-yolo8","yolo9000"
    std::string data_file_ = "coco"; // "coco","voc","combine9k"
    
    // Classes
    std::vector<std::string> classes_;
    std::vector<cv::Scalar> colors_;    

    // Networks
    network net_;

    float *avg_;
    float **predictions_;
    float **probs_;
    box *boxes_;

    image dnIm_;
    image dnIms_;
    image det_;

    int avg_frames_ = 1;
    double thresh_ = 0.3;
    double heir_ = 0.5;
    double nms_ = 0.4;
    int frames_ = 1;
    int index_ = 0;

    bool verbose_ = true;
    bool determine_state_ = true;

    double time_ = 0;
    
    std::vector<std::string> classes_to_classify_ = {"tv","lamp","light","tvmonitor"};
};