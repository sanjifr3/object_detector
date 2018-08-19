#include <object_detector/YoloObjectDetector.hpp>

YoloObjectDetector::YoloObjectDetector(){
  loadROSParams();
  init(model_file_, data_file_);
  
  ROS_INFO("[YoloObjectDetector] Yolo Object Detector Initalized!");
}

YoloObjectDetector::YoloObjectDetector(std::string model_file, std::string data_file){
  loadROSParams();
  init(model_file, data_file);
  
  ROS_INFO("[YoloObjectDetector] Yolo Object Detector Initalized!");
}

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh, ros::NodeHandle rqt_nh): server_(rqt_nh){
  this->nh_ = nh;

  loadROSParams();

  f_ = boost::bind(&YoloObjectDetector::rqtCb, this, _1, _2);
  server_.setCallback(f_);
  
  init(model_file_,data_file_);

  sc_client_ = nh_.serviceClient<object_detector::object_state_classifier>("ObjectStateClassification");
  //if (determine_state_)
  //  ros::service::waitForService("ObjectStateClassification");

  ROS_INFO("[YoloObjectDetector] Yolo Object Detector Initalized!");
}

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh, ros::NodeHandle rqt_nh, std::string model_file, std::string data_file): server_(rqt_nh){
  this->nh_ = nh;

  loadROSParams();

  f_ = boost::bind(&YoloObjectDetector::rqtCb, this, _1, _2);
  server_.setCallback(f_);
  
  init(model_file, data_file);

  sc_client_ = nh_.serviceClient<object_detector::object_state_classifier>("ObjectStateClassification");
  //if (determine_state_)
  //  ros::service::waitForService("ObjectStateClassification");


  ROS_INFO("[YoloObjectDetector] Yolo Object Detector Initalized!");
}


YoloObjectDetector::~YoloObjectDetector(){
}

void YoloObjectDetector::init(std::string& model_file, std::string& data_file){
  std::string name_file = darknet_dir_ + name_dir_;
  
  if (data_file == "combine9k")
    name_file += "9k";
  else
    name_file += data_file;

  std::string weight_file = darknet_dir_;
  std::string cfg_file = darknet_dir_;

  if(data_file == "obj"){
    weight_file += my_weights_dir_ + model_file;
    cfg_file += my_cfg_dir_ + model_file;
  }

  else{
    weight_file += weights_dir_ + model_file;
    cfg_file += cfg_dir_ + model_file;

    if (data_file == "voc" || data_file == "voc8"){
      weight_file += "-voc";
      cfg_file += "-voc";
    }
  }

  weight_file += ".weights";
  cfg_file += ".cfg";
  name_file += ".names";
  
  ROS_INFO("[YoloObjectDetector] Reading classes from %s",name_file.c_str());
  ROS_INFO("[YoloObjectDetector] Reading network configuration from %s",cfg_file.c_str());
  ROS_INFO("[YoloObjectDetector] Reading weights from %s",weight_file.c_str());

  getClasses(name_file);
  setupNetwork(cfg_file, weight_file,name_file);
  return;
}

void YoloObjectDetector::getClasses(std::string data_file){

  ROS_INFO("[YoloObjectDetector] Parsing %s!",data_file.c_str());

  std::ifstream file (data_file.c_str(), std::ifstream::in );

  classes_.clear();

  if (!file){
    ROS_ERROR("[YoloObjectDetector] Could not access classes file: %s", data_file.c_str());
    return;
  }

  std::string line, cls;
  while (getline(file,line)){
    classes_.push_back(line);
  }

  ROS_INFO("[YoloObjectDetector] Loaded with %d classes!",(int)classes_.size());

  colors_.resize(classes_.size());
  int col_incr = floor(255/float(classes_.size()));

  for (unsigned int i = 0; i < classes_.size(); i++){
    if (verbose_) std::cout << "   " << classes_[i] << std::endl;

    colors_[i] = cv::Scalar(255-col_incr*i, 0 + col_incr*i, 255 - col_incr*i);
  }

  return;
}

void YoloObjectDetector::setupNetwork(std::string& cfg_file, std::string& weight_file, std::string& name_file){
  std::cout << "Loading network...\n" << std::endl;
  ROS_INFO("[YoloObjectDetector] Loading network...");
  net_ = parse_network_cfg_custom(strdup(cfg_file.c_str()),1); //set batch=1
  ROS_INFO("[YoloObjectDetector] Network cfg file parsed");;
  load_weights(&net_, strdup(weight_file.c_str()));
  ROS_INFO("[YoloObjectDetector] Weights Loaded!");;
  //set_batch_network(&net_, 1);

  srand(2222222);
  
  ROS_INFO("[YoloObjectDetector] Network Ready!");;

  // Initialize results arrays

  layer l = net_.layers[net_.n-1];
  int numDetections = l.n*l.w*l.h;

  avg_ = (float *) calloc(l.outputs, sizeof(float));
  
  predictions_ = (float **) calloc(frames_, sizeof(float*));
  for (int j = 0; j < frames_; ++j){
    predictions_[j] = (float *) calloc(l.outputs, sizeof(float));
    //images_[j] = make_image(1,1,3);
  }

  boxes_ = (box *) calloc(numDetections, sizeof(box));
  
  probs_ = (float **) calloc(numDetections, sizeof(float *));
  for (int j = 0; j < numDetections; ++j)
    probs_[j] = (float*)calloc(l.classes, sizeof(float *));
    //probs_[j] = (float *) calloc(l.classes + 1, sizeof(float));
      
  return;
}

IplImage* YoloObjectDetector::toIplImage(cv::Mat &im){
  IplImage* ROS_img = new IplImage(im);
  return ROS_img;
}

void YoloObjectDetector::getImage(cv::Mat& im){
  IplImage* ROS_img = toIplImage(im);
  dnIm_ = ipl_to_image(ROS_img);
  delete ROS_img;
  ROS_img = NULL;

  dnIms_ = letterbox_image(dnIm_, net_.w, net_.h);

  free(dnIm_.data);
  return;
}


std::vector<RPS::objectInfo> YoloObjectDetector::detect(cv::Mat &im){
  std::vector<RPS::objectInfo> results;

  getImage(im);

  layer l = net_.layers[net_.n-1];
  int numDetections = l.n*l.w*l.h;

  float *X = dnIms_.data;
  float *predictions = network_predict(net_, X);
  free(dnIms_.data);

  memcpy(predictions_[index_], predictions, l.outputs*sizeof(float));
  mean_arrays(predictions_, frames_, l.outputs, avg_);

  l.output = avg_;

  int letter = 0;
  int nboxes = 0;
  detection *dets = get_network_boxes(&net_, det_.w, det_.h, thresh_, heir_, 0, 1, &nboxes, letter);
  if (nms_ > 0)
    do_nms_obj(dets, nboxes, l.classes, nms_);

  for(int i = 0 ; i < nboxes; i++){
    box b = dets[i].bbox;

    float xmin = std::max(0.0,(b.x - (b.w / 2.)));
    float xmax = std::min(1.0,(b.x + (b.w / 2.)));
    float ymin = std::max(0.0,(b.y - (b.h / 2.)));
    float ymax = std::min(1.0,(b.y + (b.h / 2.)));

    float height = ymax-ymin;
    float width = xmax-xmin;

    if(width > 0.01 && height > 0.01){
      RPS::objectInfo obj;

      // Find class with the highest probabability greater than the threshold
      for (int j=0; j < l.classes; j++){
        if(dets[i].prob[j] > thresh_ && dets[i].prob[j] > obj.det_conf){
          obj.det_conf = dets[i].prob[j];
          obj.id = j;
          obj.color = colors_[j];
        }
      }

      // If a class with a high probability is found, and the classes vector is populated,
      // and the id number is valid continue
      if(obj.id != -1 && classes_.size() != 0 && obj.id < classes_.size()){
        // Get name
        obj.name = classes_[obj.id];

        // Object from dn given in percentage of net_
        // Rescale to get object position in net_.w x net_.h
        obj.loc = cv::Rect(
          xmin*net_.w,
          ymin*net_.h,
          width*net_.w,
          height*net_.h
        );

        // Shift origin to center
        obj.loc.x -= (net_.w/2.);
        obj.loc.y -= (net_.h/2.);

        // Rescale to im size
        obj.loc.x *= im.cols/float(net_.w);
        obj.loc.y *= im.rows/float(net_.h);
        obj.loc.width *= im.cols/float(net_.w);
        obj.loc.height *= im.rows/float(net_.h);

        // Rescale y for letterbox scaling
        obj.loc.y *= im.rows/float(net_.h);
        obj.loc.height *= im.rows/float(net_.h);

        // Shift origin to top left in im.cols x im.rows
        obj.loc.x += (im.cols/2.);
        obj.loc.y += (im.rows/2.);

        results.push_back(obj);
      }
    }
  }

  free(dets);

  if(determine_state_)
    classify(im, results);

  return results;
}

std::vector<dn_yolo::results> YoloObjectDetector::detect(cv::Mat& im, bool draw){
  if (time_ == 0) time_ = getWallTime();  

  std::vector<dn_yolo::results> results;

  getImage(im);

  layer l = net_.layers[net_.n-1];
  int numDetections = l.n*l.w*l.h;

  float *X = dnIms_.data;
  float *predictions = network_predict(net_, X);
  free(dnIms_.data);

  memcpy(predictions_[index_], predictions, l.outputs*sizeof(float));
  mean_arrays(predictions_, frames_, l.outputs, avg_);

  l.output = avg_;

  int letter = 0;
  int nboxes = 0;
  detection *dets = get_network_boxes(&net_, det_.w, det_.h, thresh_, heir_, 0, 1, &nboxes, letter);
  if (nms_ > 0)
    do_nms_obj(dets, nboxes, l.classes, nms_);

  for(int i = 0 ; i < nboxes; i++){
    box b = dets[i].bbox;

    float xmin = std::max(0.0,(b.x - (b.w / 2.)));
    float xmax = std::min(1.0,(b.x + (b.w / 2.)));
    float ymin = std::max(0.0,(b.y - (b.h / 2.)));
    float ymax = std::min(1.0,(b.y + (b.h / 2.)));

    float height = ymax-ymin;
    float width = xmax-xmin;

    if(width > 0.01 && height > 0.01){
      dn_yolo::results obj;

      // Find class with the highest probabability greater than the threshold
      for (int j=0; j < l.classes; j++){
        if(dets[i].prob[j] > thresh_ && dets[i].prob[j] > obj.det_conf){
          obj.det_conf = dets[i].prob[j];
          obj.class_id = j;
        }
      }

      // If a class with a high probability is found, and the classes vector is populated,
      // and the id number is valid continue
      if(obj.class_id != -1 && classes_.size() != 0 && obj.class_id < classes_.size()){
        // Get name
        obj.name = classes_[obj.class_id];
        
        // Object from dn given in percentage of net_
        // Rescale to get object position in net_.w x net_.h
        obj.loc = cv::Rect(
          xmin*net_.w,
          ymin*net_.h,
          width*net_.w,
          height*net_.h
        );

        // Shift origin to center
        obj.loc.x -= (net_.w/2.);
        obj.loc.y -= (net_.h/2.);

        // Rescale to im frame
        obj.loc.x *= im.cols/float(net_.w);
        obj.loc.y *= im.rows/float(net_.h);
        obj.loc.width *= im.cols/float(net_.w);
        obj.loc.height *= im.rows/float(net_.h);

        // Rescale y for letterbox scaling
        obj.loc.y *= im.rows/float(net_.h);
        obj.loc.height *= im.rows/float(net_.h);

        // Shift origin to top left in im.cols x im.rows
        obj.loc.x += (im.cols/2.);
        obj.loc.y += (im.rows/2.);

        results.push_back(obj);
      }
    }
  }

  free(dets);

  if(determine_state_){
    classify(im, results);
  }

  if(verbose_) printFPS();
  if(verbose_) printResults(results);
  if(draw) drawResults(im, results);

  return results;
}

void YoloObjectDetector::classify(cv::Mat& im, std::vector<RPS::objectInfo>& objects){
  if(objects.size() == 0)
    return;

  object_detector::object_state_classifier sc_msg;
  std::vector<int> ids;

  cv::Rect search_frame = cv::Rect(5000,5000,-5000,-5000);

  for(unsigned int i = 0; i < objects.size(); i++){
    // Check if class should be classified
    if(utils::findInVector(objects[i].name, classes_to_classify_) != -1){
      ids.push_back(i);
      search_frame.x = std::min(objects[i].loc.x, search_frame.x);
      search_frame.y = std::min(objects[i].loc.y, search_frame.y);

      // Using width and height parameter to store max x, and y for now
        // correct after for loop
      search_frame.width = std::max(objects[i].loc.x + objects[i].loc.width, search_frame.width);
      search_frame.height = std::max(objects[i].loc.y + objects[i].loc.height, search_frame.height);
    }
  }

  // No class that needs to be classified found
  if(ids.size() == 0)
    return;

  sc_msg.request.types.resize(ids.size());
  sc_msg.request.rects.resize(ids.size());

  search_frame.width = search_frame.width - search_frame.x;
  search_frame.height = search_frame.height - search_frame.y;

  cv::Mat search_window = im(search_frame);

  cv_bridge::CvImage(std_msgs::Header(), "bgr8", search_window).toImageMsg(sc_msg.request.im);

  for(unsigned int i = 0; i < ids.size(); i++){
    cv::Rect rect = objects[ids[i]].loc;
    sc_msg.request.types[i] = objects[ids[i]].name;
    sc_msg.request.rects[i].x = rect.x - search_frame.x;
    sc_msg.request.rects[i].y = rect.y - search_frame.y;
    sc_msg.request.rects[i].w = rect.width;
    sc_msg.request.rects[i].h = rect.height;
  }

  ros::Time local_time = ros::Time::now();

  ros::service::waitForService("ObjectStateClassification");
  sc_client_.call(sc_msg);
  
  std::cout << "  Server Call: " << (ros::Time::now() - local_time).toSec() << std::endl;
  local_time = ros::Time::now();

  for(unsigned int i = 0; i < sc_msg.response.results.size(); i++){
    objects[ids[i]].state = sc_msg.response.results[i].state;
    objects[ids[i]].rec_conf = sc_msg.response.results[i].confidence;
  }

  return;
}

void YoloObjectDetector::classify(cv::Mat& im, std::vector<dn_yolo::results>& objects){
  if(objects.size() == 0)
    return;

  object_detector::object_state_classifier sc_msg;
  std::vector<int> ids;

  cv::Rect search_frame = cv::Rect(5000,5000,-5000,-5000);

  for(unsigned int i = 0; i < objects.size(); i++){
    // Check if class should be classified
    if(utils::findInVector(objects[i].name, classes_to_classify_) != -1){
      ids.push_back(i);
      search_frame.x = std::min(objects[i].loc.x, search_frame.x);
      search_frame.y = std::min(objects[i].loc.y, search_frame.y);

      // Using width and height parameter to store max x, and y for now
        // correct after for loop
      search_frame.width = std::max(objects[i].loc.x + objects[i].loc.width, search_frame.width);
      search_frame.height = std::max(objects[i].loc.y + objects[i].loc.height, search_frame.height);
    }
  }

  // No class that needs to be classified found
  if(ids.size() == 0)
    return;

  sc_msg.request.types.resize(ids.size());
  sc_msg.request.rects.resize(ids.size());

  search_frame.width = search_frame.width - search_frame.x;
  search_frame.height = search_frame.height - search_frame.y;

  cv::Mat search_window = im(search_frame);

  cv_bridge::CvImage(std_msgs::Header(), "bgr8", search_window).toImageMsg(sc_msg.request.im);

  for(unsigned int i = 0; i < ids.size(); i++){
    cv::Rect rect = objects[ids[i]].loc;
    sc_msg.request.types[i] = objects[ids[i]].name;
    sc_msg.request.rects[i].x = rect.x - search_frame.x;
    sc_msg.request.rects[i].y = rect.y - search_frame.y;
    sc_msg.request.rects[i].w = rect.width;
    sc_msg.request.rects[i].h = rect.height;
  }

  ros::Time local_time = ros::Time::now();

  ros::service::waitForService("ObjectStateClassification");
  sc_client_.call(sc_msg);
  
  std::cout << "  Server Call: " << (ros::Time::now() - local_time).toSec() << std::endl;
  local_time = ros::Time::now();

  for(unsigned int i = 0; i < sc_msg.response.results.size(); i++){
    objects[ids[i]].state = sc_msg.response.results[i].state;
    objects[ids[i]].rec_conf = sc_msg.response.results[i].confidence;
  }

  return;
}

void YoloObjectDetector::drawResults(cv::Mat& im, std::vector<dn_yolo::results>& results, int wait){
  for(unsigned int i = 0; i < results.size(); i++){
    cv::rectangle(im, results[i].loc, colors_[results[i].class_id], 3);
    int x = std::max(results[i].loc.tl().x,0);
    int y = std::max(results[i].loc.tl().y - 10, 0);
    std::string box_text = cv::format("%s (%.2f): State: %d (%.2f)",results[i].name.c_str(), results[i].det_conf, results[i].state, results[i].rec_conf);
    cv::putText(im, box_text, cv::Point(x,y), 2, 0.5, colors_[results[i].class_id], 1.0);
  }

  cv::imshow("YOLO",im);
  cv::waitKey(wait);

  return;
}

void YoloObjectDetector::printResults(std::vector<dn_yolo::results>& results){
  for(unsigned int i = 0; i < results.size(); i++){
    std::cout << i << " " <<  results[i].name << " [" << results[i].class_id << "], Prob: ";
    std::cout << results[i].det_conf << ", State: " << results[i].state << " [";
    std::cout << results[i].rec_conf << "] at " << results[i].loc << std::endl;
  }
}

double YoloObjectDetector::getWallTime(){
  struct timeval time;
  if (gettimeofday(&time, NULL)) {
    return 0;
  }
  return (double) time.tv_sec + (double) time.tv_usec * .000001;
}

void YoloObjectDetector::printFPS(){
  std::cout << "FPS " << 1./(getWallTime() - time_ + 0.00000001) << std::endl;
  time_ = getWallTime();
  return;
}

void YoloObjectDetector::loadROSParams(std::string ns){
  // Global Params
  // std::string ns = "PersonDetector"; //ros::this_node::getName();

  // Local Params
  ns += "/YoloObjectDetector";

  nh_.param(ns + "/model_file", model_file_,model_file_);
  nh_.param(ns + "/data_file", data_file_, data_file_);

  nh_.param(ns + "/verbose", verbose_, verbose_);

  nh_.param(ns + "/thresh", thresh_, thresh_);
  nh_.param(ns + "/heir_thresh", heir_, heir_);
  nh_.param(ns + "/nms", nms_, nms_);

  nh_.param(ns + "/frames", avg_frames_);
  nh_.param(ns + "/frames", frames_,frames_);

  ROS_INFO("[YoloObjectDetector] Params updated by ROS param server");
}

void YoloObjectDetector::rqtCb(object_detector::ObjectDetectorRQTConfig& config, uint32_t level){
  if ( level == 0 ){
    verbose_ = config.verbose;
    thresh_ = config.thresh;
    heir_ = config.heir_thresh;
    nms_ = config.nms;
    
    ROS_INFO("[YoloObjectDetector] Params updated: \n"
             "                     - Verbose: %s \n"
             "                     - Threshold: %.2f \n"
             "                     - Heirarchy Threshold: %.2f \n"
             "                     - NMS Threshold: %.2f",
             verbose_?"True":"False", thresh_, heir_, nms_);
  }
}
