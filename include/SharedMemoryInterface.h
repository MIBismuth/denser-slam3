#pragma once

#include <Eigen/Core>
#include <System.h>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <opencv2/opencv.hpp>

class SharedMemoryInterface {
public:
  SharedMemoryInterface();
  ~SharedMemoryInterface();

  void writeData(const std::vector<ORB_SLAM3::MapPoint *> &mapPoints,
                 const cv::Mat &image, const Eigen::Vector3f &camera_pos,
                 const Eigen::Matrix3f &camera_rot);

private:
  #pragma pack(push, 1) // Disable padding for consistent struct size
  struct SharedPoint {
    float x, y, z;
    uint64_t id; // 8 bytes
  };
  #pragma pack(pop) // Reset padding after struct definition

  struct CameraPose {
    float translation[3];
    float rotation[9]; // 3x3 matrix in row-major order
  };

  struct SharedState {
    bool new_data_available;
    bool processing_complete;
    size_t num_points;
    size_t image_size;
    int image_width;
    int image_height;
    CameraPose camera_pose;
  };

  boost::interprocess::shared_memory_object shm_obj;
  boost::interprocess::shared_memory_object state_shm;
  boost::interprocess::mapped_region region;
  boost::interprocess::mapped_region state_region;
  boost::interprocess::named_semaphore *mutex;
  boost::interprocess::named_semaphore *data_ready;
  boost::interprocess::named_semaphore *data_processed;
  SharedState *state;
  char *data;
};
