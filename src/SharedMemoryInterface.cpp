#include "SharedMemoryInterface.h"
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <cstring>

using namespace boost::interprocess;

SharedMemoryInterface::SharedMemoryInterface() {
  try {
    std::cout << "Size of SharedState: " << sizeof(SharedState) << " bytes\n";
    std::cout << "Alignment of SharedState: " << alignof(SharedState)
              << " bytes\n";

    std::cout << "Size of bool: " << sizeof(bool) << " bytes\n";
    std::cout << "Alignment of bool: " << alignof(bool) << " bytes\n";

    std::cout << "Size of size_t: " << sizeof(size_t) << " bytes\n";
    std::cout << "Alignment of size_t: " << alignof(size_t) << " bytes\n";

    std::cout << "Size of int: " << sizeof(int) << " bytes\n";
    std::cout << "Alignment of int: " << alignof(int) << " bytes\n";

    std::cout << "Size of CameraPose: " << sizeof(CameraPose) << " bytes\n";
    std::cout << "Alignment of CameraPose: " << alignof(CameraPose)
              << " bytes\n";
    // Remove any existing shared memory and semaphores
    shared_memory_object::remove("orbslam_shared_mem");
    shared_memory_object::remove("orbslam_state");
    named_semaphore::remove("orbslam_mutex");
    named_semaphore::remove("orbslam_data_ready");
    named_semaphore::remove("orbslam_data_processed");

    // Create shared memory objects
    shm_obj =
        shared_memory_object(create_only, "orbslam_shared_mem", read_write);
    state_shm = shared_memory_object(create_only, "orbslam_state", read_write);

    // Create named semaphores
    mutex = new named_semaphore(create_only, "orbslam_mutex", 1);
    data_ready = new named_semaphore(create_only, "orbslam_data_ready", 0);
    data_processed =
        new named_semaphore(create_only, "orbslam_data_processed", 1);

    // Set size for data and state
    shm_obj.truncate(1024 * 1024 * 50); // 50MB for image and points
    state_shm.truncate(sizeof(SharedState));

    // Map the shared memory
    region = mapped_region(shm_obj, read_write);
    state_region = mapped_region(state_shm, read_write);

    // Initialize the shared state
    state = new (state_region.get_address()) SharedState();
    state->new_data_available = false;
    state->processing_complete = true;

    // Get the data pointer
    data = static_cast<char *>(region.get_address());

  } catch (interprocess_exception &ex) {
    std::cerr << "Error initializing shared memory: " << ex.what() << std::endl;
    throw;
  }
}

void SharedMemoryInterface::writeData(
    const std::vector<ORB_SLAM3::MapPoint *> &mapPoints, const cv::Mat &image,
    const Eigen::Vector3f &camera_pos, const Eigen::Matrix3f &camera_rot) {

  // Acquire mutex
  mutex->wait();

  // // Wait until Python has processed the previous data
  // while (!state->processing_complete) {
  //     mutex->post();  // Release mutex while waiting
  //     data_processed->wait();  // Wait for processing completion
  //     mutex->wait();  // Reacquire mutex
  // }

  try {
    // Write points
    std::cout << "Size of SharedPoint: " << sizeof(SharedPoint) << std::endl;

    SharedPoint *points = reinterpret_cast<SharedPoint *>(data);
    state->num_points = mapPoints.size();
    for (size_t i = 0; i < mapPoints.size(); ++i) {
      if (mapPoints[i]) {
        Eigen::Vector3f pos = mapPoints[i]->GetWorldPos();
        points[i].x = pos.x();
        points[i].y = pos.y();
        points[i].z = pos.z();
        points[i].id = mapPoints[i]->mnId;
      }
    }

    // Write image
    char *image_data = data + sizeof(SharedPoint) * mapPoints.size();
    state->image_width = image.cols;
    state->image_height = image.rows;
    state->image_size = image.total() * image.elemSize();
    std::memcpy(image_data, image.data, state->image_size);

    // Write camera pose
    std::memcpy(state->camera_pose.translation, camera_pos.data(),
                3 * sizeof(float));
    std::memcpy(state->camera_pose.rotation, camera_rot.data(),
                9 * sizeof(float));

    // Signal that new data is available
    state->new_data_available = true;
    state->processing_complete = false;

  } catch (std::exception &ex) {
    mutex->post();
    throw;
  }

  // Release mutex and signal data ready
  mutex->post();
  data_ready->post();
}

SharedMemoryInterface::~SharedMemoryInterface() {
  try {
    // Clean up semaphores
    delete mutex;
    delete data_ready;
    delete data_processed;

    // Remove shared memory objects
    shared_memory_object::remove("orbslam_shared_mem");
    shared_memory_object::remove("orbslam_state");

    // Remove named semaphores
    named_semaphore::remove("orbslam_mutex");
    named_semaphore::remove("orbslam_data_ready");
    named_semaphore::remove("orbslam_data_processed");

  } catch (...) {
    // Suppress any exceptions in destructor
  }
}

