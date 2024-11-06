#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>

#include <System.h>

#include "SharedMemoryInterface.h"

using namespace std;

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

void PrintTrackedMapPoints(
    const std::vector<ORB_SLAM3::MapPoint *> &myMapPoints) {
  // Print the number of tracked map points
  std::cout << "Number of tracked map points: " << myMapPoints.size()
            << std::endl;

  if (myMapPoints.empty()) {
    std::cout << "No map points to track!" << std::endl;
  } else {
    // Iterate through the vector and access map point data
    for (ORB_SLAM3::MapPoint *mp : myMapPoints) {
      if (mp) {
        // Get the position of the map point
        Eigen::Vector3f position = mp->GetWorldPos();
        std::cout << "Map Point ID: " << mp->mnId
                  << ", Position: " << position.transpose() << std::endl;

        // Get the reference keyframe
        ORB_SLAM3::KeyFrame *refKF = mp->GetReferenceKeyFrame();
        if (refKF) {
          std::cout << "  Reference KeyFrame ID: " << refKF->mnId << std::endl;
        } else {
          std::cout << "  No reference KeyFrame" << std::endl;
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  std::cout << "test test test";
  if (argc < 5) {
    cerr
        << endl
        << "Usage: ./mono_euroc path_to_vocabulary path_to_settings "
           "path_to_sequence_folder_1 path_to_times_file_1 "
           "(path_to_image_folder_2 path_to_times_file_2 ... "
           "path_to_image_folder_N path_to_times_file_N) (trajectory_file_name)"
        << endl;
    return 1;
  }

  const int num_seq = (argc - 3) / 2;
  cout << "num_seq = " << num_seq << endl;
  bool bFileName = (((argc - 3) % 2) == 1);
  string file_name;
  if (bFileName) {
    file_name = string(argv[argc - 1]);
    cout << "file name: " << file_name << endl;
  }

  // Load all sequences:
  int seq;
  vector<vector<string>> vstrImageFilenames;
  vector<vector<double>> vTimestampsCam;
  vector<int> nImages;

  vstrImageFilenames.resize(num_seq);
  vTimestampsCam.resize(num_seq);
  nImages.resize(num_seq);

  int tot_images = 0;
  for (seq = 0; seq < num_seq; seq++) {
    cout << "Loading images for sequence " << seq << "...";
    LoadImages(string(argv[(2 * seq) + 3]) + "/mav0/cam0/data",
               string(argv[(2 * seq) + 4]), vstrImageFilenames[seq],
               vTimestampsCam[seq]);
    cout << "LOADED!" << endl;

    nImages[seq] = vstrImageFilenames[seq].size();
    tot_images += nImages[seq];
  }

  // Vector for tracking time statistics
  vector<float> vTimesTrack;
  vTimesTrack.resize(tot_images);

  cout << endl << "-------" << endl;
  cout.precision(17);

  int fps = 20;
  float dT = 1.f / fps;
  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
  float imageScale = SLAM.GetImageScale();

  double t_resize = 0.f;
  double t_track = 0.f;

  std::vector<ORB_SLAM3::MapPoint *> myMapPoints;

    
  SharedMemoryInterface shm_interface;

  for (seq = 0; seq < num_seq; seq++) {

    // Main loop
    cv::Mat im;
    int proccIm = 0;
    for (int ni = 0; ni < nImages[seq]; ni++, proccIm++) {

      // Read image from file
      im = cv::imread(vstrImageFilenames[seq][ni],
                      cv::IMREAD_UNCHANGED); //,CV_LOAD_IMAGE_UNCHANGED);
      double tframe = vTimestampsCam[seq][ni];

      if (im.empty()) {
        cerr << endl
             << "Failed to load image at: " << vstrImageFilenames[seq][ni]
             << endl;
        return 1;
      }

      if (imageScale != 1.f) {
        int width = im.cols * imageScale;
        int height = im.rows * imageScale;
        cv::resize(im, im, cv::Size(width, height));
      }

      std::chrono::steady_clock::time_point t1 =
          std::chrono::steady_clock::now();

      // Pass the image to the SLAM system
      // cout << "tframe = " << tframe << endl;
      Sophus::SE3f Tcw = SLAM.TrackMonocular(im, tframe);


      std::chrono::steady_clock::time_point t2 =
          std::chrono::steady_clock::now();

      double ttrack =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
              .count();

      vTimesTrack[ni] = ttrack;

      // Wait to load the next frame
      double T = 0;
      if (ni < nImages[seq] - 1)
        T = vTimestampsCam[seq][ni + 1] - tframe;
      else if (ni > 0)
        T = tframe - vTimestampsCam[seq][ni - 1];

      // std::cout << "T: " << T << std::endl;
      // std::cout << "ttrack: " << ttrack << std::endl;

      if (ttrack < T) {
        // std::cout << "usleep: " << (dT-ttrack) << std::endl;
        usleep((T - ttrack) * 1e6); // 1e6
      }
      // Step 2: Retrieve the tracked map points and assign them to the new
      // vector
      myMapPoints = SLAM.GetTrackedMapPoints();
            // Get the camera position in world coordinates
      Sophus::SE3f Twc = Tcw.inverse();
      Eigen::Vector3f camera_position = Twc.translation();

      // Print the camera position
      std::cout << "Camera Position (World Coordinates): "
                << camera_position.transpose() << std::endl;

      // Optionally, print the rotation (as a matrix or quaternion)
      Eigen::Matrix3f rotation_matrix = Twc.rotationMatrix();
      std::cout << "Camera Rotation (World Coordinates): \n"
                << rotation_matrix << std::endl;

      shm_interface.writeData(myMapPoints, im, camera_position, rotation_matrix);

    }
    PrintTrackedMapPoints(myMapPoints);

    if (seq < num_seq - 1) {
      string kf_file_submap =
          "./SubMaps/kf_SubMap_" + std::to_string(seq) + ".txt";
      string f_file_submap =
          "./SubMaps/f_SubMap_" + std::to_string(seq) + ".txt";
      SLAM.SaveTrajectoryEuRoC(f_file_submap);
      SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file_submap);

      cout << "Changing the dataset" << endl;

      SLAM.ChangeDataset();
    }
  }
  // Stop all threads
  SLAM.Shutdown();

  // Save camera trajectory
  if (bFileName) {
    const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
    const string f_file = "f_" + string(argv[argc - 1]) + ".txt";
    SLAM.SaveTrajectoryEuRoC(f_file);
    SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
  } else {
    SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
  }

  return 0;
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps) {
  ifstream fTimes;
  fTimes.open(strPathTimes.c_str());
  vTimeStamps.reserve(5000);
  vstrImages.reserve(5000);
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
      double t;
      ss >> t;
      vTimeStamps.push_back(t * 1e-9);
    }
  }
}
