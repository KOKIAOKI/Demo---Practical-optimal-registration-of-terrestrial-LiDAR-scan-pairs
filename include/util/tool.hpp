#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

// pcl file
#include <pcl/io/auto_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

struct PoseStamp
{
  double time;
  Eigen::Vector3d trans;
  Eigen::Quaterniond quat;

  Eigen::Matrix4d getMatrix() const
  {
    Eigen::Isometry3d T_temp = Eigen::Isometry3d::Identity();
    T_temp.translation() = trans;
    T_temp.linear() = quat.toRotationMatrix();
    Eigen::Matrix4d T = T_temp.matrix();
    return T;
  }
};

std::vector<std::string> split(const std::string &input, char delimiter)
{
  std::istringstream stream(input);
  std::string field;
  std::vector<std::string> result;
  while (std::getline(stream, field, delimiter))
  {
    result.push_back(field);
  }
  return result;
}

std::vector<PoseStamp> readGroundTruth(const std::string &file_name)
{
  std::ifstream ifs(file_name);
  std::string line;

  std::vector<PoseStamp> gt_vec;

  // skip first line
  std::getline(ifs, line);
  while (std::getline(ifs, line))
  {
    std::vector<std::string> data = split(line, ',');
    std::vector<double> row;
    for (int i = 0; i < data.size(); i++)
    {
      row.push_back(std::stod(data[i]));
    }

    PoseStamp pose_stamp;
    double time = row[0] + row[1] / 1e9;

    pose_stamp.time = time;
    pose_stamp.trans = Eigen::Vector3d(row[2], row[3], row[4]);

    Eigen::Quaterniond quat(row[8], row[5], row[6], row[7]);
    quat.normalize();
    pose_stamp.quat = quat;

    gt_vec.emplace_back(pose_stamp);
  }

  return gt_vec;
}

std::vector<std::string> find_point_cloud_files(const std::string &path)
{
  std::filesystem::path dir(path);
  std::vector<std::string> files;

  if (!std::filesystem::exists(dir))
  {
    std::cout << "[ERROR] Cannot open folder" << std::endl;
    return files;
  }

  for (const auto &entry : std::filesystem::directory_iterator(dir))
  {
    const std::string extension = entry.path().extension().string();
    if (extension == ".pcd" || extension == ".ply")
    {
      files.emplace_back(entry.path().string());
    }
  }

  return files;
}

bool can_convert_to_double(const std::vector<std::string> &name_vec)
{
  for (const auto &name : name_vec)
  {
    try
    {
      std::stod(std::filesystem::path(name).stem().string());
    }
    catch (const std::invalid_argument &e)
    {
      return false;
    }
    catch (const std::out_of_range &e)
    {
      return false;
    }
  }
  return true;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr load_point_cloud_file(const std::string &path)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
  const auto extension = std::filesystem::path(path).extension().string();
  if (extension == ".pcd" || extension == ".ply")
  {
    if (pcl::io::load<pcl::PointXYZ>(path, *cloud_ptr) == -1)
    {
      std::cout << "[WARN] Can not open pcd file: " << path << std::endl;
      return cloud_ptr;
    }
  }
  return cloud_ptr;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr create_tar_cloud(const std::vector<std::string> &cloud_set)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
  for (const auto &file : cloud_set)
  {
    const auto cloud_temp_ptr = load_point_cloud_file(file);
    *cloud_ptr += *cloud_temp_ptr;
  }
  return cloud_ptr;
}