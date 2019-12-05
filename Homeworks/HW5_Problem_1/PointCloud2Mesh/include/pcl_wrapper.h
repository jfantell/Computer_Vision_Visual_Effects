//
// Created by John Fantell on 11/30/19.

#ifndef PCL_WRAPPER_H
#define PCL_WRAPPER_H

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <librealsense2/rs.hpp>

class PCL_wrapper{
public:
    PCL_wrapper();
    ~PCL_wrapper();
    pcl::PolygonMesh create_mesh(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_xyz);
    static pcl::PointCloud<pcl::PointXYZ>::Ptr points_to_pcl(const rs2::points& points, const rs2::video_frame& color);
private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr octree_voxel_downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr crop_box_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr statistical_outlier_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    pcl::PointCloud<pcl::PointNormal>::Ptr estimate_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    pcl::PolygonMesh greedy_projection_mesh_reconstruction(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_with_normals);
};

#endif //PCL_WRAPPER_H
