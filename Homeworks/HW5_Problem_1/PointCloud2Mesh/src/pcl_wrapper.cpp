//
// Created by hayley on 11/20/19.
//

#include "../include/pcl_wrapper.h"
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <iostream>
#include <pcl/common/common.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <librealsense2/rs.hpp>
#include <pcl/filters/crop_box.h>

using namespace std;

PCL_wrapper::PCL_wrapper() {}
PCL_wrapper::~PCL_wrapper() {}

pcl::PointCloud<pcl::PointXYZ>::Ptr PCL_wrapper::points_to_pcl(const rs2::points& points, const rs2::video_frame& color){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto vertex_coordinates = points.get_vertices();
    for (int i=0; i < points.size(); i++) {
        //XYZ values
        cloud->points[i].x = vertex_coordinates[i].x;
        cloud->points[i].y = vertex_coordinates[i].y;
        cloud->points[i].z = vertex_coordinates[i].z;
    }
    return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PCL_wrapper::octree_voxel_downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud){
    float leaf = 0.01;
    pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZ> octree(leaf);
    octree.setInputCloud(cloud);
    octree.defineBoundingBox();
    octree.addPointsFromInputCloud();
    pcl::octree::OctreePointCloud<pcl::PointXYZ>::AlignedPointTVector centroids;
    octree.getVoxelCentroids(centroids);

    pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
    output->points.assign(centroids.begin(), centroids.end());
    output->width = uint32_t(centroids.size());
    output->height = 1;
    output->is_dense = true;
    return output;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PCL_wrapper::crop_box_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::CropBox<pcl::PointXYZ> box_filter;
    box_filter.setMin(Eigen::Vector4f(-0.5, -0.5, 0, 1.0));
    box_filter.setMax(Eigen::Vector4f(0.5, 0.5, 1.2, 1.0));
    box_filter.setInputCloud(cloud);
    box_filter.filter(*cloud_filtered);
    return cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PCL_wrapper::statistical_outlier_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_filtered);
    return cloud_filtered;
}

pcl::PointCloud<pcl::PointNormal>::Ptr PCL_wrapper::estimate_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud){
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);
    n.setNumberOfThreads(16);
    n.setInputCloud (cloud);
    n.setSearchMethod (tree);
    n.setKSearch (50);
    n.compute (*normals);

    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
    return cloud_with_normals;
}

pcl::PolygonMesh PCL_wrapper::greedy_projection_mesh_reconstruction(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_with_normals){
    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree_mesh (new pcl::search::KdTree<pcl::PointNormal>);
    tree_mesh->setInputCloud(cloud_with_normals);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (2);
    //    gp3.setSearchRadius (5);

    // Set typical values for the parameters
    gp3.setMu (2.5);
    gp3.setMaximumNearestNeighbors (100);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(false);

    // Get result
    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree_mesh);
    gp3.reconstruct (triangles);

    return triangles;
}

pcl::PolygonMesh PCL_wrapper::create_mesh(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_xyz) {
    cout << "Beginning Mesh Construction" << endl;

    //Uniform Sampling
    pcl::PointCloud<pcl::PointXYZ>::Ptr uniform_sample = octree_voxel_downsample(cloud_xyz);

    //Pass through filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr pass_through = crop_box_filter(uniform_sample);

    //Statistical Outlier
    pcl::PointCloud<pcl::PointXYZ>::Ptr sor = statistical_outlier_filter(pass_through);

    // Estimate normals
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals = estimate_normals(sor);

    // Construct mesh
    pcl::PolygonMesh triangles = greedy_projection_mesh_reconstruction(cloud_with_normals);

    cout << "Finished Mesh Construction" << endl;

    return triangles;
}