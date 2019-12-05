#include <iostream>
#include "../include/pcl_wrapper.h"
#include <pcl/io/vtk_io.h>
#include <librealsense2/rs.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

using namespace std;

int main(int argc, char** argv) {
    //Mode: 1 - New scan; Mode 2 - Load existing model (Not implemented)
    if(argc < 2){
        cout << "Please specify a mode" << endl;
        return 1;
    }
    int mode = atoi(argv[1]);

    //Create new point cloud object to store cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    //Mode 1, create new scan
    rs2::pipeline pipe;
    rs2::config cfg;
    rs2::pointcloud pc; // Point cloud object
    rs2::points points; // RealSense points object

    //Mode 2, load existing point cloud
    const char * path;

    if(mode == 1) {
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

        rs2::pipeline_profile profile = pipe.start(cfg);
    }
    else if(mode == 2) {
        if(argc < 3) {
            cout << "Please specify a file path for existing point cloud" << endl;
            return 1;
        }
        path = argv[2];
        cout << "Path " << path << endl;
    }
    while (1) {
        if(mode == 1){
            //Frames
            auto frames = pipe.wait_for_frames();
            if (!frames) {
                cerr << "Error occured while attempting to get camera frames" << endl;
                return 1;
            }

            // Wait for the next set of frames from the RealSense Camera
            auto color = frames.get_color_frame();

            // Tell pointcloud object to map to this color frame
            pc.map_to(color);

            auto depth = frames.get_depth_frame();

            // Generate the pointcloud and texture mappings
            points = pc.calculate(depth);

            // Convert RealSense point cloud to PCL point cloud
            cloud = PCL_wrapper::points_to_pcl(points, color);
        }
        else if(mode == 2){
            //Open file
            /// NOT WORKING
            pcl::io::loadPLYFile(path,*cloud);
        }

        // Cloud stats
        cout << "Cloud Size: " << cloud->size() << endl;

        //Create mesh
        PCL_wrapper pcl_wrapper;
        pcl::PolygonMesh mesh = pcl_wrapper.create_mesh(cloud);

        //Get local time
        time_t t = time(0);   // get time now
        struct tm *now = localtime(&t);
        char time_buffer[80];
        strftime(time_buffer, 80, "%Y-%m-%d-%H-%M-%S.", now);
        cout << "Current Time " << time_buffer << endl;

        //Save filename with current local timestamp
        pcl::io::saveOBJFile((string) "final_point_cloud_" + string(time_buffer) + (string) ".obj", mesh);
        pcl::io::savePolygonFileSTL((string) "final_point_cloud_" + string(time_buffer) + (string) ".stl", mesh);
//            pcl::io::saveVTKFile((string)"final_point_cloud_" + string(time_buffer) + (string) ".vtk",mesh);
//            pcl::io::savePolygonFilePLY("final_point_cloud_" + string(time_buffer) + (string) ".ply",mesh);

        //No loop necessary for single point cloud file
        if(mode == 2){
            break;
        }

        //Wait for user input
        cout << "View the mesh vtk or obj file...if you would like to take a new scan, press c" << endl;
        cout << "Otherwise, press any key and the program will terminate" << endl;
        char o;

        cin >> o;
        cout << o << endl;

        if (o != 'c') {
            break;
        }
    }
    return 0;
}