import os
import numpy as np

import cv2
import open3d as o3d

import glob

from load_utils import *


if __name__=='__main__':

    # File path
    Parsing_Dir = '/root/data/Waymo/parsed'
    Seperate = '/training'
    Seq = '/0000'
    Scene = '/scene_04'

    camera_path       = Parsing_Dir + Seperate + Seq + Scene + '/camera'
    camera_label_path = Parsing_Dir + Seperate + Seq + Scene + '/camera_label'
    lidar_path        = Parsing_Dir + Seperate + Seq + Scene + '/lidar'
    lidar_label_path  = Parsing_Dir + Seperate + Seq + Scene + '/lidar_label'
    spherical_img     = Parsing_Dir + Seperate + Seq + Scene + '/spherical_img'
    img_indx_path     = Parsing_Dir + Seperate + Seq + Scene + '/image_indx'

    data_len = len(glob.glob(camera_path + "/*FRONT.png")) 

    for frame_idx in range(data_len):

        ################################################################
        # Load Camera Image and Visualization
        ################################################################
        # Load camera images
        img_front, img_front_left, img_front_right, img_side_left, img_side_right = \
                                                                load_camera_image(camera_path, frame_idx)

        # Load camera labels
        label_front, label_front_left, label_front_right, label_side_left, label_side_right = \
                                                                load_camera_label(camera_label_path, frame_idx)

        # Show camera images with 2d bbox
        show_camera_image_with_label('Front', img_front, label_front)
        show_camera_image_with_label('Front_left', img_front_left, label_front_left)
        show_camera_image_with_label('Front_right', img_front_right, label_front_right)
        show_camera_image_with_label('Side_left', img_side_left, label_side_left)
        show_camera_image_with_label('Side_right', img_side_right, label_side_right)


        ################################################################
        # Load LiDAR ponits and Visualization
        ################################################################
        # Load lidar points
        lidar_points = load_points(lidar_path, frame_idx)   # [x, y, z, intensity]
        print(lidar_points.shape)

        # Load lidar labels
        lidar_labels = load_lidar_label(lidar_label_path, frame_idx)

        # Show lidar points with 3d bbox
        # show_points_with_label(lidar_points, lidar_labels)


        ################################################################
        # Load Spherical images and Visualization
        ################################################################
        range_img, intensity_img, elongation_img = load_spherical_image(spherical_img, frame_idx)
        print(range_img.shape)
        show_spherical_image('range', range_img)
        show_spherical_image('intensity', intensity_img)
        show_spherical_image('elongation', elongation_img)
        # spherical_view = cv2.vconcat([np_range_image_range, np_range_image_intensity, np_range_image_elongation])
        # cv2.imshow('spherical images', spherical_view)


        ################################################################
        # Create Projection images and Visualization
        ################################################################
        proj_indx = load_proj_indx(img_indx_path, frame_idx)
        print(proj_indx.shape)
        proj_range_img, proj_intensity_img = create_proj_img(proj_indx, lidar_points, img_front)

        show_proj_image('proj_range', proj_range_img)
        show_proj_image('proj_intensity', proj_intensity_img)


        points_on_img = create_points_on_image(proj_indx, img_front)
        show_proj_image('points on front image', points_on_img)
        

        if cv2.waitKey(0) & 0xFF == 27:
            break
        elif cv2.waitKey(0) & 0xFF == ord('n'):
            continue

            
            


    