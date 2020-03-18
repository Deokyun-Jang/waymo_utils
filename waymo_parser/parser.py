import os
import tensorflow as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import cv2
import open3d as o3d

import glob

from parser_utils import *


if __name__=='__main__':

    '''
    Read one frame
    Each file in the dataset is a sequence of frames ordered by frame start timestamps. 
    We have extracted two frames from the dataset to demonstrate the dataset format.
    '''
    # File path
    Data_Dir = '/root/data/Waymo'
    Seq = '/0000'
    # Seperate = '/training'      # /training, or /validation
    Seperate = '/validation'

    # filename_list = glob.glob(Data_Dir + Seq + "/tf_training/*.tfrecord")   # /tf_training/*.tfrecord or /tf_validation/*.tfrecord
    filename_list = glob.glob(Data_Dir + Seq + "/tf_validation/*.tfrecord")

    for scene_num, filename in enumerate(filename_list):

        # make directory
        save_path = Data_Dir + Seq + Seperate + '/' + 'scene_' + str(scene_num).zfill(2) 
        os.makedirs(save_path, exist_ok=True)
        camera_save_path = save_path + '/camera'
        os.makedirs(camera_save_path, exist_ok=True)
        camera_label_save_path = save_path + '/camera_label'
        os.makedirs(camera_label_save_path, exist_ok=True)
        lidar_save_path = save_path + '/lidar'
        os.makedirs(lidar_save_path, exist_ok=True)
        lidar_label_save_path = save_path + '/lidar_label'
        os.makedirs(lidar_label_save_path, exist_ok=True)
        spherical_save_path = save_path + '/spherical_img'
        os.makedirs(spherical_save_path, exist_ok=True)
        image_indx_save_path = save_path + '/image_indx'
        os.makedirs(image_indx_save_path, exist_ok=True)
        

        # Load frames
        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        frames = []
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frames.append(frame)

        # Read one frame
        for frame_num, frame in enumerate(frames):
            print('Scene%d_%d' %(scene_num, frame_num) )

            # Save frame context
            # print(frame.context)      # frame information
            
            for index, image in enumerate(frame.images):
                ################################################################
                # Save Camera Image
                ################################################################
                # # visualize and save images
                # show_camera_image_cv(image, frame)      
                
                # save images
                save_camera_image(image, frame_num, camera_save_path)
                
                # save labels
                save_camera_labels(image, frame, frame_num, camera_label_save_path)


            ################################################################
            # Save Spherical Image (Range, Intensity, elongation)
            ################################################################
            (range_images, camera_projections,range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            frame.lasers.sort(key=lambda laser: laser.name)
                
            # Get Spherical Images
            spherical_img = get_spherical_image(get_range_image(range_images, open_dataset.LaserName.TOP, 0))
            #spherical_img = get_spherical_image(get_range_image(range_images, open_dataset.LaserName.TOP, 1))     # second return 

            # # Show Spherical images
            # show_spherical_img(spherical_img)
                
            # Save Spherical images
            save_spherical_image(spherical_img, frame_num, spherical_save_path)

            ################################################################
            # Save LiDAR Points (x, y, z, intensity)
            ################################################################
            points, cp_points, xyzI_points = frame_utils.convert_range_image_to_point_cloud( frame,
                                                                                     range_images,
                                                                                     camera_projections,
                                                                                     range_image_top_pose )
            # points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud( frame,
            #                                                                             range_images,
            #                                                                             camera_projections,
            #                                                                             range_image_top_pose,
            #                                                                             ri_index=1 )
            # points_all_ri2 = np.concatenate(points_ri2, axis=0)         # second return
            # cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)   # second return
            
            # 3d points in vehicle frame.
            points_all = np.concatenate(points, axis=0)                   # (126059,3)
            # camera projection corresponding to each point.
            cp_points_all = np.concatenate(cp_points, axis=0)             # (126059,6)
            # xyzI points
            xyzI_points_all = np.concatenate(xyzI_points, axis=0)
            
            # Check
            print('All points : ', points_all.shape)
            # print('xyzI points : ', xyzI_points_all.shape)
                
            # # Visualization 
            # show_points_with_label(points_all, frame)

            # Save points & label
            save_lidar_points(xyzI_points_all, lidar_save_path, frame_num)
            save_lidar_labels(frame, lidar_label_save_path, frame_num)
            

            ################################################################
            # Save image projection index in Front view image (LiDAR index, proj_width, proj_height)
            ################################################################
            images = sorted(frame.images, key=lambda i:i.name)

            # image index
            cp_points_all= cp_points_all.astype('int32')

            # masking points in image fov
            mask = np.where(cp_points_all[..., 0]==images[0].name)
            img_indx = cp_points_all[mask]    
            print('Points in Front image view : ', cp_points_all[mask].shape)

            projection_indx = np.concatenate( (np.array([mask[0]]).T, img_indx[...,1:3]), axis=1 )   # front img index (first return) 
            projection_indx = projection_indx.astype('int32')

            save_image_index(projection_indx, image_indx_save_path,  frame_num)


            print('===========================================')


            # if cv2.waitKey(0) & 0xFF == 27:
            #     break
            # elif cv2.waitKey(0) & 0xFF == ord('n'):
            #     continue

            
            


    