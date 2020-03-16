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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import open3d as o3d
import pcl




#################################################################################################
# Visualize Range Images
#################################################################################################
def get_range_image(range_images, laser_name, return_index):
    """Returns range image given a laser name and its return index."""
    '''
        index
        0 : first return
        1 : second return
    '''
    return range_images[laser_name][return_index]


def show_range_image(range_image, frame_num, save_path):
    """Shows range image.

    Args:
        range_image: the range image data from a given lidar of type MatrixFloat.
        layout_index_start: layout offset
    """
    range_image_tensor = tf.convert_to_tensor(range_image.data)
    range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
    lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
    range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                  tf.ones_like(range_image_tensor) * 1e10)

    '''
        channel 0: range (see spherical coordinate system definition)
        channel 1: lidar intensity
        channel 2: lidar elongation
        channel 3: is_in_nlz (1 = in, -1 = not in)
    '''
    range_image_range = range_image_tensor[...,0] 
    range_image_intensity = range_image_tensor[...,1]
    range_image_elongation = range_image_tensor[...,2]

    np_range_image_range = range_image_range.numpy()
    np_range_image_intensity = range_image_intensity.numpy()
    np_range_image_elongation = range_image_elongation.numpy()
    # print(np_range_image_range.shape)   # (64,2650)

    # normalization (0~1)
    np_range_image_range[np_range_image_range>75.0] = 0.0    # range : set 0.0 if value is bigger than 75m
    np_range_image_range /= 75.0
    np_range_image_intensity[np_range_image_intensity>1.0] = 0.0    # intensity : set 0.0 if value is bigger than 1.0
    np_range_image_elongation[np_range_image_elongation>1.0] = 0.0    # elongation : set 0.0 if value is bigger than 1.0


    spherical_img = cv2.vconcat([np_range_image_range, np_range_image_intensity, np_range_image_elongation])
    cv2.imshow('spherical images', spherical_img)

    # save image (0~255 gray scale images)
    save_range_image = np_range_image_range*255.0
    save_intensity_image = np_range_image_intensity*255.0
    save_elongation_image = np_range_image_elongation*255.0
    save_spherical_image(save_range_image, 'range', frame_num, save_path)
    save_spherical_image(save_intensity_image, 'intensity', frame_num, save_path)
    save_spherical_image(save_elongation_image, 'elongation', frame_num, save_path)
    

def save_spherical_image(image, img_name, frame_num, save_path):
    """Save camera images."""
    save_path_ = save_path + "/spherical/" + str(frame_num).zfill(6) + "_" + img_name + ".png"
    cv2.imwrite(save_path_, image)
    # txt_path = save_path + "/spherical/" + str(frame_num).zfill(6) + "_" + img_name + ".txt"
    # np.savetxt( txt_path, image)
    

#################################################################################################
# Visualize Pointcloud
#################################################################################################
def get_bbox(label):
    '''
            X
            ^
            |
            |
    Y <------

    bounding box parameters
            0 -------- 3
           /|         /|
          1 -------- 2 .
          | |        | |
          . 4 -------- 7
          |/         |/
          5 -------- 6
    '''

    # Points
    points = []
    
    c_x = label.box.center_x    # center x
    c_y = label.box.center_y
    c_z = label.box.center_z
    L = label.box.length        # length
    W = label.box.width
    H = label.box.height
    yaw = label.box.heading     # yaw (heading)
    

    # 0
    points.append(
        [
            c_x - 0.5*W*np.sin(yaw) + 0.5*L*np.cos(yaw),     # x
            c_y + 0.5*W*np.cos(yaw) + 0.5*L*np.sin(yaw),      # y
            c_z + 0.5*H              # z
        ]
    ) 
    # 1
    points.append(
        [
            c_x - 0.5*W*np.sin(yaw) - 0.5*L*np.cos(yaw),     # x
            c_y + 0.5*W*np.cos(yaw) - 0.5*L*np.sin(yaw),      # y
            c_z + 0.5*H              # z
        ]
    ) 
    # 2
    points.append(
        [
            c_x + 0.5*W*np.sin(yaw) - 0.5*L*np.cos(yaw),     # x
            c_y - 0.5*W*np.cos(yaw) - 0.5*L*np.sin(yaw),      # y
            c_z + 0.5*H              # z
        ]
    ) 
    # 3
    points.append(
        [
            c_x + 0.5*W*np.sin(yaw) + 0.5*L*np.cos(yaw),     # x
            c_y - 0.5*W*np.cos(yaw) + 0.5*L*np.sin(yaw),      # y
            c_z + 0.5*H              # z
        ]
    ) 
    # 4
    points.append(
        [
            c_x - 0.5*W*np.sin(yaw) + 0.5*L*np.cos(yaw),     # x
            c_y + 0.5*W*np.cos(yaw) + 0.5*L*np.sin(yaw),      # y
            c_z - 0.5*H              # z
        ]
    ) 
    # 5
    points.append(
        [
            c_x - 0.5*W*np.sin(yaw) - 0.5*L*np.cos(yaw),     # x
            c_y + 0.5*W*np.cos(yaw) - 0.5*L*np.sin(yaw),      # y
            c_z - 0.5*H              # z
        ]
    ) 
    # 6
    points.append(
        [
            c_x + 0.5*W*np.sin(yaw) - 0.5*L*np.cos(yaw),     # x
            c_y - 0.5*W*np.cos(yaw) - 0.5*L*np.sin(yaw),      # y
            c_z - 0.5*H              # z
        ]
    ) 
    # 7
    points.append(
        [
            c_x + 0.5*W*np.sin(yaw) + 0.5*L*np.cos(yaw),     # x
            c_y - 0.5*W*np.cos(yaw) + 0.5*L*np.sin(yaw),      # y
            c_z - 0.5*H              # z
        ]
    ) 
    # Lines
    lines = [
        [0, 1],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 7],
        [5, 6],
        [6, 7],
    ]

    # opend3d line set
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    # color
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def get_arrow(label):
    pass


def show_points_with_label(points, frame):
    draw_list = []

    # Show Point cloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    draw_list.append(o3d_pcd)

    # draw bounding box
    for laser_label in frame.laser_labels:
        line_set = get_bbox(laser_label)

        Type     = laser_label.type

        draw_list.append(line_set)

    # print(np.asarray(pcd.points))      # change pcd to numpy array
    o3d.visualization.draw_geometries(draw_list)      # draw pcd

def save_points(points, save_path, frame_num):
    txt_path = save_path + "/lidar/" + str(frame_num).zfill(6) + '_lidar' + ".txt"
    np.savetxt(txt_path, points)

def save_labels(frame, save_path, frame_num):
    label_path = save_path + "/lidar_label/" + str(frame_num).zfill(6) + '_lidar_label' + ".txt"
    f = open(label_path, 'w')

    # label
    for laser_label in frame.laser_labels:
        '''
            < Label >
            id       : Object ID
            num_lidar_points_in_box : The total number of lidar points in this box.
            center_x : (m)
            center_y : (m)
            center_z : (m)
            length   : (m)
            width    : (m)
            height   : (m)
            heading  : yaw -3.14 ~3.14
            type     : class
            detection_difficulty_level : The difficulty level of this label. The higher the level, the harder it is.
            tracking_difficulty_level
        '''
        Type     = laser_label.type
        Id       = laser_label.id
        # in_points = laser_label.num_lidar_points_in_box
        center_x = laser_label.box.center_x
        center_y = laser_label.box.center_y
        center_z = laser_label.box.center_z
        length   = laser_label.box.length
        width    = laser_label.box.width
        height   = laser_label.box.height
        heading  = laser_label.box.heading
        det_diff_level = laser_label.detection_difficulty_level
        track_diff_level = laser_label.tracking_difficulty_level

        # label_list = [Type, Id, in_points, center_x, center_y, center_z, length, height, heading, det_diff_level, track_diff_level]
        label_list = [Type, Id, center_x, center_y, center_z, length, height, heading, det_diff_level, track_diff_level]
        indx_num = len(label_list)
        for indx in range(indx_num):
            if indx == (indx_num-1):
                f.write(str(label_list[indx]))
            else:
                f.write(str(label_list[indx]) + ' ')
        f.write('\n')
    f.close()



if __name__=='__main__':

    '''
    Read one frame
    Each file in the dataset is a sequence of frames ordered by frame start timestamps. 
    We have extracted two frames from the dataset to demonstrate the dataset format.
    '''
    # File path
    Data_Dir = '/root/data/Waymo'
    Seq = '/0000/training'
    FILENAME = '/original/segment-1022527355599519580_4866_960_4886_960_with_camera_labels.tfrecord'
    file_path = Data_Dir + Seq + FILENAME
    save_path = Data_Dir + Seq

    # Load frames
    dataset = tf.data.TFRecordDataset(file_path, compression_type='')
    frames = []
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(frame)

    # Read one frame
    for frame_num, frame in enumerate(frames):
        (range_images, camera_projections,range_image_top_pose) = \
                                        frame_utils.parse_range_image_and_camera_projection(frame)

        # print(frame.context)      # frame information


        '''
            Visualize Range Images
        '''
        frame.lasers.sort(key=lambda laser: laser.name)
        # show_range_image(get_range_image(range_images, open_dataset.LaserName.TOP, 0), frame_num, save_path)
        # #show_range_image(get_range_image(range_images, open_dataset.LaserName.TOP, 1), frame_num, save_path)     # second return 
        # if cv2.waitKey(0) & 0xFF == 27:
        #     break
        # elif cv2.waitKey(0) & 0xFF == ord('n'):
        #     continue

        '''
            Point Cloud Conversion and Visualization
        '''
        points, cp_points, xyzI_points = frame_utils.convert_range_image_to_point_cloud( frame,
                                                                                         range_images,
                                                                                         camera_projections,
                                                                                         range_image_top_pose )
        # pcl_XYZI = pcl.PointCloud_PointXYZI()
        # pcl_XYZI.from_array(xyzI_points_all)
        # print(pcl_XYZI)
        # pcl_XYZI.to_file('test.pcd')
        # # print(      camera_projections,
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
        print('points : ', points_all.shape)
        print('xyzI points : ', xyzI_points_all.shape)

        # # convert numpy to pcl
        # pcl_XYZI = pcl.PointCloud_PointXYZI()
        # pcl_XYZI.from_array(xyzI_points_all)
        # print(pcl_XYZI)
        # pcl_XYZI.to_file('test.pcd')
        # # print( pcl_XYZI.size() )
        
        # visualization & Save
        show_points_with_label(points_all, frame)

        save_points(xyzI_points_all, save_path, frame_num)

        save_labels(frame, save_path, frame_num)
        



