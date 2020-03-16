import os
import tensorflow as tf
import math
import numpy as np
import itertools
import glob

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import cv2
import open3d as o3d




#################################################################################################
# Visualize Camera Images and Camera Labels & SAVE
#################################################################################################
'''
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
cyan = (255, 255, 0)
magenta = (255, 0, 255)
'''
def get_corners(label):
    corners = np.zeros((4, 2), dtype=np.float32)
    '''
    1----4
    |    |
    |    |
    2----3
    '''

    # 1
    corners[0, 0] = label.box.center_x - 0.5 * label.box.length
    corners[0, 1] = label.box.center_y - 0.5 * label.box.width

    # 2
    corners[1, 0] = label.box.center_x - 0.5 * label.box.length
    corners[1, 1] = label.box.center_y + 0.5 * label.box.width

    # 3
    corners[2, 0] = label.box.center_x + 0.5 * label.box.length
    corners[2, 1] = label.box.center_y + 0.5 * label.box.width

    # 4
    corners[3, 0] = label.box.center_x + 0.5 * label.box.length
    corners[3, 1] = label.box.center_y - 0.5 * label.box.width

    return corners

def show_camera_image_cv(camera_image, frame):
    """Show a camera image and the given camera labels."""

    img = tf.image.decode_jpeg(camera_image.image).numpy()

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)     # RGB --> BGR (opencv)
    img_cv_labels = img_cv.copy()
    
    # Draw the camera labels.
    for camera_labels in frame.camera_labels:       # Front, Front right, Front left, Side right, Side left
        # Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != camera_image.name:
            continue
        
        # Iterate over the individual labels.
        for label in camera_labels.labels:
        
            # get corner points (img-coord.)
            img_corners = get_corners(label).reshape(-1, 1, 2).astype(int)

            # Draw the object bounding box.
            cv2.polylines(img_cv_labels, [img_corners], True, (0, 0, 255), 1)
    
    img_name = str( open_dataset.CameraName.Name.Name(camera_image.name) )
    
    # Show the camera image.
    cv2.imshow(img_name, img_cv_labels)



def save_camera_image(camera_image, frame_num, camera_save_path_):
    """Save camera images."""
    
    img = tf.image.decode_jpeg(camera_image.image).numpy()      # load image
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)     # RGB --> BGR (opencv)

    img_name = str( open_dataset.CameraName.Name.Name(camera_image.name) )
    
    save_path_ = camera_save_path_ + '/' + str(frame_num).zfill(3) + '_' + img_name + ".png"     # save path

    cv2.imwrite(save_path_, img_cv)


def save_camera_labels(camera_image, frame, frame_num, camera_label_save_path):
    '''
        < Label >
        id       : Object ID (tracking)
        center_x : (m)
        center_y : (m)
        length   : (m)
        width    : (m)
        type     : class (0:Unkown, 1:vehicle, 2:pedestrain, 3:sign, 4:cyclist)
        detection_difficulty_level : The difficulty level of this label. The higher the level, the harder it is.
        tracking_difficulty_level
    '''
    img_name = str( open_dataset.CameraName.Name.Name(camera_image.name) )
    label_path = camera_label_save_path + '/' + str(frame_num).zfill(3) + '_' + img_name + ".txt"
    f = open(label_path, 'w')
    
    for camera_labels in frame.camera_labels:
        # Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != camera_image.name:
            continue

        # Iterate over the individual labels.
        for label in camera_labels.labels:

            Type     = label.type
            Id       = label.id
            center_x = label.box.center_x
            center_y = label.box.center_y
            length   = label.box.length
            width    = label.box.width
            det_diff_level = label.detection_difficulty_level
            track_diff_level = label.tracking_difficulty_level
            
            # label_list = [Type, Id, center_x, center_y, length, width, det_diff_level, track_diff_level]
            label_list = [Type, Id, center_x, center_y, length, width, det_diff_level, track_diff_level]
            
            indx_num = len(label_list)
            for indx in range(indx_num):
                if indx == (indx_num-1):
                    f.write(str(label_list[indx]))
                else:
                    f.write(str(label_list[indx]) + ' ')
            f.write('\n')
    
    f.close()


#################################################################################################
# Visualize Spherical Images and Save
#################################################################################################
def get_range_image(range_images, laser_name, return_index):
    """Returns range image given a laser name and its return index."""
    '''
        index
        0 : first return
        1 : second return
    '''
    return range_images[laser_name][return_index]


def get_spherical_image(range_image):
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

    # # data check --> trash value (10000000.0)
    # print(np.max(np_range_image_range))
    # print(np.min(np_range_image_range))
    # print(np.max(np_range_image_intensity))
    # print(np.min(np_range_image_intensity))
    # print(np.max(np_range_image_elongation))
    # print(np.min(np_range_image_elongation))
    # print(np_range_image_range.shape[0]*np_range_image_range.shape[1])
    # print( len(np.where(np_range_image_range>75.0)[0]) )
    # print(np_range_image_range[np_range_image_range>75.0])

    # normalization (0~1)
    np_range_image_range /= 75.0
    np_range_image_range[np_range_image_range>1.0] = -1    # range : set 75.0 if value is bigger than 75m
    np_range_image_intensity[np_range_image_intensity>1.0] = -1    # intensity : set 0.0 if value is bigger than 1.0
    np_range_image_elongation[np_range_image_elongation>1.0] = -1    # elongation : set 0.0 if value is bigger than 1.0

    # np_range_image_range[np_range_image_range>75.0] = 75.0    # range : set 75.0 if value is bigger than 75m
    # np_range_image_range /= 75.0
    # np_range_image_intensity[np_range_image_intensity>1.0] = 0.0    # intensity : set 0.0 if value is bigger than 1.0
    # np_range_image_elongation[np_range_image_elongation>1.0] = 0.0    # elongation : set 0.0 if value is bigger than 1.0

    spherical_img = np.array([np_range_image_range, np_range_image_intensity, np_range_image_elongation])

    return spherical_img


def show_spherical_img(spherical_img):
    np_range_image_range = spherical_img[0,...]
    np_range_image_intensity =spherical_img[1,...]
    np_range_image_elongation = spherical_img[2,...]

    spherical_view = cv2.vconcat([np_range_image_range, np_range_image_intensity, np_range_image_elongation])
    cv2.imshow('spherical images', spherical_view)


def save_spherical_image(spherical_img, frame_num, save_path):
    """Save camera images."""
    # image Save path
    save_path_range = save_path + '/' + str(frame_num).zfill(3) + "_" + 'range' + ".bmp"
    save_path_intensity = save_path + '/' + str(frame_num).zfill(3) + "_" + 'intensity' + ".bmp"
    save_path_elongation = save_path + '/' + str(frame_num).zfill(3) + "_" + 'elongation' + ".bmp"
    # save image (0~255 gray scale images)  ---> opencv는 저장하려면 grayscale은 255로 normalization 해야함
    save_range_image = spherical_img[0,...]*255.0
    save_intensity_image = spherical_img[1,...]*255.0
    save_elongation_image = spherical_img[2,...]*255.0
    cv2.imwrite(save_path_range, save_range_image)      # png로 저장하면 8bit로 변환되어서 저장됨
    cv2.imwrite(save_path_intensity, save_intensity_image)
    cv2.imwrite(save_path_elongation, save_elongation_image)

    # text Save path
    save_path_range_ = save_path + '/' + str(frame_num).zfill(3) + "_" + 'range' + ".txt"
    save_path_intensity_ = save_path + '/' + str(frame_num).zfill(3) + "_" + 'intensity' + ".txt"
    save_path_elongation_ = save_path + '/' + str(frame_num).zfill(3) + "_" + 'elongation' + ".txt"
    # save text
    np.savetxt(save_path_range_, spherical_img[0,...])      # (64,2650)
    np.savetxt(save_path_intensity_, spherical_img[1,...])  # scale (0~1)
    np.savetxt(save_path_elongation_, spherical_img[2,...])



#################################################################################################
# Visualize Pointcloud and Save
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

def save_lidar_points(points, save_path, frame_num):
    txt_path = save_path + "/" + str(frame_num).zfill(3) + '_lidar' + ".txt"
    np.savetxt(txt_path, points)

def save_lidar_labels(frame, save_path, frame_num):
    label_path = save_path + "/" + str(frame_num).zfill(3) + '_lidar_label' + ".txt"
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


################################################################
# Save image projection index in Front view image (LiDAR index, proj_width, proj_height)
################################################################
def save_image_index(arr_indx, save_path, frame_num):
    # Save path
    save_path_indx = save_path + '/' + str(frame_num).zfill(3) + ".txt"

    # save text
    np.savetxt(save_path_indx, arr_indx, fmt='%i')      