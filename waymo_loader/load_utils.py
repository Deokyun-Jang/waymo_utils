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