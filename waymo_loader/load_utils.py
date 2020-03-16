import os
import math
import numpy as np
import itertools
import glob

import cv2
import open3d as o3d

def read_label_txt(path):
    label_list = []
    with open(path) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            label_list.append(inner_list)
    
    return label_list


#################################################################################################
# Visualize Camera Images and Camera Labels
#################################################################################################

def load_camera_image(img_path, frame_num):
    '''
        Load front, front left, front right, side left, side right images
    '''
    path_front = img_path + '/' + str(frame_num).zfill(3) + '_FRONT.png'
    path_front_left = img_path + '/' + str(frame_num).zfill(3) + '_FRONT_LEFT.png'
    path_front_right = img_path + '/' + str(frame_num).zfill(3) + '_FRONT_RIGHT.png'
    path_side_left = img_path + '/' + str(frame_num).zfill(3) + '_SIDE_LEFT.png'
    path_side_right = img_path + '/' + str(frame_num).zfill(3) + '_SIDE_RIGHT.png'

    img_front = cv2.imread(path_front, cv2.IMREAD_COLOR)
    img_front_cv = cv2.cvtColor(np.array(img_front), cv2.COLOR_RGB2BGR)
    img_front_left = cv2.imread(path_front_left, cv2.IMREAD_COLOR)
    img_front_left_cv = cv2.cvtColor(np.array(img_front_left), cv2.COLOR_RGB2BGR)
    img_front_right = cv2.imread(path_front_right, cv2.IMREAD_COLOR)
    img_front_right_cv = cv2.cvtColor(np.array(img_front_right), cv2.COLOR_RGB2BGR)
    img_side_left = cv2.imread(path_side_left, cv2.IMREAD_COLOR)
    img_side_left_cv = cv2.cvtColor(np.array(img_side_left), cv2.COLOR_RGB2BGR)
    img_side_right = cv2.imread(path_side_right, cv2.IMREAD_COLOR)
    img_side_right_cv = cv2.cvtColor(np.array(img_side_right), cv2.COLOR_RGB2BGR)

    return img_front_cv, img_front_left_cv, img_front_right_cv, img_side_left_cv, img_side_right_cv




def load_camera_label(img_label_path, frame_num):
    path_label_front = img_label_path + '/' + str(frame_num).zfill(3) + '_FRONT.txt'
    path_label_front_left = img_label_path + '/' + str(frame_num).zfill(3) + '_FRONT_LEFT.txt'
    path_label_front_right = img_label_path + '/' + str(frame_num).zfill(3) + '_FRONT_RIGHT.txt'
    path_label_side_left = img_label_path + '/' + str(frame_num).zfill(3) + '_SIDE_LEFT.txt'
    path_label_side_right = img_label_path + '/' + str(frame_num).zfill(3) + '_SIDE_RIGHT.txt'

    label_front = read_label_txt(path_label_front)
    label_front_left = read_label_txt(path_label_front_left)
    label_front_right = read_label_txt(path_label_front_right)
    label_side_left = read_label_txt(path_label_side_left)
    label_side_right = read_label_txt(path_label_side_right)

    return label_front, label_front_left, label_front_right, label_side_left, label_side_right


'''
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
cyan = (255, 255, 0)
magenta = (255, 0, 255)
'''
def get_corners(xylw):
    corners = np.zeros((4, 2), dtype=np.float32)
    '''
    1----4
    |    |
    |    |
    2----3
    '''

    # 1
    corners[0, 0] = xylw[0] - 0.5 * xylw[2]
    corners[0, 1] = xylw[1] - 0.5 * xylw[3]

    # 2
    corners[1, 0] = xylw[0] - 0.5 * xylw[2]
    corners[1, 1] = xylw[1] + 0.5 * xylw[3]

    # 3
    corners[2, 0] = xylw[0] + 0.5 * xylw[2]
    corners[2, 1] = xylw[1] + 0.5 * xylw[3]

    # 4
    corners[3, 0] = xylw[0] + 0.5 * xylw[2]
    corners[3, 1] = xylw[1] - 0.5 * xylw[3]

    return corners


def show_camera_image_with_label(camera_name, camera_img, camera_labels):
    """Show a camera image and the given camera labels."""

    img_cv = cv2.cvtColor(camera_img, cv2.COLOR_RGB2BGR)     # RGB --> BGR (opencv)
    
    # Draw the camera labels.
    for label in camera_labels:       # Front, Front right, Front left, Side right, Side left

        xylw = list(map(float, label[2:6]))

        # get corner points (img-coord.)
        img_corners = get_corners(xylw).reshape(-1, 1, 2).astype(int)

        # Draw the object bounding box.
        cv2.polylines(img_cv, [img_corners], True, (0, 0, 255), 1)
    
    # Show the camera image.
    cv2.imshow(camera_name, img_cv)


#################################################################################################
# Visualize Pointcloud and Labels
#################################################################################################
def get_bbox(xyzlwh_y):
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
    
    c_x = xyzlwh_y[0]    # center x
    c_y = xyzlwh_y[1]
    c_z = xyzlwh_y[2]
    L = xyzlwh_y[3]        # length
    W = xyzlwh_y[4]
    H = xyzlwh_y[5]
    yaw = xyzlwh_y[6]     # yaw (heading)
    

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


def show_points_with_label(points, lidar_labels):
    draw_list = []

    # Show Point cloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points[...,:3])
    draw_list.append(o3d_pcd)

    # draw bounding box
    for lidar_label in lidar_labels:

        xyzlwh_yaw = list(map(float, lidar_label[2:9]))

        line_set = get_bbox(xyzlwh_yaw)

        Type     = lidar_label[0]

        draw_list.append(line_set)

    # print(np.asarray(pcd.points))      # change pcd to numpy array
    o3d.visualization.draw_geometries(draw_list)      # draw pcd

def load_points(lidar_path, frame_num):
    lidar_path_  = lidar_path + '/' + str(frame_num).zfill(3) + '_lidar.txt'
    points = np.loadtxt(lidar_path_, delimiter=' ')
    return points

def load_lidar_label(lidar_label_path, frame_num):
    lidar_label_path_ = lidar_label_path + '/' + str(frame_num).zfill(3) + '_lidar_label.txt'

    lidar_label_list = read_label_txt(lidar_label_path_)

    return lidar_label_list




#################################################################################################
# Create Projection Images and Visualize
#################################################################################################
