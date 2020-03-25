import os
import math
import numpy as np
import itertools
import glob

import cv2
import open3d as o3d

from numpy import linalg as LA



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

# def load_points(lidar_path, frame_num):
#     lidar_path_  = lidar_path + '/' + str(frame_num).zfill(3) + '_lidar.txt'
#     points = np.loadtxt(lidar_path_, delimiter=' ')
#     return points

def load_points(lidar_path, frame_num):
    lidar_path_  = lidar_path + '/' + str(frame_num).zfill(3) + '.bin'
    points = np.fromfile(lidar_path_, dtype=np.float32)
    points = points.reshape((-1, 4))
    return points

def load_lidar_label(lidar_label_path, frame_num):
    lidar_label_path_ = lidar_label_path + '/' + str(frame_num).zfill(3) + '.txt'

    lidar_label_list = read_label_txt(lidar_label_path_)

    return lidar_label_list


#################################################################################################
# Load Spherical Images and Visualize
#################################################################################################
# def load_spherical_image(spherical_img_path, frame_num):
#     range_img_path = spherical_img_path + '/' + str(frame_num).zfill(3) + '_range.txt'
#     intensity_img_path = spherical_img_path + '/' + str(frame_num).zfill(3) + '_intensity.txt'
#     elongation_img_path = spherical_img_path + '/' + str(frame_num).zfill(3) + '_elongation.txt'

#     range_img = np.loadtxt(range_img_path, delimiter=' ')
#     intenisty_img = np.loadtxt(intensity_img_path, delimiter=' ')
#     elongation_img = np.loadtxt(elongation_img_path, delimiter=' ')

#     return range_img, intenisty_img, elongation_img

def load_spherical_image(spherical_img_path, frame_num):
    range_img_path = spherical_img_path + '/' + str(frame_num).zfill(3) + '_range.bin'
    intensity_img_path = spherical_img_path + '/' + str(frame_num).zfill(3) + '_intensity.bin'
    elongation_img_path = spherical_img_path + '/' + str(frame_num).zfill(3) + '_elongation.bin'

    range_img = np.fromfile(range_img_path, dtype=np.float32)
    intenisty_img = np.fromfile(intensity_img_path, dtype=np.float32)
    elongation_img = np.fromfile(elongation_img_path, dtype=np.float32)
    range_img = range_img.reshape((-1, 2650))       #(64,2650)
    intenisty_img = intenisty_img.reshape((-1, 2650))
    elongation_img = elongation_img.reshape((-1, 2650))

    return range_img, intenisty_img, elongation_img


def show_spherical_image(spherical_name, spherical_img):
    cv2.imshow(spherical_name, spherical_img)



#################################################################################################
# Create Projection Images and Visualize
#################################################################################################
def load_proj_indx(proj_indx_path, frame_num):
    '''
        [mask_indx, proj_width, proj_height]
    '''
    proj_indx_path_ = proj_indx_path + '/' + str(frame_num).zfill(3) + '.txt'
    proj_indx = np.loadtxt(proj_indx_path_, delimiter=' ')

    return proj_indx

# def show_points_on_image(projected_points, camera_image):
#     img = tf.image.decode_jpeg(camera_image.image).numpy()
#     img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)     # RGB --> BGR (opencv)

#     for point in projected_points:
#         # point[0] : width, col
#         # point[1] : height, row
#         cv2.circle(img_cv, (point[0], point[1]), 1, color=(0,0,255), thickness=-1)
    
#     cv2.imshow('points_on_image', img_cv)

def create_points_on_image(proj_indx, camera_image):
    img_cv = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)     # RGB --> BGR (opencv)

    for indx in proj_indx:
        # indxx[0] : mask index of lidar points in front camera coordinate  
        # indxx[1] : width, col
        # indxx[2] : height, row
        cv2.circle(img_cv, (int(indx[1]), int(indx[2])), 1, color=(0,0,255), thickness=-1)

    return img_cv
    


def create_proj_img(proj_indx, points, camera_image):
    img_height, img_width, _ = camera_image.shape

    # range_img = np.zeros((img_height,img_width), dtype='uint8')
    # intensity_img = np.zeros((img_height,img_width), dtype='uint8')
    # height_img = np.zeros((img_height,img_width), dtype='uint8')
    proj_range_img = np.zeros((img_height,img_width), dtype='float32')
    proj_intensity_img = np.zeros((img_height,img_width), dtype='float32')
    # proj_height_img = np.zeros((img_height,img_width), dtype='float32')

    for indxx in proj_indx:
        # indxx[0] : mask index of lidar points in front camera coordinate  
        # indxx[1] : width, col
        # indxx[2] : height, row
        
        lidar_mask = int(indxx[0])
        indx_width = int(indxx[1])
        indx_height = int(indxx[2])
        
        lidar_xyz = [points[lidar_mask,0], points[lidar_mask,1], points[lidar_mask,2]]
        lidar_range = LA.norm(lidar_xyz)
        lidar_intensity = points[lidar_mask ,3]
        

        proj_range_img[indx_height, indx_width] = lidar_range
        proj_intensity_img[indx_height, indx_width] = lidar_intensity
        # proj_height_img[int(point[1]), int(point[0])] = point[4]*255

    return proj_range_img, proj_intensity_img


def show_proj_image(proj_name, proj_img):
    cv2.imshow(proj_name, proj_img)