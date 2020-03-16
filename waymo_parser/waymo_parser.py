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


#################################################################################################
# Visualize Camera Images and Camera Labels
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

def show_camera_image_cv(camera_image, frame, frame_num, save_path):
    """Show a camera image and the given camera labels."""

    img = tf.image.decode_jpeg(camera_image.image).numpy()
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)     # RGB --> BGR (opencv)
    img_cv_labels = img_cv.copy()
    
    # Draw the camera labels.
    for camera_labels in frame.camera_labels:
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

    # save images
    save_camera_image(img_cv, img_name, frame_num, save_path)


def save_camera_image(camera_image, img_name, frame_num, save_path):
    """Save camera images."""
    save_path_ = save_path + "/camera/" + str(frame_num).zfill(6) + "_" + img_name + ".png"
    cv2.imwrite(save_path_, camera_image)



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
    np_range_image_range[np_range_image_range>75.0] = 75.0    # range : set 0.0 if value is bigger than 75m
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
def show_points_with_label(points, frame):
    # Show Point cloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)

    # label
    for laser_label in frame.laser_labels:

        print(laser_label.box.center_x)
        print(laser_label.box.length)
        print(laser_label.box.heading)


    # print(np.asarray(pcd.points))      # change pcd to numpy array
    o3d.visualization.draw_geometries([o3d_pcd])      # draw pcd

def save_points(points):
    txt_path = save_path + "/lidar/" + str(frame_num).zfill(6) + '_lidar' + ".txt"
    np.savetxt(txt_path, points)


#################################################################################################
# Visualize Projection Images
#################################################################################################
def show_points_on_image(projected_points, camera_image):
    img = tf.image.decode_jpeg(camera_image.image).numpy()
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)     # RGB --> BGR (opencv)

    for point in projected_points:
        # point[0] : width, col
        # point[1] : height, row
        cv2.circle(img_cv, (point[0], point[1]), 1, color=(0,0,255), thickness=-1)
    
    cv2.imshow('points_on_image', img_cv)




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
            Visualize Camera Images and Camera Labels
        '''
        for index, image in enumerate(frame.images):
            show_camera_image_cv(image, frame, frame_num, save_path)      # visualize and save images


        '''
            Visualize Range Images
        '''
        frame.lasers.sort(key=lambda laser: laser.name)
        show_range_image(get_range_image(range_images, open_dataset.LaserName.TOP, 0), frame_num, save_path)
        #show_range_image(get_range_image(range_images, open_dataset.LaserName.TOP, 1), frame_num, save_path)     # second return 


        '''
            Point Cloud Conversion and Visualization
        '''
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
        print('points : ', points_all.shape)
        print('xyzI points : ', xyzI_points_all.shape)
        
        # visualization & Save
        show_points_with_label(points_all, frame)
        save_points(xyzI_points_all)
        
        
        '''
            Visualize Camera Projection
        '''
        images = sorted(frame.images, key=lambda i:i.name)
        cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
        cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

        # The distance between lidar points and vehicle frame origin.
        points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

        cp_points_all_tensor = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
        points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

        projected_points_all_from_raw_data = tf.concat([cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()
        
        # check
        print('projected points : ', projected_points_all_from_raw_data.shape)

        # Visualize
        show_points_on_image(projected_points_all_from_raw_data, images[0])


        if cv2.waitKey(0) & 0xFF == 27:
            break
        elif cv2.waitKey(0) & 0xFF == ord('n'):
            continue


