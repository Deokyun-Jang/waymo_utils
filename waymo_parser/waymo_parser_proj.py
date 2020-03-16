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


def show_camera_image_cv(camera_image):
    """Show a camera image and the given camera labels."""

    img = tf.image.decode_jpeg(camera_image.image).numpy()
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)     # RGB --> BGR (opencv)

    img_name = str( open_dataset.CameraName.Name.Name(camera_image.name) )
    
    # Show the camera image.
    cv2.imshow(img_name, img_cv)




#################################################################################################
# Visualize Spherical Images
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

    spherical_img = np.array([np_range_image_range, np_range_image_intensity, np_range_image_elongation])
    
    spherical_view = cv2.vconcat([np_range_image_range, np_range_image_intensity, np_range_image_elongation])
    cv2.imshow('spherical images', spherical_view)

    return spherical_img


    
def save_spherical_image(spherical_img, frame_num, save_path):
    """Save camera images."""
    # save image (0~255 gray scale images)  ---> opencv는 저장하려면 grayscale은 255로 normalization 해야함
    save_range_image = spherical_img[0,...]*255.0
    save_intensity_image = spherical_img[1,...]*255.0
    save_elongation_image = spherical_img[2,...]*255.0

    save_path_range = save_path + "/spherical/" + str(frame_num).zfill(6) + "_" + 'range' + ".bmp"
    save_path_intensity = save_path + "/spherical/" + str(frame_num).zfill(6) + "_" + 'intensity' + ".bmp"
    save_path_elongation = save_path + "/spherical/" + str(frame_num).zfill(6) + "_" + 'elongation' + ".bmp"

    cv2.imwrite(save_path_range, save_range_image)      # png로 저장하면 8bit로 변환되어서 저장됨
    cv2.imwrite(save_path_intensity, save_intensity_image)
    cv2.imwrite(save_path_elongation, save_elongation_image)
    

    


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

def show_projection_image(projected_points, camera_image):
    img = tf.image.decode_jpeg(camera_image.image).numpy()
    img_height, img_width, _ = img.shape

    range_img = np.zeros((img_height,img_width), dtype='uint8')
    intensity_img = np.zeros((img_height,img_width), dtype='uint8')
    # height_img = np.zeros((img_height,img_width), dtype='uint8')

    # range_img = np.zeros((img_height,img_width), dtype='float32')
    # intensity_img = np.zeros((img_height,img_width), dtype='float32')

    for point in projected_points:
        # point[0] : width, col
        # point[1] : height, row
        
        range_img[int(point[1]), int(point[0])] = point[2]*255
        intensity_img[int(point[1]), int(point[0])] = point[3]*255
        # height_img[int(point[1]), int(point[0])] = point[4]*255
        
        # histogram equalization (균일화)
        hist_range_img = cv2.equalizeHist(range_img) 
        hist_intenisty_img = cv2.equalizeHist(intensity_img) 
        
    # cv2.imshow('range', range_img)
    # cv2.imshow('intensity', intensity_img)

    cv2.imshow('range', hist_range_img)
    cv2.imshow('intensity', hist_intenisty_img)



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


        # '''
        #     Visualize Camera Images 
        # '''
        # for index, image in enumerate(frame.images):
        #     if index == 0:
        #         show_camera_image_cv(image)      # visualize and save images
            

        '''
            Visualize Spherical image (Range, Intensity, elongation) & Save
        '''
        frame.lasers.sort(key=lambda laser: laser.name)
        spherical_img = show_range_image(get_range_image(range_images, open_dataset.LaserName.TOP, 0), frame_num, save_path)
        #show_range_image(get_range_image(range_images, open_dataset.LaserName.TOP, 1), frame_num, save_path)     # second return 
        save_spherical_image(spherical_img, frame_num, save_path)

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
        
        
        '''
            Visualize Camera Projection
        '''
        images = sorted(frame.images, key=lambda i:i.name)

        ####################### ?
        cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
        cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)
        ####################### ?

        # The distance between lidar points and vehicle frame origin.
        points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)     # range (x,y,z --> norm)
        # intensity
        np_instensity = np.transpose(np.array([xyzI_points_all[...,3]]))
        intensity_all_tensor = tf.convert_to_tensor(np_instensity, dtype=tf.float32)   
        # height 
        np_height = np.transpose(np.array([xyzI_points_all[...,2]]))
        height_all_tensor = tf.convert_to_tensor(np_height, dtype=tf.float32)    

        # image index
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        # masking points in image fov
        mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

        cp_points_all_tensor = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
        points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))
        intensity_all_tensor = tf.gather_nd(intensity_all_tensor, tf.where(mask))
        height_all_tensor = tf.gather_nd(height_all_tensor, tf.where(mask))
        

        # tensor to numpy
        np_image_indx = cp_points_all_tensor[..., 1:3].numpy()
        np_range = points_all_tensor.numpy()
        np_intensity = intensity_all_tensor.numpy()
        np_height = height_all_tensor.numpy()

        # normalization (0~1)
        np_range[np_range>75.0] = 75.0    # range : set 0.0 if value is bigger than 75m
        np_range /= 75.0
        np_intensity[np_intensity>1.0] = 0.0    # intensity : set 0.0 if value is bigger than 1.0
        np_height = (np_height + 1.27)/4  # normalize (height: -1.27~2.73m)
        np_height[np_height>4] = 1.0
        np_height[np_height<0] = 0.0

        # projected points
        projected_points_all_from_raw_data = tf.concat([cp_points_all_tensor[..., 1:3], points_all_tensor, intensity_all_tensor], axis=-1).numpy()
        np_projected_points_all = np.concatenate([np_image_indx, np_range, np_intensity], axis=-1)
        # np_projected_points_all = np.concatenate([np_image_indx, np_range, np_intensity, np_height], axis=-1)


        # check
        print('projected points : ', np_projected_points_all.shape)  # (17210,4) --> width, height, range, intensity

        # Visualize
        show_points_on_image(projected_points_all_from_raw_data, images[0])

        show_projection_image(np_projected_points_all, images[0])


        if cv2.waitKey(0) & 0xFF == 27:
            break
        elif cv2.waitKey(0) & 0xFF == ord('n'):
            continue


