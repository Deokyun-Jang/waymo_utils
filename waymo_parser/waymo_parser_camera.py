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



def save_camera_image(camera_image, frame_num, save_path):
    """Save camera images."""
    
    img = tf.image.decode_jpeg(camera_image.image).numpy()      # load image
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)     # RGB --> BGR (opencv)

    img_name = str( open_dataset.CameraName.Name.Name(camera_image.name) )
    
    save_path_ = save_path + "/camera/" + str(frame_num).zfill(6) + "_" + img_name + ".png"     # save path

    cv2.imwrite(save_path_, img_cv)


def save_labels(camera_image, frame, frame_num, save_path):
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
    label_path = save_path + "/camera_label/" + str(frame_num).zfill(6) + '_' + img_name + ".txt"
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
            # visualize and save images
            show_camera_image_cv(image, frame)      

            # save images
            save_camera_image(image, frame_num, save_path)

            # save labels
            save_labels(image, frame, frame_num, save_path)

        if cv2.waitKey(0) & 0xFF == 27:
            break
        elif cv2.waitKey(0) & 0xFF == ord('n'):
            continue


