# waymo_dataset_parser
* Load the waymo dataset from TF record and parsing datset to the text files

## Build Dependencies
* open3D
* opencv-python
* tensorflow

## Running the waymo parser 
* `python3 parser.py`

## Directory configuration
- 0000 (dataset number)
    - training
        - scene_00 (scene number)
            - camera
            - camera_label
            - lidar
            - lidar_label
            - shperical_img
            - image_indx
    - validation
        - scene_00 (scene number)
            - camera
            - camera_label
            - lidar
            - lidar_label
            - shperical_img
            - image_indx

* camera : Front, Front_left, Front_right, Side_left, Side_right (png)
* camera_label : `[Type, Id, center_x, center_y, length, width, det_diff_level, track_diff_level]` (double, txt)
* lidar : `[x, y, z, intensity]` (double. txt)
* lidar_label : `[[Type, center_x, center_y, center_z, length, height, heading, det_diff_level, track_diff_level]]` (double, txt)
* sphercal_img : range, intenisty, elongation spherical coordinate image (float, bmp, txt)
* image_indx : LiDAR points in the front image coordinate `[LiDAR_mask_index, image_width(x), image_height(y)]` (int, txt)