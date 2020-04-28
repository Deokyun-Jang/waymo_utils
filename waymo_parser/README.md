# waymo_parser
* Load the waymo dataset from TF record and parsing datset to the text files

## Requirement Dependencies
* python3
* open3D
* opencv-python
* tensorflow

## Running the waymo parser 
`python3 parser.py`
* You have to change the path of tf record files
```
    # Load tf record files path
    Data_Dir = '/root/data/Waymo/original'
    Seperate = '/training'      # /training, or /validation

    # parsing path
    Parsing_Dir = '/root/data/Waymo/parsed'
```

## Directory configuration of parsed Waymo dataset 
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

## Data Specifications
### Camera images 
* Front, Front_left, Front_right, Side_left, Side_right (png)
* camera_label : `[Type, Id, center_x, center_y, length, width, det_diff_level, track_diff_level]` (txt)
* format : `float32`, `string(Id)`
### LiDAR data
* lidar : `[x, y, z, intensity]` (bin)
* lidar_label : `[Type, Id, center_x, center_y, center_z, length, width, height, heading, det_diff_level, track_diff_level]` (txt)
* format : `float32`
### LiDAR spherical images
* sphercal_img : range, intenisty, elongation spherical coordinate image (bin)
* range  
    - Normalize (0, 75m) to (0, 1)
* All of images were normalized in [0,1] --> [0,1] 범위 밖의 데이터는 -1로 채워넣음
* format : `float32`
### LiDAR Camera projection mask & index
* image_indx : LiDAR points in the front image coordinate `[LiDAR_mask_index, image_width(x), image_height(y)]` (txt)
    - LiDAR_make_index : LiDAR points masking in Front camera coord.
    - image_width, height : LiDAR index in Front camera coord.
* format : `int`

