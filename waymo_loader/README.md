# waymo_loader
* Load the waymo dataset from pared dataset 

## Requirement Dependencies
* python3
* open3D
* opencv-python

## Running the waymo loader
`python3 loader.py`
* You have to change the
    - Seperate: training or validation or testing
    - Seq: 0000 ~
    - Scene: scene_00 ~
```
    # File path
    Parsing_Dir = '/root/data/Waymo/parsed'
    Seperate = '/training'
    Seq = '/0000'
    Scene = '/scene_04'
```

## Load dataset
- Camera images: Front, Front Left, Front Right, Side Left, Side Right 
- Range images: Intensity, Elongation, Range
- Projection images: Intensity, Range
- LiDAR points on projection image
