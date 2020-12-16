# Grasp Stability

## Dependencies

Install [scipyplot](https://github.com/robertocalandra/scipyplot), [deepdish](https://github.com/uchicago-cs/deepdish), [pytorch](https://pytorch.org/), [torchvision](https://pytorch.org/docs/stable/torchvision/index.html).
```
pip install scipyplot deepdish torch torchvision   
```

## Content

1) grasp_data_collection.py: collect data with different grasp configuration; save data with vision, touch and label (success/failure).
2) robot.py: helper class for controlling the robot.
3) train.py: learning grasp stability from vision and touch.
4) draw.py: plot accuracy with different input modality (vision/touch/both) and different amount of data.

## Usage

Collect grasp dataset. Data saved in ./data folder. Each file contains 100 samples.
```
python grasp_data_collection.py
```

Learning grasp stability with different amount of data (N x 100 samples). Logs saved in ./logs folder.
```
python train.py -N 10
```

Plot test accuracy with different input modalities and different amount of data, loading results from ./logs folder.
```
python draw.py
```