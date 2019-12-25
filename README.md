# CNN Visualizer

This is a CNN-based object detection network visualizer using PyQt and Caffe.

The work is based on [Deep Visualization Toolbox](https://github.com/yosinski/deep-visualization-toolbox).
The toolbox and methods are described casually [here](http://yosinski.com/deepvis) and more formally in this paper:
 * Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. [Understanding neural networks through deep visualization](http://arxiv.org/abs/1506.06579). Presented at the Deep Learning Workshop, International Conference on Machine Learning (ICML), 2015.

The code of [the improved version](https://github.com/x1155665/CNN-Visualizer) by Rui Zhang were also used. 

## Features
Main features of the Deep Visualization Toolbox that are implemented in the CNN Visualizer:
 * Object detection result
 ![image](https://github.com/Liudadadabao/YOLOv3_Visualizer/blob/master/YOLOvis_results.png)
 * Activation of each neuron
 * Images that cause high activation (These images are pre-computed by the scripts in _./find_maxes/_)
 * Channels that output high activation to a certain object
 * Table of detected objects and their confidence
 ![image](https://github.com/Liudadadabao/YOLOv3_Visualizer/blob/master/YOLO_demo.png)
 
Missing feature:
 * Synthetic images that cause high activation
 * Backprop/Deconvolution/Guided backprop
 
 
## Supported network architectures
In the very beginning, this tool was intended for YOLOv3. It can support YOLO-like object detection algorithms.
 
## Setting up
### 1. Work under anaconda environment.
You may need to install [Anaconda3](https://anaconda.com)

### 2. Install prerequisites by creating a conda environment 
The prerequisites for this  demonstrater are caffe, opencv3 and pyqt5, which can be installed as follows(other install options exist as well)
This Visualizer could be ran under CPU or GPU mode. The environment have been saved to text files. So the environment can be create by running one of the following command:
```
conda env create --file cpucaffe.txt
conda env create --file gpucaffe.txt

```

### 3. Download and configure the CNN visualizer


Change the general settings in _main_settings.yaml_. 

Use _model_setting_template.yaml_ as template to create settings file for the new model and then register this file in _main_settings.yaml_ 

Compute the top images (optional):
You may first modify or mannually set the size of top images under _find_maxes/misc.py_ function _convert_region_dag_ to save the computing time and then run:
```
python find_maxes/find_max_act.py --model model_name
python find_maxes/crop_max_patches.py --model model_name
```

Run the tool by:
```
python NN_Vis_Demo.py
```



