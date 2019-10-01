import cv2
import os
import sys
import re
import numpy as np
import errno
import skimage
import caffe


class LayerRecord:

    def __init__(self, layer_def):

        self.layer_def = layer_def
        self.name = layer_def.name
        self.type = layer_def.type

        # keep filter, stride and pad
        if layer_def.type == 'Convolution':
            self.filter = list((layer_def.convolution_param.kernel_h,layer_def.convolution_param.kernel_w))
            if len(self.filter) == 1:
                self.filter *= 2
            self.pad = list(layer_def.convolution_param.pad)
            if len(self.pad) == 0:
                self.pad = [0, 0]
            elif len(self.pad) == 1:
                self.pad *= 2
            self.stride = list(layer_def.convolution_param.stride)
            if len(self.stride) == 0:
                self.stride = [1, 1]
            elif len(self.stride) == 1:
                self.stride *= 2

        elif layer_def.type == 'Pooling':
            self.filter = [layer_def.pooling_param.kernel_size]
            if len(self.filter) == 1:
                self.filter *= 2
            self.pad = [layer_def.pooling_param.pad]
            if len(self.pad) == 0:
                self.pad = [0, 0]
            elif len(self.pad) == 1:
                self.pad *= 2
            self.stride = [layer_def.pooling_param.stride]
            if len(self.stride) == 0:
                self.stride = [1, 1]
            elif len(self.stride) == 1:
                self.stride *= 2

        else:
            self.filter = [0, 0]
            self.pad = [0, 0]
            self.stride = [1, 1]

        # keep tops
        self.tops = list(layer_def.top)

        # keep bottoms
        self.bottoms = list(layer_def.bottom)

        # list of parent layers
        self.parents = []

        # list of child layers
        self.children = []

    pass


from caffe.proto import caffe_pb2
from google.protobuf import text_format

    # load prototxt file
network_def = caffe_pb2.NetParameter()
#with open('/HOMES/jingru-liu/PycharmProjects/YOLOv3-caffe/Models/caffenet-yos/caffenet-yos-deploy.prototxt.processed_by_deepvis','r') as proto_file:
with open('/HOMES/jingru-liu/PycharmProjects/YOLOv3-caffe/Models/yolo_v3/yolov3m.prototxt.processed_by_deepvis', 'r') as proto_file:
    text_format.Merge(str(proto_file.read()), network_def)

    # map layer name to layer record
layer_name_to_record = dict()
for layer_def in network_def.layer:
    if (len(layer_def.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer_def.include]):
        layer_name_to_record[layer_def.name] = LayerRecord(layer_def)
        print(layer_def.name, layer_name_to_record[layer_def.name].filter)




top_to_layers = dict()
for layer in network_def.layer:
        # no specific phase, or TEST phase is specifically asked for
    if (len(layer.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer.include]):
        for top in layer.top:
            if top not in top_to_layers:
                top_to_layers[top] = list()
            top_to_layers[top].append(layer.name)

    # find parents and children of all layers
for child_layer_name in layer_name_to_record.keys():
    child_layer_def = layer_name_to_record[child_layer_name]
    for bottom in child_layer_def.bottoms[1:]:
        for parent_layer_name in top_to_layers[bottom]:
            if parent_layer_name in layer_name_to_record:
                parent_layer_def = layer_name_to_record[parent_layer_name]
                if parent_layer_def not in child_layer_def.parents:
                    child_layer_def.parents.append(parent_layer_def)
                if child_layer_def not in parent_layer_def.children:
                    parent_layer_def.children.append(child_layer_def)

    # update filter, strid, pad for maxout "structures"
for layer_name in layer_name_to_record.keys():
    layer_def = layer_name_to_record[layer_name]
    if layer_def.type == 'Eltwise' and \
        len(layer_def.parents) == 1 and \
        layer_def.parents[0].type == 'Slice' and \
        len(layer_def.parents[0].parents) == 1 and \
        layer_def.parents[0].parents[0].type in ['Convolution', 'InnerProduct']:
        layer_def.filter = layer_def.parents[0].parents[0].filter
        layer_def.stride = layer_def.parents[0].parents[0].stride
        layer_def.pad = layer_def.parents[0].parents[0].pad


    # keep helper variables in settings
    #settings._network_def = network_def
    #settings._layer_name_to_record = layer_name_to_record



