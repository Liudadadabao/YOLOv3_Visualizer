import caffe
import numpy as np
import cv2
from utils import *
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser('YOLOv3')

    parser.add_argument('--prototxt', type=str, default='Models/yolo_v3/yolov3.prototxt')
    parser.add_argument('--caffemodel', type=str, default='Models/yolo_v3/yolov3.caffemodel')
    parser.add_argument('--classfile', type=str, default='Models/yolo_v3/coco.names')
    parser.add_argument('--image', type=str, default='Models/yolo_v3/images/family_photo.jpg')
    parser.add_argument('--resolution', type=int, default=416)

    return parser.parse_args()

def main():
    args = parse_args()

    #model = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    caffe.set_mode_cpu()
    caffe.set_device(0)
    model = caffe.Classifier("./Models/yolo_v3/yolov3m.prototxt", "./Models/yolo_v3/yolov3.caffemodel", raw_scale=255.0)

    img_ori = cv2.imread(args.image)
    inp_dim = args.resolution, args.resolution
    img = img_prepare(img_ori, inp_dim)


    model.blobs['data'].data[...] = img
    output = model.forward()
    rects = rects_prepare(output)
    mapping = get_classname_mapping(args.classfile)
    scaling_factor = min(1, args.resolution / img_ori.shape[1])
    COLORS = tuple(map(tuple,(np.uint8(np.random.uniform(0, 255, (80,3))))))

    for pt1, pt2, cls, prob in rects:

        pt1[0] -= (args.resolution - scaling_factor*img_ori.shape[1])/2
        pt2[0] -= (args.resolution - scaling_factor*img_ori.shape[1])/2
        pt1[1] -= (args.resolution - scaling_factor*img_ori.shape[0])/2
        pt2[1] -= (args.resolution - scaling_factor*img_ori.shape[0])/2



        pt1[0] = np.clip(np.int_(pt1[0]/scaling_factor), 0, img_ori.shape[1])
        pt2[0] = np.clip(np.int_(pt2[0]/scaling_factor), 0, img_ori.shape[1])
        pt1[1] = np.clip(np.int_(pt1[1]/scaling_factor), 0, img_ori.shape[1])
        pt2[1] = np.clip(np.int_(pt2[1]/scaling_factor), 0, img_ori.shape[1])



        label = "{}:{:.2f}".format(mapping[cls], prob)
        #color = tuple(map(int, np.uint8(np.random.uniform(0, 255, 3))))
        color = (np.asscalar(COLORS[cls][0]), np.asscalar(COLORS[cls][1]), np.asscalar(COLORS[cls][2]))


        cv2.rectangle(img_ori, (pt1[0],pt1[1]),(pt2[0],pt2[1]), color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
        cv2.rectangle(img_ori, (pt1[0],pt1[1]),(pt2[0],pt2[1]), color, -1)
        cv2.putText(img_ori, label, (pt1[0], t_size[1] + 4 + pt1[1]),cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 0))

    plt.imshow(img_ori, cmap='gray')
    plt.show()
    cv2.imshow(args.image,img_ori)
    cv2.waitKey()

if __name__ == '__main__':
    main()
