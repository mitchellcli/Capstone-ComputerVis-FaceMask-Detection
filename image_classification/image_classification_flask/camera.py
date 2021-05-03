import os
import cv2 as cv
from base_camera import BaseCamera
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
import tensorflow as tf
from math import ceil as r
from model import MobileNetV3_6c as MobileNetV3

import time
from pathlib import Path
import torch.backends.cudnn as cudnn
from numpy import random

# face detection model
from face.anchor_generator import generate_anchors
from face.anchor_decode import decode_bbox
from face.nms import single_class_non_max_suppression
from face.pytorch_loader import load_pytorch_model, pytorch_inference


# set imgage classification model
model_path = 'model/140_6c_blur_weight.h5'
net = MobileNetV3.build_mobilenet()
net.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
net.build((1,64,64,3))
net.load_weights(model_path)


# set face detection model
model = load_pytorch_model('face/model360.pth')
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
anchors_exp = np.expand_dims(anchors, axis=0)
conf_thresh=0.5
iou_thresh=0.4
target_shape=(360, 360)


labels = {
    0:'No mask',
    1:'Non medical mask full',
    2:'Non medical mask partial',
    3:'Medical mask full',
    4:'Medical mask partial',
    5:'Shield'

    }
color_dict={
    0:(255,0,255),
    1:(255,0,0),
    2:(255,255,0),
    3:(0,255,0),
    4:(0,255,255),
    5:(0,0,255)
    }

video_source = 'uploads/Mask.mp4'

class Camera(BaseCamera):
    video_source = 'uploads/Mask.mp4'

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = video_source

    @staticmethod
    def frames():
        vid_path, vid_writer = None, None
        cap = cv.VideoCapture(video_source)

        while True:
            success, frame = cap.read()

            if success:
                #width, height, _ = frame.shape
                # if the video is too big uncomment the below code
                #frame = resize(frame, height, width)

                #padding the image to avoid the bounding going out of the image
                #and crashes the program
                image =  cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                #converting numpy array into image
                #image = Image.fromarray(padding)
                height, width, _ = image.shape

                image_resized = cv.resize(image, target_shape)
                image_np = image_resized / 255.0  # 归一化到0~1
                image_exp = np.expand_dims(image_np, axis=0)

                image_transposed = image_exp.transpose((0, 3, 1, 2))

                y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
                # remove the batch dimension, for batch is always 1 for inference.
                y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
                y_cls = y_cls_output[0]
                # To speed up, do single class NMS, not multiple classes NMS.
                bbox_max_scores = np.max(y_cls, axis=1)
                bbox_max_score_classes = np.argmax(y_cls, axis=1)

                # keep_idx is the alive bounding box after nms.
                keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                            bbox_max_scores,
                                                            conf_thresh=conf_thresh,
                                                            iou_thresh=iou_thresh,
                                                            )
                people_all = 0
                mask_detected = 0

                for idx in keep_idxs:
                    people_all += 1
                    conf = float(bbox_max_scores[idx])
                    class_id = bbox_max_score_classes[idx]
                    bbox = y_bboxes[idx]
                    # clip the coordinate, avoid the value exceed the image boundary.
                    x1 = max(0, int(bbox[0] * width))
                    y1 = max(0, int(bbox[1] * height))
                    x2 = min(int(bbox[2] * width), width)
                    y2 = min(int(bbox[3] * height), height)
                    image_test = image[y1:y2 ,x1:x2, 0:3]
                    #image_list.append(image_test)

                    if np.min(np.shape(image_test))<1:
                            continue

                    if image.max() <= 1.0:
                        resized = tf.image.resize_with_pad(image_test,64,64,)
                    else:
                        resized = tf.image.resize_with_pad(image_test/255.0,64,64,)

                    test_images = np.zeros((1,64,64,3), dtype = float)

                    test_images[0] = resized.numpy()
                    pred_labels = net.predict(test_images)
                    pred = np.argmax(pred_labels, axis=1)

                    if pred[0]==1 or pred[0]==3 or pred[0]==5:
                        mask_detected += 1

                    scale = round((y2-y1)*35/100)

                    cv.rectangle(frame, (x1,y1), (x2,y2),color_dict[pred[0]],2)
                    cv.putText(frame,labels[pred[0]], 
                                (x1,y1-5),cv.FONT_HERSHEY_SIMPLEX,
                                                        2.0,color_dict[pred[0]],2)

                if people_all:
                    cv.putText(frame, "Compliance rate = %.2f %%" % (mask_detected/people_all*100), 
                    (5, round(height/20)), cv.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)


                save_path = 'result/Mask.mp4'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = cap.get(cv.CAP_PROP_FPS)
                    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(frame)
        
                yield cv.imencode('.jpg', frame)[1].tobytes()

            else:
                print('End')
                break    