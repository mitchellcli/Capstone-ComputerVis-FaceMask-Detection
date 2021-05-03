import os
import cv2
from base_camera import BaseCamera
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
from utils.datasets import *
#from utils.utils import *
import pandas as pd
from collections import defaultdict
import platform
import shutil

import time
from pathlib import Path
import torch.backends.cudnn as cudnn
from numpy import random

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def Overlap(mask_coord, person_coord, img_width, img_height):

  min_x_1, min_y_1, max_x_1, max_y_1 = mask_coord
  min_x_0, min_y_0, max_x_0, max_y_0 = person_coord


  overlap_minx = max(min_x_1,min_x_0)
  overlap_miny = max(min_y_1,min_y_0)
  overlap_maxx = min(max_x_1,max_x_0)
  overlap_maxy = min(max_y_1,max_y_0)

  if (overlap_maxx > overlap_minx) and (overlap_maxy > overlap_miny):
    overlap = (overlap_maxx-overlap_minx) * (overlap_maxy - overlap_miny)
  else:
    overlap = 0
  
  mask_area = (max_x_1-min_x_1) * (max_y_1 - min_y_1)
  fraction = overlap/mask_area
  return(fraction)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

class Camera(BaseCamera):
    video_source = 'uploads/Mask.mp4'

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        out, weights1,weights2, imgsz = \
        'result/','weights/yolov5x.pt', 'weights/best.pt', 416
        source = 'uploads/Mask.mp4'

        save_txt = True
        txt_path = 'content/outputs.txt'
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file('deep_sort_pytorch/configs/deep_sort.yaml')
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        #Initialize
        device = select_device()
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        # Half precision
        # half = False and device.type != 'cpu'
        half = True and device.type != 'cpu'
        print('half = ' + str(half))

        # Load deepsort model
        model = torch.load(weights1, map_location=device)['model'].float()  # load to FP32
        model.to(device).eval()
        if half:
            model.half()  # to FP16
        
        dataset = LoadImages(source, img_size=imgsz)
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        # run once
        _ = model(img.half() if half else img) if device.type != 'cpu' else None

        #save_path = str(Path(out))
        #txt_path = str(Path(out)) + '/results.txt'

        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, 0.6, 0.3, classes=0, agnostic=False)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s
                s += '%gx%g ' % img.shape[2:]  # print string

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    bbox_xywh = []
                    confs = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, im0)
                    #print(outputs)
                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        draw_boxes(im0, bbox_xyxy, identities)
                    #print(det)
                    # Write MOT compliant results to file
                    if save_txt and len(outputs) != 0:
                        print('inside savetxt')
                        print(f'{s}Done. ({t2 - t1:.3f}s)')
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

                else:
                    deepsort.increment_ages()

        # Load yolo model
        model = attempt_load(weights2, map_location=device)
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if half:
            model.half()
        
        #model.to(device).float().eval()

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            #modelc.to(device).float().eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        dataset = LoadImages(source, img_size=imgsz)
        #dataset = LoadStreams(source, img_size=imgsz)
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        history = defaultdict(list)
        DSOutput = pd.read_csv('content/outputs.txt', sep = ' ', header = None)

        for frameNumber, (path, img, im0s, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, 0.3, 0.3, classes=None, agnostic=False)
            t2 = time_synchronized()
            
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                save_path = 'result/Mask.mp4'
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if len(det):
                    num_cat = 6

                    classes = (det[:,-1].cpu().numpy()).astype(int)
                    one_hot_cats = np.eye(num_cat)[classes].reshape(-1, num_cat)

                    counts_per_cat = one_hot_cats.sum(axis=0)
                    #print("Countspercat ", counts_per_cat)
                    score = round(counts_per_cat[[1,3,5]].sum() / len(det),3)

                    weighted_counts_per_cat = one_hot_cats.T @ np.asarray(det[:,-2].cpu())
                    WeightedCompliance = weighted_counts_per_cat[[1,3,5]].sum() / weighted_counts_per_cat.sum()
                    
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                    person_coords = DSOutput[DSOutput.iloc[:,0]==frameNumber].values.reshape(-1,11)

                    CurrentFrameDetection = -1*np.zeros(len(det))

                    if(len(person_coords != 0)):
                        for itemp,mask_coord in enumerate(det):
                        
                        # overlaps = [Overlap(mask_coord[:4], person_coord, img.shape[2], img.shape[3]) for person_coord in person_coords[:,2:6]]
                            overlaps = [Overlap(mask_coord[:4].cpu(), person_coord, 10000, 10000) for person_coord in person_coords[:,2:6]]

                            best_overlap = np.argmax(overlaps)
                            best_person = person_coords[best_overlap,1]
                            history[best_person].append(mask_coord[-1].cpu().item())
                            CurrentFrameDetection[itemp] = best_person

                    #for c in det[:, -1].unique():  #probably error with torch 1.5
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        
                    CurrentFrameDetection = list(reversed(CurrentFrameDetection))
                    
                    for mask, (*xyxy, conf, cls) in enumerate(reversed(det)):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, score, label=label, color=colors[int(cls)], personid=CurrentFrameDetection[mask], line_thickness=3)
                print(f'{s}Done. ({t2 - t1:.3f}s)')


                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)
 
            yield cv2.imencode('.jpg', im0)[1].tobytes()

        compliance = 0
        total=0
        txt_result_path = 'result/result.txt'
        for k,v in history.items():
            # 1,3,5 are full
            # 2,4 are partial
            # 0 no
            good_frames = sum(np.array(v)%2==1)
            bad_frames = sum(np.array(v)%2==0)
            if len(v) > 4:
                total += 1
                if good_frames >= bad_frames:
                    compliance +=1
                    print('Person {} is compliant'.format(k))
                    with open(txt_result_path, 'a') as f:
                            f.write('Person {} is compliant \n'.format(k))
                else:
                    print('Person {} is not compliant'.format(k))
                    with open(txt_result_path, 'a') as f:
                            f.write('Person {} is not compliant \n'.format(k))
        Overall_Compliance = round(compliance/total,3)
        with open(txt_result_path, 'a') as f:
                            f.write('Overall compliance:' + str(Overall_Compliance))
        print('Overall compliance:', Overall_Compliance)
        return(Overall_Compliance)
            
        print(f'Done. ({time.time() - t0:.3f}s)')         