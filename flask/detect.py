import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from numpy import random
from collections import defaultdict

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
  source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
  webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
      ('rtsp://', 'rtmp://', 'http://'))

  print('reading in Deep Sort Readings')
  DSOutput = pd.read_csv('/content/outputs.txt', sep = ' ', header = None)
  
  print('Successfully read in Deep Sort Reading')
  print(DSOutput)
  
  history = defaultdict(list)
  
  # Directories
  save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
  (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

  # Initialize
  set_logging()
  device = select_device(opt.device)
  half = device.type != 'cpu'  # half precision only supported on CUDA

  # Load model
  model = attempt_load(weights, map_location=device)  # load FP32 model
  stride = int(model.stride.max())  # model stride
  imgsz = check_img_size(imgsz, s=stride)  # check img_size
  # print(imgsz)
  if half:
      model.half()  # to FP16

  # Second-stage classifier
  classify = False
  if classify:
      modelc = load_classifier(name='resnet101', n=2)  # initialize
      modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

  # Set Dataloader
  vid_path, vid_writer = None, None
  if webcam:
      view_img = check_imshow()
      cudnn.benchmark = True  # set True to speed up constant image size inference
      dataset = LoadStreams(source, img_size=imgsz, stride=stride)
  else:
      save_img = True
      dataset = LoadImages(source, img_size=imgsz, stride=stride)

  # Get names and colors
  names = model.module.names if hasattr(model, 'module') else model.names
  colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

  # Run inference
  if device.type != 'cpu':
      model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
  t0 = time.time()
  for frameNumber, (path, img, im0s, vid_cap) in enumerate(dataset):
      img = torch.from_numpy(img).to(device)
      img = img.half() if half else img.float()  # uint8 to fp16/32
      img /= 255.0  # 0 - 255 to 0.0 - 1.0
      if img.ndimension() == 3:
          img = img.unsqueeze(0)

      # Inference
      t1 = time_synchronized()
      pred = model(img, augment=opt.augment)[0]
     

      # Apply NMS
      pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
      t2 = time_synchronized()

      # Apply Classifier
      if classify:
          pred = apply_classifier(pred, modelc, img, im0s)
      print('pred shape', len(pred))
      # Process detections
      
      for i, det in enumerate(pred):  # detections per image
          print('det shape', det.shape)
          if webcam:  # batch_size >= 1
              p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
          else:
              p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

          p = Path(p)  # to Path
          save_path = str(save_dir / p.name)  # img.jpg
          txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
          s += '%gx%g ' % img.shape[2:]  # print string
          gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
          if len(det):
              num_cat = 6

              classes = (det[:,-1].cpu().numpy()).astype(int)
              one_hot_cats = np.eye(num_cat)[classes].reshape(-1, num_cat)

              counts_per_cat = one_hot_cats.sum(axis=0)
              #print("Countspercat ", counts_per_cat)
              score = counts_per_cat[[1,3,5]].sum() / len(det)
              #ConfidenceMetric = 

              weighted_counts_per_cat = one_hot_cats.T @ np.asarray(det[:,-2].cpu())
              WeightedCompliance = weighted_counts_per_cat[[1,3,5]].sum() / weighted_counts_per_cat.sum()
              
              # print(score)
              # print(WeightedCompliance)
              
              # Rescale boxes from img_size to im0 size
              det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
              # print(det)
              
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
              #print(frameNumber , history)
              # Print results
              #0 - no mask 
              #1 - non medical full 
              #2 - non medical partial 
              #3 - medical full 
              #4 medical partial 
              #5 face shield
              for c in det[:, -1].unique():
                  n = (det[:, -1] == c).sum()  # detections per class
                  # print("n: " , n)
                  # print("c: " , c)
                  s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

              # Write results
              #save_txt = True
              CurrentFrameDetection = list(reversed(CurrentFrameDetection))
              for mask, (*xyxy, conf, cls) in enumerate(reversed(det)): 
                  if save_txt:  # Write to file
                      xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                      # print(xywh)
                      line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                      with open(txt_path + '.txt', 'a') as f:
                          f.write(('%g ' * len(line)).rstrip() % line + '\n')

                  if save_img or view_img:  # Add bbox to image
                      label = f'{names[int(cls)]} {conf:.2f}'
                      plot_one_box(xyxy, im0, score, label=label, color=colors[int(cls)], personid=CurrentFrameDetection[mask], line_thickness=3)

          # Print time (inference + NMS)
          print(f'{s}Done. ({t2 - t1:.3f}s)')
          # Stream results
          if view_img:
              cv2.imshow(str(p), im0)
              cv2.waitKey(1)  # 1 millisecond

          # Save results (image with detections)
          if save_img:
              if dataset.mode == 'image':
                  cv2.imwrite(save_path, im0)
              else:  # 'video'
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


          
  if save_txt or save_img:
    
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    print(f"Results saved to {save_dir}{s}")

  compliance = 0
  total=0
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
      else:
        print('Person {} is not compliant'.format(k))
  print('overall compliance', compliance/total)
    
  print(f'Done. ({time.time() - t0:.3f}s)')

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



  # mask_coord = mask_coord.numpy().astype(int)
  # person_coord = person_coord.astype(int)
  # mask_mask = np.zeros((img_width,img_height))
  # person_mask = np.zeros((img_width,img_height))
  # # print("Mask Coord", mask_coord)
  # # print("person coord", person_coord)
  # # print(img_width, img_height)
  # mask_mask[mask_coord[0]:mask_coord[2], mask_coord[1]:mask_coord[3]] = 1
  # person_mask[person_coord[0]:person_coord[2], person_coord[1]:person_coord[3]] = 1

  # area_of_overlap = np.sum(mask_mask.astype(bool) & person_mask.astype(bool))
  # # print("Area of overlap", area_of_overlap)
  # # print(np.sum(mask_mask))
  # fraction = area_of_overlap/ np.sum(mask_mask)
  return(fraction)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()




