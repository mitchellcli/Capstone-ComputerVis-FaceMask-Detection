import cv2 as cv
import tensorflow as tf
import numpy as np
import argparse
from math import ceil as r
# imgage classification model
from model import MobileNetV3

# face detection model
from face.anchor_generator import generate_anchors
from face.anchor_decode import decode_bbox
from face.nms import single_class_non_max_suppression
from face.pytorch_loader import load_pytorch_model, pytorch_inference

# set imgage classification model
model_path = 'model/four_class_newdata.h5'
net = MobileNetV3.build_mobilenet()
net.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
net.build((1,64,64,3))
net.load_weights(model_path)


# set face detection model
model = load_pytorch_model('face/model360.pth');
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
anchors_exp = np.expand_dims(anchors, axis=0)
conf_thresh=0.5
iou_thresh=0.4
target_shape=(360, 360)


labels = {
    0:'no mask',
    1:'mask full',
    2:'mask partial',
    3:'shield'
    '''
    1:'non medical mask full',
    2:'non medical mask partial',
    3:'medical mask full',
    4:'medical mask partial',
    5:'shield'
    '''
    }
color_dict={
    0:(255,0,255),
    1:(255,0,0),
    2:(255,255,0),
    3:(0,255,0),
    4:(0,255,255),
    5:(0,0,255)
    }

def run_on_video(video_path, output_video_name):
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv.VideoWriter(
        output_video_name, cv.VideoWriter_fourcc('F','L','V','1'), fps, (w, h))
    
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

            for idx in keep_idxs:
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

                scale = round((y2-y1)*35/100)

                cv.rectangle(frame, (x1,y1), (x2,y2),color_dict[pred[0]],2)
                cv.putText(frame,labels[pred[0]], 
                            (x1,y1-5),cv.FONT_HERSHEY_SIMPLEX,
                                                    scale*0.04,color_dict[pred[0]],2)

            vid_writer.write(frame)
            cv.imshow('image',frame)
            if cv.waitKey(1)  == ord('q'):
                break


        else:
            print('End')
            break


    cap.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection on Video")
    parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    args = parser.parse_args()
    video_path = args.video_path
    if args.video_path == '0':
        video_path = 0
    run_on_video(video_path, './output.flv')