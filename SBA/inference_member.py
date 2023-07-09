from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2 
import glob
import mmdet
import numpy as np

config_file = '/home/user/mmdetection/work_dirs/23.04.12_slfa_fasterrcnn_1333x800/final.py'
checkpoint_file = '/home/user/mmdetection/work_dirs/23.04.12_slfa_fasterrcnn_1333x800/latest.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

classes =["Sleeper", "fastener"]
colors = [[0,0,255], [0, 255,0]]

filenames = glob.glob('/home/user/230412_slfa/test/*.*')

color_map = dict(zip(classes, colors))

def show_result_with_color(img, result, color_map):
    pred_img = img.copy()
    if len(result) > 0:
        for i in range(len(result)):
            if len(result[i]) > 0:
                bboxes = result[i]
                for _bbox in bboxes:
                    bbox = [int(x) for x in _bbox[:4]] 
                    color = color_map[classes[i]]
                    if _bbox[4] > 0.70:
                        cv2.rectangle(pred_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        text = classes[i]
                        text_position = (bbox[0], bbox[1]-10)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        font_color = (255,255,255)
                        thickness = 1
                        line_type = cv2.LINE_AA
                        bottomLeftOrigin = False
                        cv2.putText(pred_img, text, text_position, font, font_scale, font_color, thickness, line_type, bottomLeftOrigin)
    return pred_img
for filename in filenames:
    img = cv2.imread(filename)
    img = cv2.resize(img, (960, 540))
    result = inference_detector(model, img) 
    pred_img = show_result_with_color(img, result, color_map)
    # pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    save_filename = '/home/user/infer_trigger/memberdetection' + filename.split('/')[-1]
    cv2.imwrite(save_filename, pred_img)
                            

