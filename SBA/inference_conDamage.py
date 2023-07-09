from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2 
import glob
import mmdet
import numpy as np
import os
import json
from tqdm import tqdm


config_file = "/home/user/mmdetection/work_dirs/23.03.23_fasterrcnn_1333x800/final.py"
checkpoint_file = "/home/user/mmdetection/work_dirs/23.03.23_fasterrcnn_1333x800/epoch_12.pth"

device = "cuda:1" # or "cpu"
model = init_detector(config_file, checkpoint_file, device)

classes =["crack", "spall"]
colors = [[0,0,255], [0, 255,0]]

input_dir = "/home/user/infer_trigger/input"
input_dir = os.path.join(input_dir, "*", "*.*")
input_dir = glob.glob(input_dir)
save_dir = '/home/user/infer_trigger/output/infer_conDamage'

os.makedirs(save_dir, exist_ok =True)

color_map = dict(zip(classes, colors))

def show_result_with_color(img, result, color_map):
    pred_img = img.copy()
    
    
            
    if len(result) > 0:
        for i in range(len(result)):
            if len(result[i]) > 0:
                # 0. crack, 1. spalling
                bboxes = result[i]
                
                for _bbox in bboxes:
                    bbox = [int(x) for x in _bbox[:4]] 
                    color = color_map[classes[i]]
                    if _bbox[4] > 0.50:
                        cv2.rectangle(pred_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        score = round(float(_bbox[4]), 3)
                        text = f"{classes[i]}: {score}"
                        text_position = (bbox[0], bbox[1]-10)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        font_color = (255,255,255)
                        thickness = 1
                        line_type = cv2.LINE_AA
                        bottomLeftOrigin = False
                        cv2.putText(pred_img, text, text_position, font, font_scale, font_color, thickness, line_type, bottomLeftOrigin)

             

    return pred_img

all_images_data = []

for idx, filename in enumerate(tqdm(input_dir, desc="Saving conDamage_output.json")):
    img = cv2.imread(filename)
    result = inference_detector(model, img)
    img_name = os.path.basename(filename)
    pred_img = show_result_with_color(img, result, color_map)
    save_filename = os.path.join(save_dir, img_name)
    cv2.imwrite(save_filename, pred_img)
    
    image_data = {
        'image_number': idx,
        'file_name': img_name,
        'predictions': []
    }
    
    if len(result) > 0:
        for i in range(len(result)):
            if len(result[i]) > 0:
                bboxes = result[i]
                for _bbox in bboxes:
                    bbox = [int(x) for x in _bbox[:4]]
                    score = round(float(_bbox[4]), 3)
                    class_name = classes[i]
                    prediction = {
                        'class': class_name,
                        'bbox': bbox,
                        'score': score
                    }
                    image_data['predictions'].append(prediction)
    
    all_images_data.append(image_data)

json_filename = os.path.join(save_dir, 'conDamage_output.json')
with open(json_filename, 'w') as f:
    json.dump(all_images_data, f)