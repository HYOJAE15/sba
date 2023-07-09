from mmdet.apis import init_detector, inference_detector
from mmcls.apis import init_model, inference_model

import cv2 
import glob
import numpy as np
import os
import json
from tqdm import tqdm

config_file_class = "/home/user/mmclassification/work_dirs/2023.04.11_fastner_Resnet50/final.py"
checkpoint_file_class = "/home/user/mmclassification/work_dirs/2023.04.11_fastner_Resnet50/epoch_100.pth"

device = "cuda" # or "cpu"

model_class = init_model(config_file_class, checkpoint_file_class, device)

json_data_list = []

input_dir = '/home/user/infer_trigger/output/infer_member/crop'

output_dir = '/home/user/infer_trigger/output/infer_fastener'

input_dir_img = os.path.join(input_dir, "*.*")
img_list = glob.glob(input_dir_img)

for crop_img in tqdm(img_list, desc="Classifying fastener damage"):
    
    # print(crop_img)
    img = cv2.imread(crop_img)

    result = inference_model(model_class, img)
    
    pred_class_idx = result['pred_label']
    pred_class = model_class.CLASSES[pred_class_idx]
    pred_score = result['pred_score']    

    json_data = {
        'file_name': os.path.basename(crop_img),
        'pred_class': pred_class ,
        'pred_score': round(float(pred_score), 3)        
    }

    json_data_list.append(json_data)    

    
os.makedirs(output_dir, exist_ok=True)

json_filename = os.path.join(output_dir, 'fastener_output.json')
with open(json_filename, 'w') as f:
    json.dump(json_data_list, f)
