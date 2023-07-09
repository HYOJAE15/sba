from PIL import Image
import cv2
import os
import mmcv
import glob
import json
from tqdm import tqdm
from mmcls.apis import init_model, inference_model, show_result_pyplot

device = 'cuda'
config_file = '/home/user/mmclassification/work_dirs/2023.03.02_Resnet50/final.py'
checkpoint_file = '/home/user/mmclassification/work_dirs/2023.03.02_Resnet50/epoch_100.pth'
model = init_model(config_file, checkpoint_file, device=device)

# Define the ROI rail pixel area
left_ch3 = 920
right_ch3 = 1300

left_ch4 = 600
right_ch4 = 980

top = 0
bottom = 1080

input_dir_ch3 = '/home/user/infer_trigger/input/3'

input_dir_ch4 = '/home/user/infer_trigger/input/4'

infer_output_dir = '/home/user/infer_trigger/output/infer_railDamage'

# Process channel 3 images
for filename in tqdm(os.listdir(input_dir_ch3), desc="Cropping ch3"):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_dir_ch3, filename)
        image = cv2.imread(image_path)

        cropped_image = image[top:bottom, left_ch3:right_ch3]

        os.makedirs(infer_output_dir, exist_ok=True)

        output_path = os.path.join(infer_output_dir, filename.replace('.jpg', '.png'))
        cv2.imwrite(output_path, cropped_image)

# Process channel 4 images
for filename in tqdm(os.listdir(input_dir_ch4), desc="Cropping ch4"):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_dir_ch4, filename)
        image = cv2.imread(image_path)

        cropped_image = image[top:bottom, left_ch4:right_ch4]

        output_path = os.path.join(infer_output_dir, filename.replace('.jpg', '.png'))
        cv2.imwrite(output_path, cropped_image)

# # Process channel 3 images
# for filename in tqdm(os.listdir(input_dir_ch3), desc="Cropping ch3"):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         image_path = os.path.join(input_dir_ch3, filename)
#         # image = Image.open(image_path)
#         image = cv2.imread(image_path)

#         cropped_image = image.crop((left_ch3, top, right_ch3, bottom))
        
#         os.makedirs(infer_output_dir, exist_ok=True)

#         output_path = os.path.join(infer_output_dir, filename.replace('.jpg', '.png'))
#         # cropped_image.save(output_path)
#         cv2.imwrite(cropped_image, output_path)

# # Process channel 4 images
# for filename in tqdm(os.listdir(input_dir_ch4), desc="Cropping ch4"):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         image_path = os.path.join(input_dir_ch4, filename)
#         # image = Image.open(image_path)
#         image = cv2.imread(image_path)
#         cropped_image = image.crop((left_ch4, top, right_ch4, bottom))

#         output_path = os.path.join(infer_output_dir, filename.replace('.jpg', '.png'))
#         # cropped_image.save(output_path)
#         cv2.imwrite(cropped_image, output_path)



json_data_list = []

filenames = glob.glob(os.path.join(infer_output_dir, "*.png"))

for filename in tqdm(filenames, desc="Processing railDamage"):
    img = cv2.imread(filename)
        
    result = inference_model(model, img)

    pred_class_idx = result['pred_label']
    pred_class = model.CLASSES[pred_class_idx]
    pred_score = result['pred_score']

    
    json_data = {
        'file_name': os.path.basename(filename),
        'pred_class': pred_class,
        'pred_score': round(float(pred_score), 3),        
    }
    json_data_list.append(json_data)


json_filename = '/home/user/infer_trigger/output/railDamage_output.json'
with open(json_filename, 'w') as f:
    json.dump(json_data_list, f)