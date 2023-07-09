from PIL import Image
import cv2
import os
import mmcv
import glob
from tqdm import tqdm
from mmcls.apis import init_model,inference_model,show_result_pyplot

device = 'cuda:0'
config_file = '/home/user/mmclassification/work_dirs/2023.03.02_Resnet50/final.py'

checkpoint_file = '/home/user/mmclassification/work_dirs/2023.03.02_Resnet50/epoch_100.pth'

model = init_model(config_file, checkpoint_file, device=device)

# define the ROI rail pixel area
left_ch3 = 920
right_ch3 = 1300

left_ch4 = 600
right_ch4 = 980

top = 0
bottom = 1080

input_dir_ch3 = '/home/user/infer_trigger/input/3'
output_dir_ch3 = '/home/user/infer_trigger/output/railDamage_ch3'

input_dir_ch4 = '/home/user/infer_trigger/input/4'
output_dir_ch4 = '/home/user/infer_trigger/output/railDamage_ch4'

for filename in os.listdir(input_dir_ch3):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_dir_ch3, filename)
        image = Image.open(image_path)
 
        cropped_image = image.crop((left_ch3, top, right_ch3, bottom))
        
        os.makedirs(output_dir_ch3, exist_ok=True)

        output_path = os.path.join(output_dir_ch3, filename.replace('.jpg', '.png'))
        cropped_image.save(output_path)

for filename in os.listdir(input_dir_ch4):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_dir_ch4, filename)
        image = Image.open(image_path)

        cropped_image = image.crop((left_ch4, top, right_ch4, bottom))

        os.makedirs(output_dir_ch4, exist_ok=True)
        output_path = os.path.join(output_dir_ch4, filename.replace('.jpg', '.png'))
        cropped_image.save(output_path)


infer_output_dir = '/home/user/infer_trigger/output/infer_railDamage_ch3'


filenames = glob.glob(os.path.join(output_dir_ch3, "*.png"))

for filename in tqdm(filenames, desc="Processing ch3 railDamage"):
    img = cv2.imread(filename)

        
    result = inference_model(model, img)
    if hasattr(model, 'module'):
        model = model.module
    
    pred_class_idx = result['pred_label']
    pred_class = model.CLASSES[pred_class_idx]

    class_dir = os.path.join(infer_output_dir, pred_class)
    os.makedirs(class_dir, exist_ok=True)

    save_filename = os.path.join(class_dir, os.path.basename(filename))
    cv2.imwrite(save_filename, model.show_result(img, result, show=False))

    # pred_img = model.show_result(img, result, show=False) 
    # save_filename = '/home/user/230323_rail_crop/ch3_cropped_predicted/' + filename.split('/')[-1]
    # cv2.imwrite(save_filename, pred_img)


infer_output_dir_ch4 = '/home/user/infer_trigger/output/infer_railDamage_ch4'


filenames = glob.glob(os.path.join(output_dir_ch4, "*.png"))

for filename in tqdm(filenames, desc="Processing ch4 railDamage"):
    img = cv2.imread(filename)

    result = inference_model(model, img)
    if hasattr(model, 'module'):
        model = model.module
    
    pred_class_idx = result['pred_label']
    pred_class = model.CLASSES[pred_class_idx]

    class_dir = os.path.join(infer_output_dir_ch4, pred_class)
    os.makedirs(class_dir, exist_ok=True)

    save_filename = os.path.join(class_dir, os.path.basename(filename))
    cv2.imwrite(save_filename, model.show_result(img, result, show=False))

