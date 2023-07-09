import json
from shapely.geometry import box
from rtree import index
from tqdm import tqdm

with open('/home/user/infer_trigger/output/infer_member/member_output.json', 'r') as f:
    member_data = json.load(f)

with open('/home/user/infer_trigger/output/infer_conDamage/conDamage_output.json', 'r') as f:
    damage_data = json.load(f)

output_data = []
idx = index.Index()

for i, member in enumerate(tqdm(member_data, desc="Building json index")):
    for pred in member['predictions']:
        if pred['score'] >= 0.7:
            idx.insert(i, pred['bbox'])

for damage in tqdm(damage_data, desc="Categorizing damage data"):
    for prediction in damage['predictions']:
        if prediction['score'] >= 0.7:
            pred_bbox = box(*prediction['bbox'])
            intersecting_indices = list(idx.intersection(prediction['bbox']))

            for i in intersecting_indices:
                member_pred = member_data[i]['predictions'][0]
                mem_bbox = box(*member_pred['bbox'])

                if pred_bbox.within(mem_bbox):
                    new_prediction = prediction.copy()
                    new_prediction['class'] = 'Sleeper ' + prediction['class']
                    damage_copy = damage.copy()
                    damage_copy['predictions'] = [new_prediction]
                    output_data.append(damage_copy)

                elif pred_bbox.intersects(mem_bbox):
                    intersection = pred_bbox.intersection(mem_bbox)
                    output_data.append({
                        'image_number': damage['image_number'],
                        'file_name': damage['file_name'],
                        'predictions': [
                            {'class': 'Sleeper ' + prediction['class'], 'bbox': list(intersection.bounds), 'score': prediction['score']},
                            {'class': 'Track ' + prediction['class'], 'bbox': list(pred_bbox.difference(mem_bbox).bounds), 'score': prediction['score']}
                        ]
                    })

for damage in tqdm(damage_data, desc="Categorizing damage data"):
    for pred in damage['predictions']:
        if 'Sleeper' not in pred['class'] and pred['score'] >= 0.5:
            output_data.append({'image_number': damage['image_number'], 'file_name': damage['file_name'], 'predictions': [{'class': 'Track ' + pred['class'], 'bbox': pred['bbox'], 'score': pred['score']}]})

with open('/home/user/infer_trigger/output/intersected_output.json', 'w') as f:
    json.dump(output_data, f)