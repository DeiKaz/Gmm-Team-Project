from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import json
import os
import torch
from tqdm import tqdm

questions_path = r""
annotations_path = r""
images_folder = r""
output_json_path = "predictions.json"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) 
print(f"Running on: {device}")

with open(questions_path, "r") as f:
    questions_data = json.load(f)

with open(annotations_path, "r") as f:
    annotations_data = json.load(f)

annotations_dict = {ann['question_id']: ann for ann in annotations_data['annotations']}

def get_image_path(image_id: int) -> str:
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    return os.path.join(images_folder, filename)

correct_top1 = 0
correct_top5 = 0
total = 0
results = []

for question in tqdm(questions_data['questions'], desc="Processing questions"):
    image_id = question['image_id']
    question_text = question['question']
    question_id = question['question_id']
    img_path = get_image_path(image_id)

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        continue

    encoding = processor(image, question_text, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits

    top5_indices = torch.topk(logits, k=5, dim=-1).indices[0].tolist()
    top5_answers = [model.config.id2label[i].lower() for i in top5_indices]
    pred_answer = top5_answers[0]  # Top-1

    if question_id not in annotations_dict:
        print(f"No annotation for question id {question_id}")
        continue
    ground_truths = [ans['answer'].lower() for ans in annotations_dict[question_id]['answers']]

    # Evaluate
    top1_score = int(pred_answer in ground_truths)
    top5_score = int(any(ans in ground_truths for ans in top5_answers))

    correct_top1 += top1_score
    correct_top5 += top5_score
    total += 1

    results.append({
        "question_id": question_id,
        "image_id": image_id,
        "question": question_text,
        "top1_answer": pred_answer,
        "top5_answers": top5_answers,
        "ground_truth_answers": ground_truths,
        "top1_score": top1_score,
        "top5_score": top5_score
    })

with open(output_json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} predictions to {output_json_path}")