import os
import json
from torch.utils.data import Dataset
from PIL import Image

class OKVQADataset(Dataset):
    def __init__(self, image_dir, questions_path, annotations_path=None, transform=None, allowed_question_types=None):
        self.image_dir = image_dir
        self.transform = transform
        self.allowed_question_types = allowed_question_types

        if "train" in os.path.basename(image_dir).lower():
            self.image_prefix = "COCO_train2014_"
        elif "val" in os.path.basename(image_dir).lower():
            self.image_prefix = "COCO_val2014_"
        else:
            raise ValueError("image_dir must contain 'train' or 'val' to determine the prefix.")

        with open(questions_path, 'r') as f:
            questions_json = json.load(f)
        questions = questions_json["questions"]

        if annotations_path:
            with open(annotations_path, 'r') as f:
                annotations_json = json.load(f)

            # Filter annotations by allowed question types if provided
            if allowed_question_types:
                filtered_annotations = [
                    ann for ann in annotations_json["annotations"]
                    if ann["question_type"] in allowed_question_types
                ]
            else:
                filtered_annotations = annotations_json["annotations"]

            self.answers = {
                ann["question_id"]: [a["answer"] for a in ann["answers"]]
                for ann in filtered_annotations
            }

            # Filter questions to only those that have valid annotations
            question_ids = set(self.answers.keys())
            self.questions = [q for q in questions if q["question_id"] in question_ids]
        else:
            self.answers = {}
            self.questions = questions


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        question_id = q["question_id"]
        image_id = q["image_id"]
        question = q["question"]

        image_file = f"{self.image_prefix}{image_id:012d}.jpg"
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        answers = self.answers.get(question_id, [])

        return {
            "question_id": question_id,
            "image_id": image_id,
            "question": question,
            "answers": answers,
            "image": image,
        }
