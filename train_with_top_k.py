import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from collections import Counter
from okvqa_dataset import OKVQADataset
import os
from tqdm import tqdm

# Config
image_dir = r""
questions_path = r""
annotations_path = r""
top_k_answers = 1000
batch_size = 2
num_epochs = 30
allowed_types = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
val_image_dir = r""
val_questions_path = r""
val_annotations_path = r""

# Load dataset
dataset = OKVQADataset(image_dir, questions_path, annotations_path, transform=None, allowed_question_types=allowed_types)
val_dataset = OKVQADataset(val_image_dir, val_questions_path, val_annotations_path, transform=None, allowed_question_types=allowed_types)

# Top-K answers
all_answers = [ans for sample in dataset for ans in sample["answers"]]
answer_counts = Counter(all_answers)
most_common_answers = [a for a, _ in answer_counts.most_common(top_k_answers)]
answer_to_idx = {ans: i for i, ans in enumerate(most_common_answers)}
idx_to_answer = {i: ans for ans, i in answer_to_idx.items()}
print(f"Top-{top_k_answers} answers used. Total: {len(answer_to_idx)}")

# Helper functions
def get_answer_idx(answers):
    for ans in answers:
        if ans in answer_to_idx:
            return answer_to_idx[ans]
    return None

def custom_collate(batch):
    return {
        "question_id": [item["question_id"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "question": [item["question"] for item in batch],
        "answers": [item["answers"] for item in batch],
        "image": [item["image"] for item in batch],
    }

# Filter and Load
filtered_samples = [sample for sample in dataset if get_answer_idx(sample["answers"]) is not None]
val_filtered = [sample for sample in val_dataset if get_answer_idx(sample["answers"]) is not None]

loader = DataLoader(filtered_samples, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_filtered, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

# CLIP Model and VQA Head
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval().to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

for name, param in clip_model.text_model.named_parameters():
    if any(f"encoder.layers.{i}." in name for i in range(8, 12)):
        param.requires_grad = True
    else:
        param.requires_grad = False

class VQAModel(nn.Module):
    def __init__(self, hidden_dim=512, num_answers=len(answer_to_idx)):
        super().__init__()
        self.text_proj = nn.Linear(512, hidden_dim)
        self.image_proj = nn.Linear(512, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, num_answers)
        )

    def forward(self, image_feat, text_feat):
        i = self.image_proj(image_feat)
        t = self.text_proj(text_feat)
        x = torch.cat([i, t], dim=1)
        return self.classifier(x)

model = VQAModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Training + Evaluation
best_val_score = -float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for i, batch in enumerate(progress_bar):
        images = batch["image"]
        questions = batch["question"]
        labels = [get_answer_idx(ans) for ans in batch["answers"]]
        valid_indices = [i for i, label in enumerate(labels) if label is not None]

        if len(valid_indices) == 0:
            continue

        images = [images[i] for i in valid_indices]
        questions = [questions[i] for i in valid_indices]
        labels = torch.tensor([labels[i] for i in valid_indices], device=device)

        inputs = processor(text=questions, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = clip_model(**inputs)
        img_feats = outputs.image_embeds
        txt_feats = outputs.text_embeds

        logits = model(img_feats, txt_feats)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

    # Validation with Top-K Evaluation
    model.eval()
    total_score = 0
    total_samples = 0
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"]
            questions = batch["question"]
            answers = batch["answers"]

            inputs = processor(text=questions, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = clip_model(**inputs)
            img_feats = outputs.image_embeds
            txt_feats = outputs.text_embeds

            logits = model(img_feats, txt_feats)
            preds = torch.argmax(logits, dim=1)
            val_loss += loss_fn(logits, torch.tensor([
                get_answer_idx(ans) for ans in answers
            ], device=device)).item()

            for i in range(len(preds)):
                pred_ans = idx_to_answer[preds[i].item()]
                gt_ans_list = answers[i]
                match_count = sum(pred_ans == a for a in gt_ans_list)
                score = min(match_count / 3, 1.0)
                total_score += score
                total_samples += 1

    avg_val_score = total_score / total_samples
    val_loss /= len(val_loader)

    print(f"Validation - Loss: {val_loss:.4f} | VQA Accuracy: {avg_val_score:.4f}")

    # Save best model
    if avg_val_score > best_val_score:
        best_val_score = avg_val_score
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "vqamodel_best.pth"))
        print(f"Saved best model with val score {avg_val_score:.4f}")
