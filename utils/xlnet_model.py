import os
import torch
from torch import nn
from transformers import XLNetModel, XLNetTokenizer
from huggingface_hub import hf_hub_download

# Set Hugging Face cache directory
os.environ["HF_HOME"] = "/tmp/huggingface"

# Download trained model weights from Hugging Face Hub
MODEL_PATH = hf_hub_download(
    repo_id="yeswanthvarma/xlnet-evaluator-model",
    filename="xlnet_answer_assessment_model.pt"
)

# Define your trained model architecture
class XLNetAnswerAssessmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained("xlnet-base-cased")
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask=None):
        pooled = self.xlnet(input_ids, attention_mask).last_hidden_state.mean(dim=1)
        x = torch.relu(self.fc1(pooled))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.output(x))  # Output: score in range [0, 1]

# Load tokenizer and model
xlnet_available = False
try:
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    model = XLNetAnswerAssessmentModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    xlnet_available = True
    print("✅ Custom XLNet model loaded.")
except Exception as e:
    print("⚠️  Could not load XLNet model → fallback will be used\n", e)

# -------------------------------
# Main prediction function
# -------------------------------
def get_model_prediction(q, s, r):
    if not xlnet_available:
        raise RuntimeError("XLNet model not available")

    # Combine input text as during training
    combined = f"{q} [SEP] {s} [SEP] {r}"
    inputs = tokenizer(combined, return_tensors="pt", truncation=True, max_length=512, padding=True)

    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        score = output.squeeze().item() * 100  # Convert from [0,1] → [0,100]

    return round(score)

# Optional: Fallback similarity using word overlap
def fallback_similarity(t1, t2):
    w1, w2 = set(t1.lower().split()), set(t2.lower().split())
    return round(len(w1 & w2) / len(w1 | w2) * 100) if w1 and w2 else 0

# Final score API (use in app.py)
def get_similarity_score(q, s, r):
    try:
        return get_model_prediction(q, s, r)
    except Exception as e:
        print("❌ XLNet failed, using fallback:", e)
        return fallback_similarity(s, r)
