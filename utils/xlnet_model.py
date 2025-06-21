import os, requests, tqdm, torch, numpy as np
from torch import nn
from transformers import XLNetModel, XLNetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# Hugging‑Face model download
HF_URL   = "https://huggingface.co/yeswanthvarma/xlnet-evaluator-model/resolve/main/xlnet_answer_assessment_model.pt"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xlnet_answer_assessment_model.pt")

def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        return
    print("▶️  Downloading XLNet weights from Hugging Face …")
    with requests.get(HF_URL, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(MODEL_PATH, "wb") as f, tqdm.tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    print("✅  Download complete.")

download_model_if_needed()
# ------------------------------------------------------------------

xlnet_available = False   # will flip to True if load succeeds

class XLNetAnswerAssessmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained("xlnet-base-cased")
        hidden = 768
        self.fc1 = nn.Linear(hidden, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask=None):
        pooled = self.xlnet(input_ids, attention_mask).last_hidden_state.mean(1)
        x = torch.relu(self.fc1(pooled))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))

try:
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    model = XLNetAnswerAssessmentModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    xlnet_available = True
    print("✅ Custom XLNet model loaded.")
except Exception as e:
    print("⚠️  Could not load XLNet model → fallback to TF‑IDF\n", e)

# ------------------------------------------------------------------
# scoring helpers (unchanged)
# ------------------------------------------------------------------
embedding_cache = {}

def get_model_prediction(q, s, r):
    if not xlnet_available:
        raise ValueError("XLNet unavailable")
    combined = f"{q} [SEP] {s} [SEP] {r}"
    inputs = tokenizer(combined, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        score = float(model(**inputs).squeeze()) * 100
    return round(score)

def tfidf_similarity(t1, t2):
    vec = TfidfVectorizer()
    mat = vec.fit_transform([t1, t2])
    return round(cosine_similarity(mat[0], mat[1])[0][0] * 100)

def fallback_similarity(t1, t2):
    w1, w2 = set(t1.lower().split()), set(t2.lower().split())
    return round(len(w1 & w2) / len(w1 | w2) * 100) if w1 and w2 else 0

def get_similarity_score(q, s, r):
    try:
        return get_model_prediction(q, s, r) if xlnet_available else tfidf_similarity(s, r)
    except Exception:
        return fallback_similarity(s, r)
