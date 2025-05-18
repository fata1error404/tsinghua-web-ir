import json
import logging
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import (
    cosine_similarity,
)  # only used for bert_fine_tuned if you want
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertTokenizer,
    DistilBertModel,
    pipeline,
)
from contextlib import asynccontextmanager
from typing import List

# ─── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── globals ────────────────────────────────────────────────────────────────
bert_fine_tuned: pipeline = None
emoji_data: list = []
embed_tokenizer: DistilBertTokenizer = None
embed_model: DistilBertModel = None
emoji_names: list = []
emb_matrix: np.ndarray = None


def get_sentence_embedding(sentence: str) -> np.ndarray:
    inputs = embed_tokenizer(
        sentence, return_tensors="pt", truncation=True, padding=True
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = embed_model(**inputs).last_hidden_state.mean(dim=1)
    emb = out.cpu().numpy().squeeze()
    return emb


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bert_fine_tuned, emoji_data
    global embed_tokenizer, embed_model, emoji_names, emb_matrix

    # 1) Load emoji data
    with open("backend/emoji.json", "r", encoding="utf-8") as f:
        emoji_data = json.load(f)
    logger.info(f"✅ Loaded emoji.json ({len(emoji_data)} entries)")

    # 2) Load fine-tuned emoji classifier
    rob_tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-emoji"
    )
    rob_model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-emoji"
    )
    bert_fine_tuned = pipeline(
        "text-classification",
        model=rob_model,
        tokenizer=rob_tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    logger.info("✅ Loaded twitter-roberta-base-emoji classifier")

    # 3) Load & speed-optimize DistilBERT embedder
    embed_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    embed_model = DistilBertModel.from_pretrained("distilbert-base-uncased").eval()
    if torch.cuda.is_available():
        embed_model = embed_model.half().to("cuda")
        logger.info("✅ Embed model on CUDA (FP16)")
    else:
        # dynamic quantization on CPU
        embed_model = torch.quantization.quantize_dynamic(
            embed_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logger.info("✅ Embed model quantized on CPU")

    # 4) Precompute & normalize all emoji name embeddings
    emoji_names = [e["name"] for e in emoji_data]
    embs = []
    for name in emoji_names:
        emb = get_sentence_embedding(name)
        embs.append(emb)
    emb_matrix = np.stack(embs)
    emb_matrix /= np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    logger.info("✅ Precomputed & normalized emoji embeddings matrix")

    yield
    logger.info("🛑 Shutting down")


# ─── FastAPI setup ────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)


class TextRequest(BaseModel):
    text: str


@app.post("/infer/bert_fine_tuned")
async def infer_emoji(req: TextRequest):
    results = bert_fine_tuned(req.text)
    top = max(results, key=lambda x: x["score"])
    return {"emoji": top["label"], "scores": results}


STOPWORDS = {
    "face",
    "with",
    "and",
    "but",
    "the",
    "of",
    "a",
    "an",
    "to",
    "for",
    "in",
    "on",
    "from",
}


@app.post("/infer/bert_base")
async def infer_text(req: TextRequest):
    # 1) Compute & normalize input embedding
    sent_emb = get_sentence_embedding(req.text)
    sent_emb /= np.linalg.norm(sent_emb)

    # 2) Vectorized cosine similarity to find best emoji name
    sims = emb_matrix @ sent_emb
    idx = int(np.argmax(sims))
    best_name = emoji_names[idx]
    best_char = next(e["char"] for e in emoji_data if e["name"] == best_name)

    # 3) Split the full name into candidate words (filter stopwords & punctuation)
    candidates: List[str] = [
        w.strip(" ,-’'\"").lower()
        for w in best_name.split()
        if w.lower() not in STOPWORDS
    ]
    if not candidates:
        short_name = best_name.replace(" ", "_")
    else:
        # 4) Score each candidate word to pick the “most relevant” one
        best_score = -1.0
        short_name = candidates[0]
        for word in candidates:
            # embed the word
            word_emb = get_sentence_embedding(word)
            word_emb /= np.linalg.norm(word_emb)
            score = float(np.dot(sent_emb, word_emb))  # cosine on normalized vectors
            if score > best_score:
                best_score = score
                short_name = word

    print("Input: " + req.text + " | Output emoji: " + best_char)

    # 5) Return both emoji and the single-word tag
    return {"emoji": best_char, "name": short_name}
