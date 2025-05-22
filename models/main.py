import json
import logging
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from sentence_transformers import SentenceTransformer

# logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()

# global variables
emoji_data: list = []
emoji_names: list = []
stopwords: list = []
base_model: SentenceTransformer = None
fine_tuned_model: pipeline = None
emb_matrix: np.ndarray = None


def get_sentence_embedding(sentence: str) -> np.ndarray:
    result = base_model.encode(sentence)
    return result


# ─── LLM server startup ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    global emoji_data, emoji_names, stopwords, base_model, fine_tuned_model, emb_matrix

    # load Unicode emoji JSON data and stopwords
    with open("backend/emoji.json", "r", encoding="utf-8") as f:
        emoji_data = json.load(f)

    with open("models/stopwords.txt", encoding="utf-8") as f:
        stopwords = [line.strip().lower() for line in f if line.strip()]

    logger.info(f"✅ Loaded emoji.json ({len(emoji_data)} entries)")

    # load fine-tuned emoji classifier
    fine_tuned_model = pipeline(
        "text-classification",
        model=AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emoji", local_files_only=True
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emoji", local_files_only=True
        ),
    )
    logger.info("✅ RoBERTa fine tuned model loaded")

    # load base sentence embedding model
    base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✅ MiniLM-L6-v2 sentence transformer model loaded")

    # precompute and normalize all emoji name embeddings (embeddings matrix represents all emoji names as dense vectors in the same semantic space, enabling similarity comparisons with future input sentences; emoji name is initially a sentence, like 'smiling face with open hands')
    emoji_names = [e["name"] for e in emoji_data]
    emoji_name_embeddings = []

    for name in emoji_names:
        emb = get_sentence_embedding(name)
        emoji_name_embeddings.append(emb)

    emb_matrix = np.stack(emoji_name_embeddings)
    emb_matrix /= np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    logger.info("✅ Precomputed and normalized emoji name embeddings matrix")

    yield


# ─── FastAPI setup ───
# create a FastAPI app instance with optional startup/shutdown logic
app = FastAPI(lifespan=lifespan)


# define the expected JSON body structure for incoming POST requests
class TextRequest(BaseModel):
    text: str


@app.post("/infer/bert_fine_tuned")
async def infer_emoji(req: TextRequest):
    results = fine_tuned_model(req.text)

    # select and return the emoji prediction with the highest confidence score
    top_emoji = max(results, key=lambda x: x["score"])

    print("")
    print("Input: " + req.text + " | Output emoji: " + top_emoji["label"])
    return {"emoji": top_emoji["label"]}


@app.post("/infer/bert_base")
async def infer_text(req: TextRequest):
    sent_emb = get_sentence_embedding(req.text)
    sent_emb /= np.linalg.norm(sent_emb)

    # compute vectorized cosine similarity (matrix multiplication) between input sentence embedding and all emoji name embeddings
    similar_names = emb_matrix @ sent_emb
    best_idx = int(np.argmax(similar_names))  # get index of the most similar emoji name
    emoji_name = emoji_names[best_idx]
    emoji_char = next(e["char"] for e in emoji_data if e["name"] == emoji_name)

    # split the full emoji name into candidate words (filter out stopwords and punctuation)
    candidates = [
        w.strip(" ,-’'\"").lower()
        for w in emoji_name.split()
        if w.lower() not in stopwords
    ]

    # compute cosine similarity between input sentence embedding and all word embeddings in the emoji name to pick 1 word that represents the input sentence the best
    best_score = -1.0
    emoji_tag = None

    for word in candidates:
        word_emb = get_sentence_embedding(word)
        word_emb /= np.linalg.norm(word_emb)

        score = float(np.dot(sent_emb, word_emb))

        if score > best_score:
            best_score = score
            emoji_tag = word

    print("")
    print(
        "Input: "
        + req.text
        + " | Output emoji: "
        + emoji_char
        + " | full name: "
        + emoji_name
        + " | short name: "
        + emoji_tag
    )

    return {"emoji": emoji_char, "name": emoji_tag}
