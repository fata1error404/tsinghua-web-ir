import os
import json
import logging
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

# logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()

# global variables
unicode_emoji_data: list = []
unicode_emoji_names: list = []
unicode_emb_matrix: np.ndarray = None
discord_emoji_data: list = []
discord_emb_matrix: np.ndarray = None
stopwords: list = []
base_model: SentenceTransformer = None
fine_tuned_model: pipeline = None


def get_sentence_embedding(sentence: str) -> np.ndarray:
    result = base_model.encode(sentence)
    return result


# ─── LLM server startup ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    global \
        unicode_emoji_data, \
        unicode_emoji_names, \
        unicode_emb_matrix, \
        discord_emoji_data, \
        discord_emb_matrix, \
        stopwords, \
        base_model, \
        fine_tuned_model

    # load Unicode emoji JSON data and stopwords
    with open("backend/emoji.json", "r", encoding="utf-8") as f:
        unicode_emoji_data = json.load(f)

    with open("models/stopwords.txt", encoding="utf-8") as f:
        stopwords = [line.strip().lower() for line in f if line.strip()]

    logger.info(f"✅ Loaded emoji.json ({len(unicode_emoji_data)} entries)")

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
    unicode_emoji_names = [e["name"] for e in unicode_emoji_data]
    unicode_emoji_name_embeddings = []

    for name in unicode_emoji_names:
        emb = get_sentence_embedding(name)
        unicode_emoji_name_embeddings.append(emb)

    unicode_emb_matrix = np.stack(unicode_emoji_name_embeddings)
    unicode_emb_matrix /= np.linalg.norm(unicode_emb_matrix, axis=1, keepdims=True)
    logger.info("✅ Precomputed and normalized Unicode emoji name embeddings matrix")

    # [UPDATED] load Discord emoji data
    client = MongoClient(
        f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME')}:{os.getenv('MONGO_INITDB_ROOT_PASSWORD')}@localhost:27017/{os.getenv('MONGO_INITDB_DATABASE')}?authSource=admin"
    )
    collection = client[os.getenv("MONGO_INITDB_DATABASE")]["Emojis"]

    discord_emoji_data.clear()
    discord_emoji_desc_embeddings = []

    for doc in collection.find(
        {
            "description": {"$exists": True},
            "link": {"$exists": True},
            "downloads": {"$exists": True},
        }
    ):
        description = doc["description"]
        link = doc["link"]
        emb = get_sentence_embedding(description)
        discord_emoji_data.append(
            {"description": description, "link": link, "downloads": doc["downloads"]}
        )
        discord_emoji_desc_embeddings.append(emb)

    discord_emb_matrix = np.stack(discord_emoji_desc_embeddings)
    discord_emb_matrix /= np.linalg.norm(discord_emb_matrix, axis=1, keepdims=True)
    logger.info("✅ Precomputed and normalized Discord emoji name embeddings matrix")

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
async def infer_text(req: TextRequest, database: str = Query(None)):
    sent_emb = get_sentence_embedding(req.text)
    sent_emb /= np.linalg.norm(sent_emb)

    if database == "enabled":
        # ─── Discord emoji prediction ───
        similar_names = discord_emb_matrix @ sent_emb

        # identify top-10 by raw cosine similarity
        top10_idxs = np.argsort(similar_names)[::-1][:10]

        print("\nTop 10 candidates (raw):")
        for idx in top10_idxs:
            e = discord_emoji_data[idx]
            print(
                f"- {e['description']} | downloads: {e.get('downloads', 0)} | sim: {similar_names[idx]:.4f}"
            )

        # build normalized download bias only for those top-10
        top10_downloads = np.array(
            [discord_emoji_data[i].get("downloads", 0) for i in top10_idxs]
        )
        if top10_downloads.max() > 0:
            norm_downloads = top10_downloads / top10_downloads.max()
        else:
            norm_downloads = np.ones_like(top10_downloads)

        # combine raw similar_names + downloads bias (for top-10 only)
        bias_strength = 0.0
        top10_similar_names = similar_names[top10_idxs]
        biased_top10 = (
            1 - bias_strength
        ) * top10_similar_names + bias_strength * norm_downloads

        # pick the best among those 10
        best_subidx = int(np.argmax(biased_top10))  # index within 0–9
        best_idx = top10_idxs[best_subidx]  # global index
        best_emoji = discord_emoji_data[best_idx]

        return {"link": best_emoji["link"]}
    else:
        # ─── Unicode emoji prediction ───
        similar_names = unicode_emb_matrix @ sent_emb
        best_idx = int(np.argmax(similar_names))
        emoji_name = unicode_emoji_names[best_idx]
        emoji_char = next(
            e["char"] for e in unicode_emoji_data if e["name"] == emoji_name
        )
        return {"emoji": emoji_char}
