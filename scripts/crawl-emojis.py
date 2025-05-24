#!/usr/bin/env python3
import time
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient, errors
import re

# — Config —
MONGO_URI = "mongodb://admin:webir2025@localhost:27017/emoji-database?authSource=admin"
DB_NAME = "emoji-database"
COLL_NAME = "Emojis"
BASE_URL = "https://emoji.gg"
MAX_PAGES = 5000
SLEEP_TIME = 0.5
PAGE_SLEEP = 0.1  # between detail requests
PAGE_SIZE = 30  # emojis per page, for debugging only

# — Mongo setup —
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLL_NAME]

# — HTTP session —
session = requests.Session()
session.headers.update({"User-Agent": "EmojiCrawler/1.0"})


def create_description(raw_name):
    """Convert raw emoji name to cleaned, lowercase, space-separated description."""
    # Normalize name: replace underscores/dashes, handle camelCase
    s = raw_name.replace("_", " ").replace("-", " ")
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    words = s.split()

    # Words to remove
    banned_words = {
        "pepe",
        "qiqi",
        "95",
        "monka",
        "cat",
        "doge",
        "peepo",
        "tt",
        "minecraft",
        "ez",
        "valorant",
        "amogus",
        "meme",
        "kanna",
        "sage",
        "pikachu",
        "beluga",
        "nezuko",
        "paimon",
        "roblox",
        "mario",
        "klee",
    }

    filtered = [w for w in words if w.lower() not in banned_words]
    return " ".join(filtered).lower()


def parse_downloads(text):
    """Convert badge text like '4.1M' or '12K' into integer thousands of downloads."""
    text = text.strip().upper()
    # Multipliers in thousands
    multipliers = {"K": 1, "M": 1_000, "B": 1_000_000}
    if text and text[-1] in multipliers:
        try:
            # e.g. '4.1M' -> 4.1 * 1000 = 4100
            return int(float(text[:-1]) * multipliers[text[-1]])
        except ValueError:
            return 0
    # plain number, convert to int thousands
    try:
        raw = int(text.replace(",", ""))
        return raw // 1000
    except ValueError:
        return 0


def fetch_emojis(page_num):
    """
    Fetch one listing page and return list of dicts:
      - detail_url
      - name
      - img_link
      - downloads (int thousands)
    """
    if page_num == 1:
        url = f"{BASE_URL}/?sort=downloads"
    else:
        url = f"{BASE_URL}/?sort=downloads&page={page_num}"

    r = session.get(url, timeout=10)
    print("→ GET", r.url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    entries = []
    for div in soup.select("div.col-md-2.col-sm-6.item"):
        btn = div.select_one(".buttons a.btn-primary")
        if not btn or not btn.get("href"):
            continue
        detail_url = btn["href"]
        if detail_url.startswith("/"):
            detail_url = BASE_URL + detail_url

        name_el = div.select_one(".item-details a.h5")
        img_el = div.select_one("img")
        if not name_el or not img_el:
            continue

        # download badge
        badge_el = div.select_one(".item-badge")
        downloads = parse_downloads(badge_el.text) if badge_el else 0

        entries.append(
            {
                "detail_url": detail_url,
                "name": name_el.text.strip(),
                "img_link": img_el.get("data-src") or img_el.get("src"),
                "downloads": downloads,
            }
        )
    return entries


def fetch_emoji_tags(detail_url):
    """Fetch detail page and return list of tags."""
    r = session.get(detail_url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    return [a.text.strip() for a in soup.select("div.card-body a.tag")]


# — Crawl loop with per-page DB save —
for page in range(1, MAX_PAGES + 1):
    print(f"\n📥 Fetching downloads page {page}…")
    entries = fetch_emojis(page)
    if not entries:
        print("🚫 No more emojis — stopping early.")
        break

    page_docs = []
    for e in entries:
        try:
            tags = fetch_emoji_tags(e["detail_url"])
        except Exception as err:
            print(f"  ⚠ Tag fetch failed for {e['name']}: {err}")
            tags = []

        description = create_description(e["name"])

        doc = {
            "link": e["img_link"],
            "name": e["name"],
            "description": description,
            "tags": tags,
            "downloads": e["downloads"],  # in thousands
        }
        page_docs.append(doc)
        print(f"  • {e['name']}  ({len(tags)} tags, description: {description})")
        time.sleep(PAGE_SLEEP)

    # Bulk insert this page’s docs
    print(f"💾 Inserting page {page} ({len(page_docs)} emojis) into MongoDB…")
    try:
        res = collection.insert_many(page_docs, ordered=False)
        print(f"✅ Inserted {len(res.inserted_ids)} new docs on page {page}")
    except errors.BulkWriteError:
        print(f"⚠ Some duplicates skipped on page {page}")

    print(f"   → Total collected so far (not persisted): {page * PAGE_SIZE}")
    time.sleep(SLEEP_TIME)

print("\n🎉 Crawl complete!")
