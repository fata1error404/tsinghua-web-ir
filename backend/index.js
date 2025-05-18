const express = require('express');
const bodyParser = require('body-parser');
const word2vec = require('word2vec');
const emojiList = require('./emoji.json'); // just require JSON normally
const { MongoClient } = require('mongodb');
const path = require('path');

const app = express();

// Serve static files first
app.use(express.static(path.join(__dirname, '../frontend')));
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

const dbName = "emoji-database";
const uri = "mongodb://localhost:27017";
const client = new MongoClient(uri);

let emojiColl;
let model;

async function initDb() {
    try {
        await client.connect();
        const db = client.db(dbName);
        emojiColl = db.collection("Emojis");
        console.log("✅ Connected to MongoDB");
    } catch (err) {
        console.error("Failed to connect to MongoDB:", err);
        process.exit(1);
    }
}

// Load Word2Vec model with async/await and explicit file descriptor
async function initModel() {
    try {
        model = await new Promise((resolve, reject) => {
            word2vec.loadModel('glove.6B.50d.word2vec.bin', (err, m) => {
                if (err) return reject(err);
                resolve(m);
            });
        });
        console.log("✅ Word2Vec model loaded");
    } catch (err) {
        console.error("Failed to load Word2Vec model:", err);
        throw err;
    }
}

// Utility: expand query tag to nearest known tag by DB lookup
async function expandTag(query) {
    // direct match in DB
    const exists = await emojiColl.countDocuments({ tags: query });
    if (exists > 0) {
        return query;
    }

    try {
        // Check if the word exists in the model vocabulary
        if (!model || !model.getVector || !model.getVector(query)) {
            // Word not in vocabulary — silently ignore and fallback
            return null;
        }

        const sims = model.mostSimilar(query, 20);
        for (const { word } of sims) {
            const found = await emojiColl.countDocuments({ tags: word });
            if (found > 0) {
                return word; // Return first matching tag found via similarity
            }
        }
    } catch {
        // Silently ignore any errors, fallback to normal search
    }

    return null;
}




app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/mainpage.html'));
});



app.get('/editor', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/editor.html'));
});


// emoji search API
app.get('/api/emoji-search', async (req, res) => {
    const useDb = req.query.database === 'enabled';
    const q = (req.query.q || '').trim().toLowerCase();
    const isSmart = req.query.mode === 'smart';
    const wantAll = req.query.type === 'all';

    try {
        let results = [];

        if (useDb) {
            // —–– Custom DB search (unchanged)
            await client.connect();

            const escaped = q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(escaped, 'i');

            // Build Mongo filter
            let filter;
            if (isSmart) {
                const expanded = await expandTag(q);
                filter = expanded
                    ? { $or: [{ name: regex }, { tags: expanded }] }
                    : { $or: [{ name: regex }, { tags: regex }] };
            } else {
                filter = { $or: [{ name: regex }, { tags: regex }] };
            }

            // Fetch up to 100 results
            let rawResults = await emojiColl
                .find(filter)
                .limit(100)
                .project({ _id: 0, link: 1, name: 1, tags: 1 })
                .toArray();

            // De-duplicate by emoji name
            const seen = new Set();
            results = rawResults.filter(e => {
                if (seen.has(e.name)) return false;
                seen.add(e.name);
                return true;
            });

            // Optionally keep only image links
            if (!wantAll) {
                results = results.filter(e =>
                    e.link.endsWith('.png') || e.link.endsWith('.jpg')
                );
            }

            await client.close();

        } else {
            // —–– Unicode JSON search: exact-name matches first
            const filtered = emojiList.filter(e => {
                if (!q) return true;
                const name = e.name.toLowerCase();
                const group = e.group.toLowerCase();
                const sub = e.subgroup.toLowerCase();
                return name.includes(q) || group.includes(q) || sub.includes(q);
            });

            const exact = [];
            const partial = [];
            filtered.forEach(e => {
                if (e.name.toLowerCase() === q) {
                    exact.push(e);
                } else {
                    partial.push(e);
                }
            });

            results = [...exact, ...partial]
                .slice(0, 100)
                .map(e => ({
                    link: null,
                    name: e.char,
                    tags: [e.name]
                }));
        }

        res.json(results);
    } catch (err) {
        console.error('Error querying emoji-search:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});



// async function emojiPredict(text) {
//     const res = await fetch('http://localhost:8000/infer/bert_base', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ text })
//     });
//     if (!res.ok) throw new Error(await res.text());
//     return res.json(); // returns { emoji, scores }
// }

const emojiToTag = {
    '❤': 'heart',
    '😃': 'smile',
    '😂': 'joy',
    '😍': 'heart_eyes',
    '😭': 'sob',
    '😊': 'blush',
    '💕': 'two_hearts',
    '🔥': 'fire',
    '😁': 'grin',
    '😒': 'unamused',
    '👍': 'thumbsup',
    '🙌': 'raised_hands',
    '😘': 'kissing_heart',
    '😩': 'weary',
    '😔': 'pensive',
    '☀': 'sun',
    '🎉': 'tada',
    '💙': 'blue_heart',
    '✨': 'sparkles',
    '💜': 'purple_heart'
};

app.post(
    '/api/emoji-predict',
    async (req, res) => {
        const useDb = req.query.database === 'enabled';
        const text = req.body.text || '';

        const askModel = await fetch(`http://localhost:8000/infer/${req.query.model}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (!askModel.ok) throw new Error(await askModel.text());

        const { emoji, name } = await askModel.json();

        if (!useDb) {
            return res.json({ link: null, name: emoji });
        }

        try {
            // 2) Map to tag
            let tag;
            if (req.query.model === "bert_fine_tuned") {
                tag = emojiToTag[emoji];
            } else {
                tag = name;
            }

            if (!tag) {
                return res.status(500).json({ error: 'Unknown emoji mapping' });
            }

            // 3) Call search API internally
            const searchUrl = `/api/emoji-search?q=${encodeURIComponent(tag)}&mode=smart&type=all&database=enabled`;
            const searchRes = await fetch(`http://localhost:3000${searchUrl}`);
            if (!searchRes.ok) {
                throw new Error(`Search failed: ${await searchRes.text()}`);
            }
            const searchResults = await searchRes.json(); // array of objects

            if (!Array.isArray(searchResults) || searchResults.length === 0) {
                return res.status(404).json({ error: 'No emojis found' });
            }

            // 4) Pick one at random
            const chosen = searchResults[0];
            // const randomIndex = Math.floor(Math.random() * searchResults.length);
            // const chosen = searchResults[randomIndex];

            // 5) Return only the link string
            return res.json({ link: chosen.link });

        } catch (err) {
            console.error('Emoji-predict error:', err);
            return res.status(500).json({ error: 'Prediction failed' });
        }
    }
);

// ____________
// START SERVER

app.listen(3000, async () => {
    console.log('Server running on port 3000');
    await initDb();
    await initModel();
});