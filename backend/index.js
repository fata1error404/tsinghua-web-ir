const express = require('express');
const bodyParser = require('body-parser');
const { MongoClient } = require('mongodb');
const word2vec = require('word2vec');
const emojisUnicode = require('./emoji.json');
const emojisTwitter = require('./emoji_twitter.json');

const path = require('path');
const app = express();

app.use(express.static(path.join(__dirname, '../frontend')));
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());


const dbName = process.env.MONGO_INITDB_DATABASE;
const uri = `mongodb://${process.env.MONGO_INITDB_ROOT_USERNAME}:${process.env.MONGO_INITDB_ROOT_PASSWORD}@localhost:27017/${process.env.MONGO_INITDB_DATABASE}?authSource=admin`;
const client = new MongoClient(uri);

let emojisCollection;
let tagsCollection;
let wordEmbeddingsModel;

async function initDb() {
    try {
        await client.connect();
        const db = client.db(dbName);
        emojisCollection = db.collection("Emojis");
        tagsCollection = db.collection("Tags");
        console.log("✅ Connected to MongoDB");
    } catch (err) {
        console.error("Failed to connect to MongoDB:", err);
        process.exit(1);
    }
}

async function initWord2Vec() {
    try {
        wordEmbeddingsModel = await new Promise((resolve, reject) => {
            word2vec.loadModel('glove.6B.50d.word2vec.bin', (err, m) => {
                if (err) return reject(err);
                resolve(m);
            });
        });
        console.log("✅ Word2Vec model loaded");
    } catch (err) {
        console.error("Failed to load Word2Vec model:", err);
        process.exit(1);
    }
}

// utility function for emoji database search: expand query tag to nearest known tag by cosine similarity
async function expandTag(query) {
    try {
        // check if the word exists in the Word2Vec model vocabulary
        if (!wordEmbeddingsModel || !wordEmbeddingsModel.getVector || !wordEmbeddingsModel.getVector(query))
            return null;

        // find the 20 tags most similar to query in the Word2Vec embedding space
        const similarTags = wordEmbeddingsModel.mostSimilar(query, 20);

        // return the first matching tag in the Tags table
        for (const { word } of similarTags) {
            const matchingTag = await tagsCollection.countDocuments({ tag: word });
            if (matchingTag > 0)
                return word;
        }
    } catch {
        // silently ignore any errors, fallback to normal search
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
    const searchInput = (req.query.q || '').trim().toLowerCase();
    const useDb = req.query.database === 'enabled';
    const useWord2Vec = req.query.mode === 'smart';
    const includeGIFs = req.query.type === 'all';

    let filter;
    let results = [];

    try {
        if (useDb) {
            // ─── Discord emoji search ───
            await client.connect();

            // build MongoDB search filter
            const escaped = searchInput.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // quote special characters with '\\'
            const regex = new RegExp(escaped, 'i'); // create a case-insensitive regular expression from the escaped input

            if (useWord2Vec) {
                const expanded = await expandTag(searchInput); // try to expand the search tags using word embeddings for semantic similarity
                filter = expanded
                    ? { $or: [{ name: regex }, { tags: expanded }] }
                    : { $or: [{ name: regex }, { tags: regex }] };
            } else
                filter = { $or: [{ name: regex }, { tags: regex }] };

            // query the first 100 matching by name or tag emojis in the Emojis table
            let resultsRaw = await emojisCollection
                .find(filter)
                .limit(100)
                .project({ _id: 0, link: 1, name: 1, tags: 1 })
                .toArray();

            // filter out duplicate emoji results based on the 'name' field
            const seen = new Set();
            results = resultsRaw.filter(e => {
                if (seen.has(e.name))
                    return false;

                seen.add(e.name);
                return true;
            });

            if (!includeGIFs)
                results = results.filter(e => e.link.endsWith('.png') || e.link.endsWith('.jpg'));

            await client.close();
        } else {
            // ─── Unicode emoji search ───

            // query emojis by exact or partial match on name, group, or subgroup
            let resultsRaw = emojisUnicode.filter(e => {
                if (!searchInput)
                    return true;

                const name = e.name.toLowerCase();
                const group = e.group.toLowerCase();
                const sub = e.subgroup.toLowerCase();
                return name.includes(searchInput) || group.includes(searchInput) || sub.includes(searchInput);
            });

            // separate exact name matches from partial matches
            const exact = [];
            const partial = [];

            resultsRaw.forEach(e => {
                if (e.name.toLowerCase() === searchInput) {
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
        console.error('Error searching emoji:', err);
        res.status(500).json({ error: 'Error searching emoji' });
    }
});



// emoji prediction API
app.post('/api/emoji-predict', async (req, res) => {
    const textInput = req.body.text || '';
    const useDb = req.query.database === 'enabled';

    // send input text to the LLM server for inference (predict the emoji by the last sentence in the text area)
    const askModel = await fetch(`http://localhost:8000/infer/${req.query.model}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textInput })
    });

    if (!askModel.ok)
        throw new Error(await askModel.text());

    const { emoji, name } = await askModel.json();

    if (!useDb)
        return res.json({ link: null, name: emoji });

    try {
        let predictedTag;
        if (req.query.model === "bert_fine_tuned")
            predictedTag = emojisTwitter.find(e => e.emoji === emoji)?.tag || null;
        else
            predictedTag = name;

        if (!predictedTag)
            return res.status(500).json({ error: 'Unknown emoji mapping' });

        // search for the first matching emoji by the predicted tag
        const searchUrl = `/api/emoji-search?q=${encodeURIComponent(predictedTag)}&mode=smart&type=all&database=enabled`;
        const searchEmoji = await fetch(`http://localhost:3000${searchUrl}`);

        if (!searchEmoji.ok)
            throw new Error(`Search failed: ${await searchEmoji.text()}`);

        const result = (await searchEmoji.json())[0];
        return res.json({ link: result.link });
    } catch (err) {
        console.error('Error predicting emoji:', err);
        return res.status(500).json({ error: 'Error predicting emoji' });
    }
});




// ____________
// START SERVER
app.listen(3000, async () => {
    console.log('Server running on port 3000');
    await initDb();
    await initWord2Vec();
});