const dbName = "emoji-database";
const emojisColl = "Emojis";
const tagsColl = "Tags";
const adminDB = db.getSiblingDB(dbName); // connect to the database

// create the Emojis table
adminDB.createCollection(emojisColl, {
    validator: {
        $jsonSchema: {
            required: ["link"],
            properties: {
                link: {
                    bsonType: "string"
                },
                name: {
                    bsonType: "string"
                },
                description: {
                    bsonType: "string"
                },
                tags: {
                    bsonType: "array",
                    items: {
                        bsonType: "string"
                    }
                },
                downloads: {
                    bsonType: "int"
                }
            }
        }
    }
});

// create a unique index for the link field (so that we don’t accidentally save the same article URL twice)
adminDB[emojisColl].createIndex(
    { link: 1 },
    { unique: true, name: "unique_link_idx" }
);

print(`Created collection '${dbName}.${emojisColl}' with schema validation and unique index on 'link'.`);


// create the Tags table
adminDB.createCollection(tagsColl, {
    validator: {
        $jsonSchema: {
            required: ["tag"],
            properties: {
                tag: { bsonType: "string" }
            }
        }
    }
});

// create a unique index for the tag field
adminDB[tagsColl].createIndex(
    { tag: 1 },
    { unique: true, name: "unique_tag_idx" }
);

print(`Created collection '${dbName}.${tagsColl}' with schema validation and unique index on 'tag'.`);