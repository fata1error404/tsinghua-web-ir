from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Paths
glove_input_file = "../backend/glove.6B.50d.txt"
word2vec_output_file = "../backend/glove.6B.50d.word2vec.txt"
word2vec_bin_file = "../backend/glove.6B.50d.word2vec.bin"

# Step 1: Convert GloVe to Word2Vec text format
glove2word2vec(glove_input_file, word2vec_output_file)

# Step 2: Load Word2Vec format (text)
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Step 3: Save to binary format
model.save_word2vec_format(word2vec_bin_file, binary=True)

# Example usage: print similar words to "king"
print(model.most_similar("kitten"))
