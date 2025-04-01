import pandas as pd

pd.set_option('display.max_colwidth', 100)  

df = pd.read_csv('sample_text.csv')

# print(df.shape)

from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-mpnet-base-v2")
vectors = encoder.encode(df.text)
# print(vectors.shape)
# print(vectors)

dim = vectors.shape[1]
# print(dim)

import faiss  
index = faiss.IndexFlatL2(dim)
# print(index)
index.add(vectors)

# search_query = "I want to buy a polo t-shirt"
search_query = "An apple a day keeps doctor away!"

vec = encoder.encode(search_query)

import numpy as np
svec = np.array(vec).reshape(1, -1)

distances, I = index.search(svec, k = 2)
print(I)

print(df.loc[I[0]])