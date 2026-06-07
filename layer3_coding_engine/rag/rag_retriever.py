
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

def get_embedding(text, model):
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(128)

def retrieve_codes(query, embeddings, metadata, w2v_model, top_k=5):
    query_emb = get_embedding(query, w2v_model).reshape(1, -1)
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"code": metadata[i]["code"], "description": metadata[i]["description"], 
             "score": round(float(scores[i]), 4)} for i in top_indices]
