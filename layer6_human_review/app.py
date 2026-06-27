
import streamlit as st
import torch
import numpy as np
import json
import re
from datetime import datetime
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Medical Coding System",
    page_icon="🏥",
    layout="wide"
)

st.title("AI-Powered Medical Coding System")
st.caption("NLP + RAG Pipeline | Trained from Scratch | ICD-10-CM 2026")

@st.cache_resource
def load_all():
    import sys
    sys.path.append("/kaggle/working/AI-Medical-Coding-System-V2")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocabularies
    with open("/kaggle/working/AI-Medical-Coding-System-V2/models/trained/word2idx.json") as f:
        word2idx = json.load(f)
    with open("/kaggle/working/AI-Medical-Coding-System-V2/models/trained/tag2idx.json") as f:
        tag2idx = json.load(f)
    
    idx2tag = {int(v): k for k, v in tag2idx.items()}
    
    # Load ICD10
    with open("/kaggle/working/AI-Medical-Coding-System-V2/layer4_knowledge/icd10/icd10_codes_2026.json") as f:
        icd10_metadata = json.load(f)
    
    icd10_embeddings = np.load(
        "/kaggle/working/AI-Medical-Coding-System-V2/layer3_coding_engine/rag/icd10_embeddings.npy"
    )
    
    w2v = Word2Vec.load(
        "/kaggle/working/AI-Medical-Coding-System-V2/models/trained/medical_word2vec.model"
    )
    
    return device, word2idx, idx2tag, icd10_metadata, icd10_embeddings, w2v

with st.spinner("Loading models..."):
    device, word2idx, idx2tag, icd10_metadata, icd10_embeddings, w2v = load_all()

st.success("Models loaded!")

def get_embedding(text, model):
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(128)

def retrieve_codes(query, top_k=5):
    query_emb = get_embedding(query, w2v).reshape(1, -1)
    scores = cosine_similarity(query_emb, icd10_embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{"code": icd10_metadata[i]["code"], 
             "description": icd10_metadata[i]["description"],
             "score": round(float(scores[i]), 4)} for i in top_idx]

# UI
st.subheader("Enter Clinical Note")
sample = "Patient has hypertension and chronic kidney disease. History of myocardial infarction."
note = st.text_area("Clinical Note", value=sample, height=150)

if st.button("Analyze & Suggest Codes", type="primary"):
    if note.strip():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Entity Detection")
            words = note.lower().split()
            st.write(f"Words detected: {len(words)}")
            
            # Simple keyword matching for demo
            medical_keywords = ["hypertension", "diabetes", "kidney", "heart", 
                               "pneumonia", "cancer", "fracture", "infection"]
            found = [w for w in words if w.strip(".,") in medical_keywords]
            for f in found:
                st.success(f)
        
        with col2:
            st.subheader("Suggested ICD-10 Codes")
            all_queries = list(set([w.strip(".,") for w in words 
                                   if w.strip(".,") in medical_keywords]))
            
            for query in all_queries:
                results = retrieve_codes(query, top_k=3)
                st.markdown(f"**{query.title()}**")
                for r in results:
                    st.info(f"{r['code']} — {r['description']} ({r['score']})")
                st.divider()
    else:
        st.error("Please enter a clinical note.")

st.markdown("---")
st.caption("AI Medical Coding System | Internship Project | Built from Scratch")
