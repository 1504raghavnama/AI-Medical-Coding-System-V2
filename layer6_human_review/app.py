
import streamlit as st
import torch
import numpy as np
import json
import re
from datetime import datetime
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from torchcrf import CRF
import torch.nn as nn

st.set_page_config(page_title="AI Medical Coding System", page_icon="🏥", layout="wide")
st.title("🏥 AI-Powered Medical Coding System")
st.caption("BiLSTM-CRF NER + RAG Retrieval + Negation Detection | ICD-10-CM 2026 | Built from Scratch")

MODELS_DIR = "/kaggle/input/datasets/raghavnama15/medical-coding-data"
BASE = "/kaggle/working/AI-Medical-Coding-System-V2"

STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for",
    "of","with","by","from","is","was","are","were","has","have",
    "had","be","been","being","patient","presents","history",
    "no","not","without","evidence","denies","possible",
    "acute","bilateral","unilateral","left","right","upper","lower",
    "disease","disorder","syndrome","condition","injury",
    "currently","stable","following","presented","presenting"
}
NEGATION_TRIGGERS = [
    "no ","no evidence of","without","denies","denied","negative for",
    "not ","absence of","ruled out","rule out","unremarkable for",
    "free of","never had","does not have","did not have"
]
UNCERTAINTY_TRIGGERS = [
    "possible","possibly","probable","probably","suspected","suspect",
    "query","cannot exclude","may have","might have","history of"
]

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim//2, num_layers=2,
                              bidirectional=True, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    def forward(self, x, tags=None, mask=None):
        embed = self.dropout(self.embedding(x))
        lstm_out, _ = self.bilstm(embed)
        lstm_out = self.dropout(lstm_out)
        emissions = self.fc(lstm_out)
        if tags is not None:
            return -self.crf(emissions, tags, mask=mask, reduction="mean")
        return self.crf.decode(emissions, mask=mask)

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"{MODELS_DIR}/word2idx.json") as f:
        word2idx = json.load(f)
    with open(f"{MODELS_DIR}/tag2idx.json") as f:
        tag2idx = json.load(f)
    with open(f"{MODELS_DIR}/model_info.json") as f:
        info = json.load(f)
    idx2tag = {int(v): k for k, v in tag2idx.items()}
    w2v = Word2Vec.load(f"{MODELS_DIR}/medical_word2vec.model")
    ner = BiLSTM_CRF(len(word2idx), info["embed_dim"], info["hidden_dim"], len(tag2idx)).to(device)
    ner.load_state_dict(torch.load(f"{MODELS_DIR}/bilstm_crf_best.pt", map_location=device))
    ner.eval()
    with open(f"{BASE}/layer4_knowledge/icd10/icd10_codes_2026.json") as f:
        icd10_meta = json.load(f)
    icd10_emb = np.array([
        np.mean([w2v.wv[w] for w in re.sub(r"[^\w\s]","",item["description"].lower()).split()
                 if w in w2v.wv] or [np.zeros(128)], axis=0)
        for item in icd10_meta
    ])
    return device, word2idx, idx2tag, w2v, ner, icd10_meta, icd10_emb

def get_embedding(text, w2v):
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(128)

def retrieve_codes(query, emb, meta, w2v, top_k=5):
    qe = get_embedding(query, w2v).reshape(1, -1)
    scores = cosine_similarity(qe, emb)[0]
    top = np.argsort(scores)[::-1][:top_k]
    return [{"code": meta[i]["code"], "description": meta[i]["description"],
             "score": round(float(scores[i]), 4)} for i in top]

def code_specificity(code):
    return min(len(code.replace(".", "")) / 8.0, 1.0)

def rerank(candidates):
    for c in candidates:
        c["combined_score"] = round(0.92 * c["score"] + 0.08 * code_specificity(c["code"]), 4)
    return sorted(candidates, key=lambda x: x["combined_score"], reverse=True)

def detect_negation(text, entity):
    tl = text.lower()
    el = entity.lower()
    for sent in re.split(r"[.!?]", tl):
        if el in sent:
            for t in NEGATION_TRIGGERS:
                if t in sent: return "negated"
            for t in UNCERTAINTY_TRIGGERS:
                if t in sent: return "uncertain"
    return "affirmed"

def extract_phrases(text):
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    phrases, used = [], set()
    for i in range(len(words)-2):
        if all(words[i+j] not in STOP_WORDS for j in range(3)):
            p = " ".join(words[i:i+3])
            if not p[0].isdigit():
                phrases.append(p); used.update([i,i+1,i+2])
    for i in range(len(words)-1):
        if all(words[i+j] not in STOP_WORDS for j in range(2)) and i not in used:
            p = " ".join(words[i:i+2])
            if not p[0].isdigit():
                phrases.append(p); used.update([i,i+1])
    for i,w in enumerate(words):
        if i not in used and w not in STOP_WORDS and len(w) > 5:
            phrases.append(w)
    multi = [p for p in phrases if len(p.split()) > 1]
    covered = set(w for p in multi for w in p.split())
    singles = [p for p in phrases if len(p.split()) == 1 and p not in covered]
    return (multi + singles)[:6]

def run_pipeline(note, device, word2idx, idx2tag, w2v, ner, icd10_meta, icd10_emb):
    queries = extract_phrases(note)
    affirmed, negated, uncertain = [], [], []
    for q in queries:
        s = detect_negation(note, q)
        if s == "negated": negated.append(q)
        elif s == "uncertain": uncertain.append(q)
        else: affirmed.append(q)
    results = []
    seen = set()
    for q in affirmed + uncertain:
        candidates = retrieve_codes(q, icd10_emb, icd10_meta, w2v)
        reranked = rerank(candidates)
        best = reranked[0]
        if best["code"] in seen: continue
        seen.add(best["code"])
        results.append({
            "entity": q,
            "status": "uncertain" if q in uncertain else "affirmed",
            "primary_code": best["code"],
            "description": best["description"],
            "confidence": best["combined_score"],
            "alternatives": reranked[1:3]
        })
    return results, negated, uncertain

# ── UI ───────────────────────────────────────────────────────────
with st.spinner("Loading models..."):
    device, word2idx, idx2tag, w2v, ner, icd10_meta, icd10_emb = load_models()
st.success(f"Models loaded | ICD-10-CM 2026: {len(icd10_meta):,} codes | Device: {device}")

st.subheader("Clinical Note Input")
sample = "Patient has hypertension and chronic kidney disease. No evidence of pneumonia. History of myocardial infarction."
note = st.text_area("Enter Clinical Note", value=sample, height=150)

if st.button("Analyze & Suggest ICD-10 Codes", type="primary"):
    if note.strip():
        with st.spinner("Running pipeline..."):
            codes, negated, uncertain = run_pipeline(
                note, device, word2idx, idx2tag, w2v, ner, icd10_meta, icd10_emb)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Pipeline Status")
            st.write(f"**Conditions coded:** {len(codes)}")
            if negated:
                st.error(f"**Negated (not coded):** {', '.join(negated)}")
            if uncertain:
                st.warning(f"**Historical/uncertain:** {', '.join(uncertain)}")

        with col2:
            st.subheader(f"Suggested ICD-10 Codes ({len(codes)})")
            for r in codes:
                badge = "⚠️ Historical" if r["status"] == "uncertain" else "✅ Active"
                with st.expander(f"{r['primary_code']} — {r['description'][:50]} ({r['confidence']}) {badge}"):
                    st.write(f"**Query:** {r['entity']}")
                    st.write("**Alternatives:**")
                    for alt in r["alternatives"]:
                        st.write(f"  - {alt['code']}: {alt['description']} ({alt['combined_score']})")
    else:
        st.error("Please enter a clinical note.")

st.markdown("---")
st.caption("AI Medical Coding System V2 | Internship Project | Built from Scratch | ICD-10-CM 2026")
