# AI-Powered Medical Coding System V2

Internship project building a complete AI pipeline for automated ICD-10-CM and HCPCS medical code suggestion from clinical notes. All models trained from scratch following a 9-layer production architecture.

## Architecture Layers
| Layer | Description | Status |
|-------|-------------|--------|
| 1 | Input Layer (EHR/EMR, Clinical Notes) | Pending |
| 2 | NLP & Information Extraction (BiLSTM-CRF NER) | Complete |
| 3 | Coding Engine (Seq2Seq + RAG + Rules) | Complete |
| 4 | Knowledge & Data Layer (ICD-10, HCPCS, ChromaDB) | Complete |
| 5 | Output Layer (Structured JSON) | Complete |
| 6 | Human Review Interface (Streamlit UI) | Complete |
| 7 | Integration Layer | Pending |
| 8 | Governance, Security & Monitoring | Pending |
| 9 | Infrastructure | Pending |

## Models Trained from Scratch
- **BiLSTM-CRF** — Named Entity Recognition on clinical text (3,785 training samples)
- **Word2Vec** — Medical embeddings trained on 81,374 sentences (128 dimensions, 20 epochs)
- **Seq2Seq Encoder-Decoder** — ICD-10 code generation from clinical text (10,253 training pairs)

## Knowledge Base
- ICD-10-CM 2026 — 74,719 diagnosis codes (CMS.gov)
- HCPCS July 2026 — 1,689 procedure codes (CMS.gov)
- New ICD-10 2026 addenda — 487 new codes, 28 deleted
- All indexed in ChromaDB vector database with semantic search

## Tech Stack
- Python 3.12
- PyTorch (model training on GPU)
- Gensim (Word2Vec)
- ChromaDB (Vector Database)
- Streamlit (UI)
- Kaggle GPU T4 x2 (30GB VRAM)

## Dataset
- MTSamples (4,966 clinical notes, 40 specialties)
- ICD-10-CM 2026 (74,719 codes)
- HCPCS 2026 (1,689 codes)
- MIMIC-IV (pending PhysioNet approval — will retrain all models)

## Pipeline Flow
Clinical Note → BiLSTM-CRF NER → Word2Vec RAG Retrieval → Seq2Seq Code Generation → Rule Engine Validation → Structured JSON Output → Streamlit UI

## Weekly Progress
See docs/weekly_updates/ for detailed progress logs.

## Author
Raghav Nama | Internship Project | June 2026
