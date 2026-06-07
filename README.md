# 🎓 Voice Learning Assistant

A voice-powered AI tutor that lets you have natural conversations about academic papers.
Upload any learning material — then ask questions using your voice and get spoken answers with citations.

[![Hugging Face](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/wesleyhuan/voice-learning-assistant)

---

## ✨ Features

- **Voice input & output** — speak your question, hear the answer
- **RAG-grounded answers** — always cited, no hallucinations
- **Multi-source ingestion** — upload PDF or any web URL at runtime
- **Three learning modes**:
  - 💡 **Explain** — concept explanation with analogies
  - 📝 **Quiz** — interactive testing based on the material
  - 💬 **Free Chat** — open-ended Q&A

---

## 🏗️ Architecture

```
User Voice Input
      ↓
faster-whisper ASR (base, CPU)
      ↓
Hybrid Search: BM25 + Vector (all-MiniLM-L6-v2) + RRF
      ↓
FlashRank Reranker
      ↓
Claude Sonnet 4.6 — Teaching Mode Prompt
      ↓
edge-tts TTS → Voice Response + Citations
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Claude Sonnet 4.6 (Anthropic) |
| ASR | faster-whisper (base, CPU/int8) |
| TTS | edge-tts (cloud) |
| Embedding | all-MiniLM-L6-v2 |
| Vector Store | Chroma |
| Retrieval | BM25 + Vector Hybrid + RRF + FlashRank |
| PDF Extraction | PyMuPDF |
| Web Scraping | trafilatura |

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/wesleyhuan/voice-learning-assistant
cd voice-learning-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API key
cp .env.example .env
# Edit .env → paste your ANTHROPIC_API_KEY

# 4. Prepare preloaded corpus (from HW3 output)
python convert_hw3.py --input arxiv_corpus.jsonl --output preloaded_corpus.jsonl

# 5. Run
streamlit run app.py
```

---

## ☁️ Hugging Face Space Deployment

1. Create a new **Streamlit** Space on [huggingface.co](https://huggingface.co)
2. Push this repo
3. Go to **Settings → Variables and secrets** → add `ANTHROPIC_API_KEY`
4. Commit `preloaded_corpus.jsonl` to the repo (pre-generated locally)
5. Space will start automatically — first load takes ~30s to build the index

---

## 📁 Project Structure

```
voice-learning-assistant/
├── app.py                    # Streamlit entry point
├── convert_hw3.py            # One-time HW3 → preloaded corpus converter
├── preloaded_corpus.jsonl    # Pre-processed arXiv EE papers (commit this)
├── requirements.txt
├── .env.example
└── src/
    ├── rag_pipeline.py       # Chunking, embedding, hybrid search, reranking
    ├── voice.py              # ASR (faster-whisper) + TTS (edge-tts)
    ├── llm_client.py         # Claude API + system prompts
    └── ingestion.py          # PDF + URL ingestion for user uploads
```

---

## 📊 Performance (HF Space free tier)

| Step | Latency |
|------|---------|
| Knowledge base build (startup) | ~30-45s (once) |
| ASR transcription | ~2-3s |
| RAG retrieval + rerank | ~1-2s |
| LLM generation | ~2-3s |
| TTS synthesis | ~1-2s |
| **Total per turn** | **~6-10s** |

---

Built as part of the **Inference.ai ML Engineering Mentorship Program (2025)**.
