import streamlit as st
import asyncio
import hashlib
import os
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY (and any other vars) from .env — Streamlit does not
# do this automatically. On HF Spaces the key comes from Space Secrets instead;
# load_dotenv() is a no-op there when .env is absent.
load_dotenv()

st.set_page_config(
    page_title="Voice Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

from src.rag_pipeline import build_index_from_jsonl, add_documents, hybrid_search
from src.voice import transcribe, synthesize
from src.llm_client import get_teaching_response

# ─── Load Knowledge Base (cached, runs once per session) ───────────────────
@st.cache_resource(show_spinner="📚 Building knowledge base from arXiv papers (~30s)...")
def load_kb():
    return build_index_from_jsonl("preloaded_corpus.jsonl")

vectorstore, bm25, chunks = load_kb()

# ─── Session State ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "explain"

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Voice Learning\nAssistant")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🎤 Voice Chat", "📚 Knowledge Base", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption(f"📄 {len(set(c.metadata.get('source','') for c in chunks))} papers loaded")
    st.caption(f"🔢 {len(chunks)} chunks indexed")

    if st.session_state.get("messages"):
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: Voice Chat
# ════════════════════════════════════════════════════════════════════════════
if page == "🎤 Voice Chat":
    col_title, col_mode = st.columns([3, 1])

    with col_title:
        st.title("🎤 Voice Chat")
    with col_mode:
        mode = st.selectbox(
            "Learning Mode",
            ["explain", "quiz", "free"],
            format_func=lambda x: {
                "explain": "💡 Explain",
                "quiz": "📝 Quiz",
                "free": "💬 Free Chat"
            }[x]
        )
        st.session_state.mode = mode

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/mp3")
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.caption(f"• {src}")

    st.divider()

    # Input: voice or text
    col_voice, col_text = st.columns([1, 1])
    with col_voice:
        audio_input = st.audio_input("🎤 Record your question")
    with col_text:
        text_input = st.chat_input("Or type your question here...")

    # Process input.
    # st.audio_input retains its recording across reruns, so transcribe only
    # when the audio is new (hash differs from the last processed one);
    # otherwise a stale recording would be re-transcribed on every rerun.
    query = None
    audio_bytes_in = None
    audio_hash = None
    if audio_input is not None:
        audio_bytes_in = audio_input.getvalue()
        audio_hash = hashlib.md5(audio_bytes_in).hexdigest()

    if audio_hash and audio_hash != st.session_state.get("last_audio_hash"):
        st.session_state.last_audio_hash = audio_hash
        with st.spinner("Transcribing..."):
            query = transcribe(audio_bytes_in)
        st.info(f"🎤 You said: *{query}*")
    elif text_input:
        query = text_input

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                results = hybrid_search(query, vectorstore, bm25, chunks)
                context = "\n\n".join(r.page_content for r in results)
                sources = list(set(r.metadata.get("source", "Unknown") for r in results))

            with st.spinner("Generating response..."):
                response = get_teaching_response(
                    query, context, st.session_state.mode
                )
                st.write(response)

            with st.spinner("Synthesizing audio..."):
                audio_bytes = asyncio.run(synthesize(response))
                st.audio(audio_bytes, format="audio/mp3")

            with st.expander("📎 Sources"):
                for src in sources:
                    st.caption(f"• {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "audio": audio_bytes,
            "sources": sources
        })

        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: Knowledge Base
# ════════════════════════════════════════════════════════════════════════════
elif page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base")

    # Stats
    col1, col2, col3 = st.columns(3)
    sources = list(set(c.metadata.get("source", "") for c in chunks))
    with col1:
        st.metric("Papers", len(sources))
    with col2:
        st.metric("Total Chunks", len(chunks))
    with col3:
        st.metric("Chunk Size", "512 tokens")

    # Loaded papers list
    st.subheader("Preloaded Papers (arXiv EE)")
    for src in sorted(sources):
        st.markdown(f"- `{src}`")

    st.divider()

    # Upload additional materials
    st.subheader("Add More Materials")
    tab1, tab2 = st.tabs(["📄 Upload PDF", "🌐 From URL"])

    with tab1:
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded and st.button("Add PDF to Knowledge Base"):
            with st.spinner(f"Processing {uploaded.name}..."):
                from src.ingestion import ingest_pdf
                new_chunks = ingest_pdf(uploaded.read(), source_name=uploaded.name)
                add_documents(vectorstore, bm25, chunks, new_chunks)
                st.success(f"✅ Added {len(new_chunks)} chunks from **{uploaded.name}**")
                st.rerun()

    with tab2:
        url = st.text_input("Enter URL")
        if url and st.button("Add URL to Knowledge Base"):
            with st.spinner(f"Processing {url}..."):
                from src.ingestion import ingest_url
                new_chunks = ingest_url(url)
                if new_chunks:
                    add_documents(vectorstore, bm25, chunks, new_chunks)
                    st.success(f"✅ Added {len(new_chunks)} chunks from URL")
                    st.rerun()
                else:
                    st.error("Could not extract content from this URL.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: About
# ════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.markdown("""
    ## 🎓 Voice Learning Assistant

    A voice-powered AI tutor built on top of a RAG pipeline.
    Ask questions about any academic content using your voice — get spoken answers with citations.

    ---

    ### 🛠️ Tech Stack

    | Component | Technology |
    |-----------|------------|
    | Frontend | Streamlit |
    | LLM | Claude Sonnet 4.6 (Anthropic) |
    | ASR | faster-whisper (base, CPU) |
    | TTS | edge-tts |
    | Embedding | all-MiniLM-L6-v2 |
    | Vector Store | Chroma |
    | Retrieval | BM25 + Vector Hybrid + RRF + FlashRank |

    ---

    ### 📚 Three Learning Modes

    | Mode | Description |
    |------|-------------|
    | 💡 Explain | Concept explanation with analogies |
    | 📝 Quiz | Interactive testing based on the material |
    | 💬 Free Chat | Open-ended Q&A |

    ---

    Built as part of the **Inference.ai ML Engineering Mentorship Program (2025)**.

    **GitHub**: [wesleyhuan/voice-learning-assistant](https://github.com/wesleyhuan)
    """)
