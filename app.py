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

from src.rag_pipeline import (
    build_index_from_jsonl,
    build_index,
    add_to_index,
    hybrid_search,
)
from src.voice import transcribe, synthesize
from src.llm_client import get_teaching_response

MAX_PDF_MB = 20           # reject huge uploads before they OOM the free tier
MAX_AUDIO_MESSAGES = 5    # keep MP3 bytes only for the most recent replies

# ─── Load Knowledge Base (cached, runs once per process) ────────────────────
# The base index is shared across all sessions and treated as READ-ONLY.
# User uploads go into a per-session index (st.session_state.user_index).
@st.cache_resource(show_spinner="📚 Building knowledge base from arXiv papers (~30s on first run)...")
def load_kb():
    return build_index_from_jsonl(
        "preloaded_corpus.jsonl",
        cache_dir=os.environ.get("INDEX_CACHE_DIR", "/tmp/vla_index"),
    )

vectorstore, bm25, chunks = load_kb()

# ─── Session State ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "explain"


def _user_chunks() -> list:
    user_index = st.session_state.get("user_index")
    return user_index[2] if user_index else []


def _search_indexes() -> list:
    indexes = [(vectorstore, bm25, chunks)]
    user_index = st.session_state.get("user_index")
    if user_index:
        indexes.append(user_index)
    return indexes


def _add_user_chunks(new_chunks: list):
    """Add uploaded chunks to this session's private index only — the shared
    base index stays untouched, so uploads never leak between visitors."""
    user_index = st.session_state.get("user_index")
    if user_index is None:
        st.session_state.user_index = build_index(new_chunks)
    else:
        vs, _, ch = user_index
        new_bm25 = add_to_index(vs, ch, new_chunks)
        st.session_state.user_index = (vs, new_bm25, ch)


def _prune_audio(messages: list):
    """Drop MP3 bytes from older messages so long sessions don't grow memory."""
    with_audio = [m for m in messages if m.get("audio")]
    for m in with_audio[:-MAX_AUDIO_MESSAGES]:
        m["audio"] = None


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
    all_chunks = chunks + _user_chunks()
    st.caption(f"📄 {len(set(c.metadata.get('source','') for c in all_chunks))} papers loaded")
    st.caption(f"🔢 {len(all_chunks)} chunks indexed")

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
        if query:
            st.info(f"🎤 You said: *{query}*")
        else:
            st.warning("🎤 Couldn't make out any speech — please try recording again.")
            query = None
    elif text_input:
        query = text_input

    if query:
        # Snapshot history before appending, so the model sees prior turns
        # but not a duplicate of the current question.
        history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                results = hybrid_search(query, _search_indexes())
                context = "\n\n".join(r.page_content for r in results)
                sources = list(set(r.metadata.get("source", "Unknown") for r in results))

            try:
                with st.spinner("Generating response..."):
                    response = get_teaching_response(
                        query, context, st.session_state.mode, history=history
                    )
                    st.write(response)
            except Exception as e:
                # Drop the unanswered user turn so a retry starts clean.
                st.session_state.messages.pop()
                st.error(f"Failed to generate a response: {e}")
                st.stop()

            audio_bytes = None
            try:
                with st.spinner("Synthesizing audio..."):
                    audio_bytes = asyncio.run(synthesize(response))
                st.audio(audio_bytes, format="audio/mp3")
            except Exception:
                st.warning("🔇 Audio synthesis failed — showing text only.")

            with st.expander("📎 Sources"):
                for src in sources:
                    st.caption(f"• {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "audio": audio_bytes,
            "sources": sources
        })
        _prune_audio(st.session_state.messages)

        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: Knowledge Base
# ════════════════════════════════════════════════════════════════════════════
elif page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base")

    # Confirmation from a just-completed upload (set before st.rerun, which
    # would otherwise wipe an inline st.success before anyone could read it).
    if st.session_state.get("kb_flash"):
        st.success(st.session_state.pop("kb_flash"))

    # Stats
    user_chunks = _user_chunks()
    all_chunks = chunks + user_chunks
    col1, col2, col3 = st.columns(3)
    sources = list(set(c.metadata.get("source", "") for c in all_chunks))
    with col1:
        st.metric("Papers", len(sources))
    with col2:
        st.metric("Total Chunks", len(all_chunks))
    with col3:
        st.metric("Chunk Size", "512 tokens")

    # Loaded papers list
    st.subheader("Preloaded Papers (arXiv EE)")
    for src in sorted(set(c.metadata.get("source", "") for c in chunks)):
        st.markdown(f"- `{src}`")

    if user_chunks:
        st.subheader("Your Uploads (this session)")
        for src in sorted(set(c.metadata.get("source", "") for c in user_chunks)):
            st.markdown(f"- `{src}`")

    st.divider()

    # Upload additional materials
    st.subheader("Add More Materials")
    st.caption("Uploads are private to your session and reset when you close the tab.")
    tab1, tab2 = st.tabs(["📄 Upload PDF", "🌐 From URL"])

    with tab1:
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded and st.button("Add PDF to Knowledge Base"):
            if uploaded.size > MAX_PDF_MB * 1024 * 1024:
                st.error(f"PDF too large — the limit is {MAX_PDF_MB} MB.")
            else:
                with st.spinner(f"Processing {uploaded.name}..."):
                    from src.ingestion import ingest_pdf
                    new_chunks = ingest_pdf(uploaded.read(), source_name=uploaded.name)
                if new_chunks:
                    _add_user_chunks(new_chunks)
                    st.session_state.kb_flash = (
                        f"✅ Added {len(new_chunks)} chunks from **{uploaded.name}**"
                    )
                    st.rerun()
                else:
                    st.error("Could not extract any text from this PDF.")

    with tab2:
        url = st.text_input("Enter URL")
        if url and st.button("Add URL to Knowledge Base"):
            with st.spinner(f"Processing {url}..."):
                from src.ingestion import ingest_url
                new_chunks = ingest_url(url)
            if new_chunks:
                _add_user_chunks(new_chunks)
                st.session_state.kb_flash = f"✅ Added {len(new_chunks)} chunks from URL"
                st.rerun()
            else:
                st.error(
                    "Could not fetch or extract content from this URL. "
                    "Only public http(s) URLs are supported."
                )

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

    **GitHub**: [wesleyhuan/voice-learning-assistant](https://github.com/wesleyhuan/voice-learning-assistant)
    """)
