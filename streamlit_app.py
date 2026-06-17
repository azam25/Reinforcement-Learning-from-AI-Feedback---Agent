"""
streamlit_app.py — Professional UI for the RLAIF RAG Agent.

Prerequisites:
  1. Server must be running:   python3 run_server.py
  2. Launch this app:          streamlit run streamlit_app.py
"""

import time
import httpx
import streamlit as st

# ─── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="RLAIF RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
API_BASE    = "http://localhost:8000"
TIMEOUT     = 120
ACCENT      = "#2563EB"   # blue-600
ACCENT_LIGHT = "#EFF6FF"  # blue-50

# ─── CSS — professional light theme ──────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #F8FAFC; }
[data-testid="stSidebar"]          { background: #FFFFFF; border-right: 1px solid #E2E8F0; }
[data-testid="stSidebar"] > div    { padding-top: 1.5rem; }

/* ── Sidebar header ── */
.sidebar-logo {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0 1rem 1.2rem 1rem;
    border-bottom: 1px solid #E2E8F0; margin-bottom: 1.2rem;
}
.sidebar-logo span { font-size: 1.15rem; font-weight: 700; color: #1E293B; }

/* ── Status badge ── */
.status-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.25rem 0.75rem; border-radius: 9999px;
    font-size: 0.78rem; font-weight: 600; margin-bottom: 1rem;
}
.status-ok  { background: #DCFCE7; color: #16A34A; }
.status-err { background: #FEE2E2; color: #DC2626; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] { border: none !important; }
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 10px !important;
    background: #F8FAFC !important;
    transition: border-color .2s;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #2563EB !important; }

/* ── Chat messages ── */
.chat-wrap { max-width: 820px; margin: 0 auto; padding-bottom: 6rem; }

.msg-user, .msg-agent {
    display: flex; gap: 0.75rem;
    padding: 0.9rem 1rem; border-radius: 12px;
    margin-bottom: 0.75rem; line-height: 1.6;
}
.msg-user  { background: #EFF6FF; border: 1px solid #BFDBFE; flex-direction: row-reverse; }
.msg-agent { background: #FFFFFF; border: 1px solid #E2E8F0; }

.avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.avatar-user  { background: #2563EB; color: white; }
.avatar-agent { background: #F1F5F9; color: #475569; }

.msg-body { flex: 1; }
.msg-meta { font-size: 0.72rem; color: #94A3B8; margin-top: 0.4rem; }

/* ── Source cards ── */
.source-card {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 8px; padding: 0.75rem 1rem;
    margin-bottom: 0.5rem; font-size: 0.83rem; color: #475569;
}
.source-idx {
    font-size: 0.72rem; font-weight: 700; color: #2563EB;
    text-transform: uppercase; letter-spacing: .05em; margin-bottom: 0.3rem;
}

/* ── Metric chips ── */
.chip-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.4rem 0 0.8rem; }
.chip {
    padding: 0.2rem 0.65rem; border-radius: 9999px;
    font-size: 0.74rem; font-weight: 600;
}
.chip-blue   { background: #DBEAFE; color: #1D4ED8; }
.chip-green  { background: #DCFCE7; color: #15803D; }
.chip-purple { background: #F3E8FF; color: #7C3AED; }

/* ── Input area ── */
.stChatInputContainer { max-width: 820px; margin: 0 auto; }

/* ── Progress / spinner ── */
.stSpinner > div { border-top-color: #2563EB !important; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important; font-weight: 600 !important;
    transition: all .15s !important;
}
.stButton > button:hover { transform: translateY(-1px); }

/* ── Section headers ── */
.section-header {
    font-size: 0.72rem; font-weight: 700; color: #94A3B8;
    text-transform: uppercase; letter-spacing: .08em;
    margin: 1.2rem 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state defaults ───────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":       [],
        "doc_loaded":     False,
        "doc_name":       None,
        "server_ok":      None,
        "rlaif":          True,
        "verbose":        1,
        "model_name":     "—",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── API helpers ──────────────────────────────────────────────────────────────
def _client():
    return httpx.Client(base_url=API_BASE, timeout=TIMEOUT)


def check_server() -> bool:
    try:
        with _client() as c:
            r = c.get("/health")
            data = r.json()
            st.session_state.model_name = data.get("model", "—")
            st.session_state.server_ok = True
            return True
    except Exception:
        st.session_state.server_ok = False
        return False


def ingest_file(uploaded_file) -> tuple[bool, str]:
    """Save upload to a temp file, call /ingest, return (ok, message)."""
    import tempfile, os
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        with _client() as c:
            r = c.post("/ingest", json={"file_path": tmp_path, "force_rebuild": False})
        if r.status_code == 200:
            return True, r.json().get("message", "OK")
        return False, r.json().get("detail", "Unknown error")
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(tmp_path)


def ask(question: str) -> dict:
    with _client() as c:
        r = c.post("/ask", json={
            "question": question,
            "rlaif":    st.session_state.rlaif,
            "verbose":  st.session_state.verbose,
        })
    if r.status_code == 200:
        return r.json()
    raise RuntimeError(r.json().get("detail", f"HTTP {r.status_code}"))


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span>🤖 RLAIF RAG Agent</span>
    </div>""", unsafe_allow_html=True)

    # Server status
    st.markdown('<p class="section-header">Server</p>', unsafe_allow_html=True)
    col_status, col_refresh = st.columns([3, 1])
    with col_refresh:
        if st.button("↺", help="Refresh server status"):
            check_server()
    if st.session_state.server_ok is None:
        check_server()
    if st.session_state.server_ok:
        st.markdown(
            f'<div class="status-badge status-ok">● Connected &nbsp;|&nbsp; {st.session_state.model_name}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-err">✕ Server offline — run: python3 run_server.py</div>',
            unsafe_allow_html=True,
        )

    # Document upload
    st.markdown('<p class="section-header">Document</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a PDF to get started",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("📥  Ingest document", use_container_width=True, type="primary"):
            with st.spinner("Embedding document…"):
                ok, msg = ingest_file(uploaded)
            if ok:
                st.session_state.doc_loaded = True
                st.session_state.doc_name   = uploaded.name
                st.session_state.messages   = []   # clear chat on new doc
                st.success(f"✓ Ready: **{uploaded.name}**")
            else:
                st.error(f"Ingestion failed: {msg}")

    if st.session_state.doc_loaded:
        st.markdown(
            f'<div class="status-badge status-ok">📄 {st.session_state.doc_name}</div>',
            unsafe_allow_html=True,
        )

    # Settings
    st.markdown('<p class="section-header">Settings</p>', unsafe_allow_html=True)
    st.session_state.rlaif = st.toggle(
        "RLAIF self-evaluation",
        value=st.session_state.rlaif,
        help="Enable iterative self-critique and refinement of the answer",
    )
    st.session_state.verbose = st.select_slider(
        "Verbosity",
        options=[0, 1, 2],
        value=st.session_state.verbose,
        help="0 = silent, 1 = normal, 2 = debug",
    )

    st.divider()

    # Clear chat
    if st.button("🗑  Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<br><p style='font-size:0.72rem;color:#94A3B8;text-align:center;'>"
        "API docs: <a href='http://localhost:8000/docs' target='_blank'>localhost:8000/docs</a></p>",
        unsafe_allow_html=True,
    )


# ─── Main area ────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='font-size:1.4rem;font-weight:700;color:#1E293B;margin-bottom:0.2rem;'>"
    "Document Q&amp;A Chat</h2>"
    "<p style='color:#64748B;font-size:0.9rem;margin-bottom:1.5rem;'>"
    "Upload a PDF in the sidebar, then ask questions below.</p>",
    unsafe_allow_html=True,
)

# ── Welcome card (empty state) ────────────────────────────────────────────────
if not st.session_state.messages:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**1. Connect**\nMake sure the server is running on port 8000.")
    with col2:
        st.info("**2. Upload**\nDrop a PDF in the sidebar and click Ingest.")
    with col3:
        st.info("**3. Ask**\nType any question in the chat box below.")
    st.markdown("<br>", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            if "meta" in msg:
                m = msg["meta"]
                chips = (
                    f'<div class="chip-row">'
                    f'<span class="chip chip-blue">⟳ {m["context_iterations"]} iteration(s)</span>'
                    f'<span class="chip chip-green">📄 {m["source_count"]} source(s)</span>'
                    f'<span class="chip chip-purple">{"✓ RLAIF" if st.session_state.rlaif else "⚡ Standard"}</span>'
                    f'</div>'
                )
                st.markdown(chips, unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg.get("ts"):
                st.caption(msg["ts"])
            if msg.get("sources"):
                with st.expander(f"📚 View {len(msg['sources'])} source chunk(s)", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        content = src if isinstance(src, str) else str(src)
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-idx">Chunk {i}</div>
                            {content[:600]}{"…" if len(content) > 600 else ""}
                        </div>""", unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    placeholder="Ask a question about your document…",
    disabled=not st.session_state.doc_loaded or not st.session_state.server_ok,
):
    # Guard: server must be up
    if not st.session_state.server_ok:
        st.error("Server is offline. Start it with: `python3 run_server.py`")
        st.stop()
    if not st.session_state.doc_loaded:
        st.warning("Please upload and ingest a document first.")
        st.stop()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call API and stream-style loading
    with st.spinner("Thinking…"):
        t0 = time.time()
        try:
            data   = ask(prompt)
            elapsed = time.time() - t0
            answer  = data.get("answer", "No answer returned.")
            meta    = {
                "context_iterations": data.get("context_iterations", 0),
                "source_count":       data.get("source_count", 0),
                "has_answer":         data.get("has_answer", False),
                "elapsed":            f"{elapsed:.1f}s",
            }
            st.session_state.messages.append({
                "role":    "agent",
                "content": answer,
                "meta":    meta,
                "ts":      f"Responded in {elapsed:.1f}s",
                "sources": [],
            })
        except RuntimeError as e:
            st.session_state.messages.append({
                "role":    "agent",
                "content": f"<span style='color:#DC2626'>⚠ Error: {e}</span>",
                "ts":      "",
            })

    st.rerun()
