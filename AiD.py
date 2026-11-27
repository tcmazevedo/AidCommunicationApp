import streamlit as st
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import re
from openai import OpenAI

# Configuration
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
LOGO_PATH = BASE_DIR / "logo.jpg"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 4
RELEVANCE_THRESHOLD = 0.65  # Tuned for cosine similarity approximation

st.set_page_config(
    page_title="AiD Communication App",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "💬",
    layout="wide"
)

# Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please add your OpenAI API key to .streamlit/secrets.toml as OPENAI_API_KEY")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=st.secrets["OPENAI_API_KEY"],
    model_name=EMBED_MODEL
)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="aid_knowledge",
    embedding_function=openai_ef
)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'©.*?(?=\n|$)', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def load_and_index_documents():
    if collection.count() > 0:
        return

    docs, metadatas, ids = [], [], []
    chunk_id = 0

    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages, 1):
                raw = page.extract_text() or ""
                text = clean_text(raw)
                if len(text) < 80:
                    continue
                for c_idx, chunk in enumerate(chunk_text(text)):
                    if len(chunk.strip()) < 80:
                        continue
                    docs.append(chunk)
                    metadatas.append({
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk_id": f"{pdf_path.stem}_p{page_num}_c{c_idx}"
                    })
                    ids.append(str(chunk_id))
                    chunk_id += 1
        except Exception as e:
            st.warning(f"Could not process {pdf_path.name}: {e}")

    if docs:
        collection.add(documents=docs, metadatas=metadatas, ids=ids)
        st.success(f"Knowledge base ready — {len(docs)} chunks indexed.")

if "indexed" not in st.session_state:
    with st.spinner("Preparing knowledge base (first run only)..."):
        load_and_index_documents()
    st.session_state.indexed = True

if "patient" not in st.session_state:
    st.session_state.patient = {
        "name": "Marli M",
        "diagnosis": "Expressive aphasia (post-stroke), possible reading difficulties",
        "stroke_date": "26 June 2021",
        "mobility": "Right-side weakness / hemiparesis",
        "medications": "Blood thinners, asthma medication",
        "oxygen": "Requires overnight oxygen; risk of desaturation and pneumonia without it",
        "languages": "Speaks 3 languages; primary English",
        "notes": "Focus on communication strategies, right-hand handwriting recovery, typing as alternative, cognitive and mobility exercises."
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
col1, col2 = st.columns([1, 5])
with col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)
with col2:
    st.title("AiD Communication App")
    st.caption("RAG-powered assistant for people with aphasia and their caregivers. Answers are grounded only in the provided documents (communication strategies, AAC resources, cognitive exercises, and mobility guides).")

st.sidebar.info("**Knowledge base:** 8 PDFs on aphasia communication, AAC, cognitive recovery, and physical mobility exercises.")
portal = st.sidebar.radio("Portal", ["Caregiver", "Patient"])

# Caregiver / Patient portals (unchanged interactive sections kept concise for length)
if portal == "Caregiver":
    st.header("Caregiver Portal")
    with st.expander("Patient File: Marli M", expanded=True):
        p = st.session_state.patient
        for k, v in p.items():
            if k != "notes":
                st.write(f"**{k.replace('_', ' ').title()}:** {v}")
        notes = st.text_area("Additional Notes", p["notes"])
        if st.button("Update Notes"):
            st.session_state.patient["notes"] = notes
            st.success("Updated")
    st.subheader("Ask grounded questions about strategies, AAC, or exercises")
else:
    st.header("Patient Portal")
    st.info("Interactive support for communication, voice practice, and motor skills recovery.")

    PHRASES = {
        "Basic needs": [
            "I need help.", "I'm thirsty.", "I'm hungry.", "I'm tired.",
            "I need the bathroom.", "I'm in pain.", "My head hurts.",
            "I'm okay.", "I'm not okay.", "Please wait.", "Give me a minute."
        ],
        "Social": [
            "How are you?", "Nice to see you.", "Thank you.", "You're welcome.",
            "I don't understand.", "Can you say that again?", "Please speak slowly.",
            "Yes.", "No.", "Maybe.", "I want to talk about…"
        ],
        "Daily life": [
            "I want coffee.", "I want water.", "I want food.", "Turn on the TV.",
            "What time is it?", "Where is my phone?", "I need my medicine.",
            "Can you help me get dressed?"
        ],
        "Health": [
            "I have a doctor's appointment.", "I need to see the speech therapist.",
            "Please call the nurse.", "I feel dizzy.", "I feel weak."
        ]
    }

    VOICE_WORDS = [
        "help", "water", "food", "tired", "pain", "bathroom",
        "yes", "no", "thank you", "please", "medicine", "home",
        "family", "doctor", "walk", "hand", "Fred", "Brazil"
    ]

    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Quick Phrases", "🗣️ Voice Practice", "✍️ Typing / Handwriting", "🏃 Motor Skills"
    ])

    with tab1:
        st.subheader("Quick Communication Phrases")
        category = st.selectbox("Category", list(PHRASES.keys()))
        cols = st.columns(2)
        for i, phrase in enumerate(PHRASES[category]):
            with cols[i % 2]:
                if st.button(phrase, key=f"phrase_{category}_{i}", use_container_width=True):
                    st.success(f"Selected: **{phrase}**")
                    st.session_state.messages.append({"role": "user", "content": phrase})

        st.divider()
        keyword = st.text_input("Type a key word for suggestions")
        if keyword:
            matches = [p for phrases in PHRASES.values() for p in phrases if keyword.lower() in p.lower()]
            if matches:
                for m in matches[:6]:
                    st.button(m, key=f"match_{m}", use_container_width=True)
            else:
                st.info("No exact match. Use the chat below.")

    with tab2:
        st.subheader("Voice Practice")
        if "voice_idx" not in st.session_state:
            st.session_state.voice_idx = 0
        if "voice_success" not in st.session_state:
            st.session_state.voice_success = []

        current = VOICE_WORDS[st.session_state.voice_idx]
        st.markdown(f"### Current word: **{current}**")
        st.progress((st.session_state.voice_idx + 1) / len(VOICE_WORDS))

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⬅️ Previous"):
                st.session_state.voice_idx = max(0, st.session_state.voice_idx - 1)
                st.rerun()
        with c2:
            if st.button("Skip / Next", type="secondary"):
                st.session_state.voice_idx = min(len(VOICE_WORDS)-1, st.session_state.voice_idx + 1)
                st.rerun()
        with c3:
            if st.button("Reset Progress"):
                st.session_state.voice_idx = 0
                st.session_state.voice_success = []
                st.rerun()

        st.write("Record yourself saying the word. The app will check if it was recognized.")
        try:
            audio = st.audio_input("Record the word")
            if audio is not None:
                st.audio(audio)
                with st.spinner("Checking what you said..."):
                    try:
                        # Transcribe with Whisper
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=("audio.wav", audio.getvalue(), "audio/wav"),
                            language="en"
                        )
                        said = transcript.text.strip().lower()
                        st.write(f"**Heard:** {said}")

                        # Simple match (exact or contained)
                        if current.lower() in said or said in current.lower():
                            st.success(f"Great! '{current}' was recognized.")
                            if current not in st.session_state.voice_success:
                                st.session_state.voice_success.append(current)
                            # Auto-advance after short delay feel
                            if st.button("Next word"):
                                st.session_state.voice_idx = min(len(VOICE_WORDS)-1, st.session_state.voice_idx + 1)
                                st.rerun()
                        else:
                            st.warning(f"Not quite. Try saying **{current}** more clearly.")
                    except Exception as e:
                        st.error(f"Could not process audio: {e}")
        except Exception:
            st.info("Audio recording requires a recent Streamlit version. You can still practice out loud.")

        if st.session_state.voice_success:
            st.caption(f"Completed: {', '.join(st.session_state.voice_success)}")

    with tab3:
        st.subheader("Typing & Handwriting Practice")
        typed = st.text_area("Type here", height=100, placeholder="Practice typing a word or sentence...")
        if typed:
            st.success(f"You typed: **{typed}**")
        st.markdown("""
        **Handwriting tips:** Start with large shapes and letters. Use a thicker pen. Practice short sessions. Typing is a good alternative.
        """)

    with tab4:
        st.subheader("Motor Skills Recovery Activities")
        activities = [
            "**Hip Flexion with Hold**: Sit and use hands to lift the affected leg toward the chest, hold briefly, lower slowly.",
            "**Hand Opening/Closing**: Slowly open and close the affected hand. Use a soft ball if available.",
            "**Shoulder Rolls**: Gently roll shoulders forward and backward.",
            "**Finger Tapping**: Tap each finger to the thumb one by one.",
            "**Seated Marching**: Lift knees alternately while sitting.",
            "**Wrist Circles**: Support the arm and make slow wrist circles."
        ]
        for i, act in enumerate(activities):
            with st.expander(f"Activity {i+1}"):
                st.markdown(act)
                if st.button("I did this today", key=f"motor_{i}"):
                    st.success("Great work!")

# Chat
st.divider()
st.subheader("Ask AiD")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** — page {s['page']} (score: {s['score']})")
                    st.caption(s["text"][:350] + "...")

if prompt := st.chat_input("Ask a question about communication, exercises, or AAC..."):
    if not prompt.strip():
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                results = collection.query(
                    query_texts=[prompt],
                    n_results=TOP_K,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                st.stop()

        sources = []
        relevant = []
        if results and results.get("documents") and results["documents"][0]:
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                score = round(1 - dist, 3)
                if score >= RELEVANCE_THRESHOLD or dist < 0.45:
                    relevant.append(doc)
                    sources.append({
                        "source": meta["source"],
                        "page": meta["page"],
                        "chunk_id": meta["chunk_id"],
                        "text": doc,
                        "score": score
                    })

        if not relevant:
            answer = "I could not find enough relevant information in the provided documents to answer that question reliably."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            context = "\n\n".join([f"[Source: {s['source']}, p. {s['page']}]\n{s['text']}" for s in sources])
            system = """You are AiD, a supportive assistant for caregivers and people with aphasia.
Answer ONLY using the retrieved context below. Never invent facts.
If the context is insufficient, say you cannot answer reliably from the documents.
Always include clear citations in the answer using the format [Source: filename, p. X].
Be clear, practical, and patient in tone. Follow the app's role of supporting communication and recovery."""
            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                    ],
                    temperature=0.2
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"Generation error: {e}. Please check your API key and connection."

            st.markdown(answer)
            with st.expander("View Sources"):
                for s in sources:
                    st.markdown(f"**{s['source']}** — page {s['page']} (score: {s['score']})")
                    st.caption(s["text"][:400] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
