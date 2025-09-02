# app.py
import os
from urllib.parse import urlparse, parse_qs

import streamlit as st
import validators

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_groq import ChatGroq

# If you still want non-YouTube pages:
from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader

# BYPASS: use get_transcript (no list_transcripts)
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable


# -----------------------
# Streamlit page config
# -----------------------
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website or Read News", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website or Read News")
st.subheader("Summarize URL")

# -----------------------
# API key handling
# -----------------------
with st.sidebar:
    user_key = st.text_input("Groq API Key (optional if set as Space secret)", value="", type="password")

# Prefer user-provided key, else env (add GROQ_API_KEY as a Space secret)
GROQ_API_KEY = (user_key or os.getenv("GROQ_API_KEY", "")).strip()

@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY. Set a Space secret or enter it in the sidebar.")
    # Build only when needed
    return ChatGroq(model="gemma2-9b-it", api_key=api_key)

# -----------------------
# Prompt
# -----------------------
prompt_template = """Provide a summary of the following content in ~300 words.
Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# -----------------------
# YouTube helpers (BYPASS list_transcripts)
# -----------------------
def extract_youtube_id(yurl: str) -> str:
    """Robustly extract video id from watch/shorts/shortened URLs."""
    u = urlparse(yurl)
    host = (u.hostname or "").lower()
    if host in ("youtu.be", "www.youtu.be"):
        return u.path.lstrip("/")
    if "youtube" in host:
        if u.path == "/watch":
            return parse_qs(u.query).get("v", [""])[0]
        if u.path.startswith("/shorts/"):
            parts = u.path.split("/")
            if len(parts) >= 3:
                return parts[2]
        # Fallback for embeds or other paths: try v= first
        vid = parse_qs(u.query).get("v", [""])[0]
        if vid:
            return vid
    raise ValueError("Unsupported YouTube URL format")

def load_youtube_as_docs(yurl: str, languages=("en", "en-US", "en-GB")) -> list[Document]:
    """
    Get transcript using get_transcript only (no list_transcripts).
    If no transcript is found for preferred languages, try without languages (let the API pick).
    """
    vid = extract_youtube_id(yurl)

    # Try preferred languages first
    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=list(languages))
    except NoTranscriptFound:
        # Try letting the lib pick any available transcript (original language)
        transcript = YouTubeTranscriptApi.get_transcript(vid)
    except (TranscriptsDisabled, VideoUnavailable) as e:
        raise RuntimeError(f"Transcript not available: {e}")

    text = " ".join(chunk["text"] for chunk in transcript if chunk.get("text"))
    if not text.strip():
        raise RuntimeError("Transcript is empty.")
    return [Document(page_content=text, metadata={"source": yurl})]

# -----------------------
# UI inputs
# -----------------------
url = st.text_input("URL", label_visibility="collapsed")

# -----------------------
# Actions
# -----------------------
if st.button("Summarize the Content from YT or Articles"):
    # Basic validation
    if not GROQ_API_KEY or not url.strip():
        st.error("Please provide the Groq API Key (sidebar) and a URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL (YouTube or website).")
    else:
        try:
            with st.spinner("Loading & summarizing..."):
                # Build docs
                if ("youtube.com" in url) or ("youtu.be" in url):
                    docs = load_youtube_as_docs(url)
                else:
                    # You can remove UnstructuredURLLoader if you want fewer deps
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                # LLM + chain
                llm = get_llm(GROQ_API_KEY)
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)

                st.success(summary)
        except Exception as e:
            st.exception(f"Exception: {e}")

elif st.button("Read Today's News"):
    if not GROQ_API_KEY:
        st.error("Please provide the Groq API Key in the sidebar or as a Space secret.")
    else:
        try:
            with st.spinner("Reading News.."):
                loader = WebBaseLoader("https://www.bbc.com")
                docs = loader.load()
                llm = get_llm(GROQ_API_KEY)
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)
                st.success(summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
