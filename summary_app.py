# app.py
import os
import sys
import subprocess
from urllib.parse import urlparse, parse_qs

import streamlit as st
import validators

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_groq import ChatGroq

# Optional: lightweight web loader (avoid Unstructured to keep deps simple)
import requests
from bs4 import BeautifulSoup


# -----------------------
# Ensure correct youtube-transcript-api
# -----------------------
def ensure_yta():
    try:
        import youtube_transcript_api as yta
        from youtube_transcript_api import YouTubeTranscriptApi
        # verify the method exists
        if not hasattr(YouTubeTranscriptApi, "get_transcript"):
            raise ImportError("YouTubeTranscriptApi missing get_transcript")
        return yta, YouTubeTranscriptApi
    except Exception:
        # Install a known-good version at runtime (works on Streamlit Cloud)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "youtube-transcript-api==0.6.2"])
        import youtube_transcript_api as yta
        from youtube_transcript_api import YouTubeTranscriptApi
        return yta, YouTubeTranscriptApi

yta, YouTubeTranscriptApi = ensure_yta()


# -----------------------
# Streamlit config
# -----------------------
st.set_page_config(page_title="Summarize YT / Web (Groq + LangChain)", page_icon="🦜")
st.title("🦜 Summarize Text From YouTube or Websites")
st.caption("Groq + LangChain (bypassing list_transcripts)")

with st.sidebar:
    user_key = st.text_input("Groq API Key", type="password", help="Or set GROQ_API_KEY as a secret/env var")

GROQ_API_KEY = (user_key or os.getenv("GROQ_API_KEY", "")).strip()

@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY")
    return ChatGroq(model="gemma2-9b-it", api_key=api_key)

prompt = PromptTemplate(
    template="Provide a concise summary (~300 words) of the following content:\n\n{text}\n",
    input_variables=["text"]
)


# -----------------------
# YouTube helpers (no list_transcripts)
# -----------------------
def extract_youtube_id(url: str) -> str:
    u = urlparse(url)
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
        vid = parse_qs(u.query).get("v", [""])[0]
        if vid:
            return vid
    raise ValueError("Unsupported YouTube URL")

def load_youtube_docs(url: str, languages=("en", "en-US", "en-GB")):
    vid = extract_youtube_id(url)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=list(languages))
    except Exception:
        # Let the library pick any available transcript
        transcript = YouTubeTranscriptApi.get_transcript(vid)
    text = " ".join(ch["text"] for ch in transcript if ch.get("text"))
    if not text.strip():
        raise RuntimeError("Transcript is empty or unavailable for this video.")
    return [Document(page_content=text, metadata={"source": url})]


# -----------------------
# Simple web page loader (keeps Space light)
# -----------------------
def load_web_docs(url: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    # crude extract: join paragraphs
    paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]
    text = " ".join(p for p in paragraphs if p)
    if not text.strip():
        raise RuntimeError("Could not extract readable text from the page.")
    return [Document(page_content=text, metadata={"source": url})]


# -----------------------
# UI
# -----------------------
url = st.text_input("Paste a YouTube or webpage URL")

col1, col2 = st.columns(2)
with col1:
    summarize_btn = st.button("Summarize")
with col2:
    news_btn = st.button("Read BBC Homepage")

if summarize_btn:
    if not GROQ_API_KEY:
        st.error("Please set your Groq API key (sidebar or env var GROQ_API_KEY).")
        st.stop()
    if not url or not validators.url(url):
        st.error("Please enter a valid URL.")
        st.stop()

    try:
        with st.spinner("Fetching & summarizing..."):
            if ("youtube.com" in url) or ("youtu.be" in url):
                docs = load_youtube_docs(url)
            else:
                docs = load_web_docs(url)

            llm = get_llm(GROQ_API_KEY)
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            summary = chain.run(docs)
            st.success(summary)
    except Exception as e:
        st.exception(f"Exception: {e}")

if news_btn:
    if not GROQ_API_KEY:
        st.error("Please set your Groq API key (sidebar or env var GROQ_API_KEY).")
        st.stop()
    try:
        with st.spinner("Loading BBC homepage..."):
            docs = load_web_docs("https://www.bbc.com")
            llm = get_llm(GROQ_API_KEY)
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            summary = chain.run(docs)
            st.success(summary)
    except Exception as e:
        st.exception(f"Exception: {e}")
