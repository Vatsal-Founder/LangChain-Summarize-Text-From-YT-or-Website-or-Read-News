# app.py
import os
from urllib.parse import urlparse, parse_qs

import streamlit as st
import validators
import requests
from bs4 import BeautifulSoup

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_groq import ChatGroq

# --- verify youtube-transcript-api is installed and has get_transcript
try:
    import youtube_transcript_api
    from youtube_transcript_api import YouTubeTranscriptApi
    _HAS_GET = hasattr(YouTubeTranscriptApi, "get_transcript")
except Exception as e:
    _HAS_GET = False

st.set_page_config(page_title="Summarize YT / Web", page_icon="🦜")
st.title("🦜 Summarize Text From YouTube or Websites")
st.caption(f"youtube-transcript-api: file={getattr(youtube_transcript_api, '__file__', 'unknown')} has_get={_HAS_GET}")

with st.sidebar:
    user_key = st.text_input("Groq API Key", type="password")
GROQ_API_KEY = (user_key or os.getenv("GROQ_API_KEY", "")).strip()

def get_llm():
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY (set sidebar input or env var).")
        st.stop()
    return ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY)

prompt = PromptTemplate(
    template="Provide a concise summary (~300 words) of the following content:\n\n{text}\n",
    input_variables=["text"]
)

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

def load_youtube_docs(url: str):
    if not _HAS_GET:
        raise RuntimeError("youtube-transcript-api is installed but missing get_transcript. "
                           "Check for a name collision or rebuild with a fresh environment.")
    vid = extract_youtube_id(url)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=["en", "en-US", "en-GB"])
    except Exception:
        transcript = YouTubeTranscriptApi.get_transcript(vid)  # fallback to any available language
    text = " ".join(ch["text"] for ch in transcript if ch.get("text"))
    if not text.strip():
        raise RuntimeError("Transcript is empty/unavailable for this video.")
    return [Document(page_content=text, metadata={"source": url})]

def load_web_docs(url: str):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]
    text = " ".join(p for p in paragraphs if p)
    if not text.strip():
        raise RuntimeError("Could not extract readable text.")
    return [Document(page_content=text, metadata={"source": url})]

url = st.text_input("Paste a YouTube or webpage URL")
if st.button("Summarize"):
    if not url or not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Fetching & summarizing..."):
                docs = load_youtube_docs(url) if ("youtube.com" in url or "youtu.be" in url) else load_web_docs(url)
                llm = get_llm()
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                st.success(chain.run(docs))
        except Exception as e:
            st.exception(f"Exception: {e}")
