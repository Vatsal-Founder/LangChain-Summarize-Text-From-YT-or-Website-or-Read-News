import os
from dotenv import load_dotenv
load_dotenv()

import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader, WebBaseLoader

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website or Read News", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website or Read News")
st.subheader("Summarize URL")

with st.sidebar:
    user_key = st.text_input("Groq API Key (optional if set as secret)", value="", type="password")

# Prefer user input; fallback to env (Space secret)
groq_api_key = user_key.strip() or os.getenv("GROQ_API_KEY", "").strip()

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

url = st.text_input("URL", label_visibility="collapsed")

def build_llm():
    if not groq_api_key:
        st.error("Missing GROQ_API_KEY. Enter it in the sidebar or set a Space secret.")
        st.stop()
    # create only when we actually need it
    return ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)

if st.button("Summarize the Content from YT or Articles"):
    if not url.strip():
        st.error("Please provide a URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL (YouTube or website).")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in url or "youtu.be" in url:
                    loader = YoutubeLoader.from_youtube_url(url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                docs = loader.load()
                llm = build_llm()
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")

elif st.button("Read Today's News"):
    if not groq_api_key:
        st.error("Missing GROQ_API_KEY. Enter it in the sidebar or set a Space secret.")
    else:
        try:
            with st.spinner("Reading News.."):
                loader = WebBaseLoader("https://www.bbc.com")
                news = loader.load()
                llm = build_llm()
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary_new = chain.run(news)
                st.success(output_summary_new)
        except Exception as e:
            st.exception(f"Exception: {e}")
