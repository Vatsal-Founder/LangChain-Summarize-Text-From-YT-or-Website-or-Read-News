# LangChain Summarize — YouTube • Websites • Today’s News

An end‑to‑end Generative AI app (built with **LangChain**) that:

* **Summarizes YouTube videos** (via transcript)
* **Summarizes any public website** (article/blog/docs)
* **Summarizes today’s news** into quick briefs

> Repo layout includes `app.py`, `requirements.txt`, and a notebook for experiments (`text_summarize.ipynb`). 

* Link: https://langchain-summarize-text-from-yt-or-website-or-read-news-abwfi.streamlit.app
---

## Features

* 🎬 **YouTube summarizer** — paste a video URL, get a clean, concise summary.
* 🌐 **Website summarizer** — paste any URL; the app fetches and compresses the main text.
* 🗞️ **Today’s news** — fetch & condense current headlines into fast briefs.
* ⚙️ **Pluggable LLM backend** — use **OpenAI** or **Groq** via env vars.
* 🧱 **Chunk‑then‑summarize** — robust for long content; uses LangChain splitters + summarization chains.

---

## How it Works

1. **Detect source**: YouTube vs. general URL vs. News mode.
2. **Load content**:

   * *YouTube*: load transcript for the video ID.
   * *Website*: fetch and clean the article text.
   * *News*: query a provider or RSS, then collect article text.
3. **Split text** with `RecursiveCharacterTextSplitter` (good defaults for length/overlap).
4. **Summarize** with a LangChain chain (map‑reduce or refine), using your selected LLM.
5. **Render** a crisp summary (plus optional bullet points or key takeaways).

---

## Quick Start

### 1) Install

```bash
git clone https://github.com/Vatsal-Founder/LangChain-Summarize-Text-From-YT-or-Website-or-Read-News.git
cd LangChain-Summarize-Text-From-YT-or-Website-or-Read-News
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # or create .env
```

### 2) Configure (env)

Pick an LLM backend and set the keys you’ll use.

```ini
# choose your model provider: openai | groq
LLM_BACKEND=openai

# OpenAI (for the summarizer)
OPENAI_API_KEY=your_openai_key
MODEL_NAME=gpt-4o-mini   # example; adjust to your code

# OR Groq (LLM via ChatGroq)
# GROQ_API_KEY=your_groq_key
# MODEL_NAME=llama-3.1-8b


```

### 3) Run (Streamlit)

```bash
streamlit run summary_app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Using the App

* **YouTube**: paste a full YouTube URL (e.g., `https://www.youtube.com/watch?v=...`) and click **Summarize**.
* **Website**: paste a public webpage URL (e.g., blog post, doc) and **Summarize**.
* **Today’s News**: pick a topic (or use Top Stories) → the app fetches the latest and returns a compact brief.

> Tip: For long videos/articles, prefer map‑reduce summarization to avoid losing context.

---

## Configuration Tips

* **MODEL\_NAME**: choose a faster model for quick scans; a larger model for deeper synthesis.
* **Splitter settings**: \~1000 chars with \~150 overlap is a solid default; tweak for very short/long pages.


---

## Project Structure

```
.
├── summary_app.py          # app entry (Streamlit)
├── requirements.txt        # Python deps
├── text_summarize.ipynb    # experiments / prototypes
└── README.md               # this file
```

---

## Example Output

> **Title:** Diffusion Models: A Gentle Introduction
> **Summary:** The video explains how diffusion models learn to denoise random noise ... (3–5 bullet points)
> **Key Points:** (1) Training objective ..., (2) Sampling schedule ..., (3) Strengths vs. GANs ...

---

## Deployment (Optional)

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "summary_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

Build & run:

```bash
docker build -t lc-summarize .
docker run -p 8501:8501 --env-file .env lc-summarize
```

### PaaS

* Command: `streamlit run summary_app.py --server.port $PORT --server.address 0.0.0.0`
* Set your env vars in the platform dashboard.

---

