# MuseAI — Text-to-Music Generation

A lightweight text-to-music generation interface built on Meta's MusicGen model. Type a description, get music back.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.32+-red.svg)
![HuggingFace](https://img.shields.io/badge/🤗-MusicGen-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## What it does

- **Text → Audio**: Describe music in plain English; the model generates a WAV file
- **Prompt Enhancement**: A rule-based layer enriches vague prompts with music-theory vocabulary (tempo, key, instrumentation context) that MusicGen responds better to
- **Streamlit UI**: Interactive controls for duration (5–30s), temperature, and model size
- **Local execution**: No API key required — runs entirely on your machine (CPU or GPU)

---

## Architecture

```
app.py                  ← Streamlit UI, generation flow
│
├── prompt_engine.py    ← PromptEnhancer: mood/instrument → musical vocabulary
│                          Rule-based; deterministic; no extra model needed
│
└── model_wrapper.py    ← MusicGenWrapper: HuggingFace model + processor
                           Lazy loading via st.cache_resource
                           Handles token math, device placement, output conversion
```

**Why rule-based prompt enhancement instead of an LLM?**

MusicGen's training data includes structured descriptions with tempo, key, and instrumentation details. A casual prompt like "sad piano" skips all of that vocabulary. The `PromptEnhancer` bridges this gap without adding an API cost or extra inference latency — just a dictionary lookup and some string assembly.

**Why `st.cache_resource` for the model?**

Streamlit reruns the whole script on every user interaction. Without caching, the ~300MB model weights would reload every time a slider moves. `st.cache_resource` keeps the model in memory and returns the same instance — standard practice for ML apps in Streamlit.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/poojasri7/museai-text-to-music
cd museai-text-to-music

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

**Hardware note:** The `musicgen-small` model (~300M parameters) runs on CPU but takes ~60–90 seconds per generation. With a CUDA GPU, this drops to ~5–10 seconds. `musicgen-medium` (~1.5B params) produces noticeably better audio but requires more RAM.

---

## Usage

1. Type a description in the text box — e.g. *"melancholic lo-fi piano, light rain in background"*
2. Optionally pick a quick-prompt preset to get started
3. Adjust duration, model, and temperature in the sidebar
4. Hit **Generate Music** and wait ~30–90 seconds
5. Play directly in the browser or download the `.wav` file

**Temperature guide:**
- `0.5–0.8` → more conservative, repetitive structure
- `1.0` → model default, balanced
- `1.5–2.0` → experimental, more varied but potentially incoherent

---

## Prompt tips

MusicGen responds well to specific, layered descriptions:

| Instead of... | Try... |
|---|---|
| "happy music" | "upbeat indie pop, 115 BPM, major key, acoustic guitar, punchy snare" |
| "sad piano" | "melancholic solo piano, minor key, 65 BPM, slow tempo, long reverb tail" |
| "movie music" | "cinematic orchestral, strings and brass, dramatic dynamics, 90 BPM" |
| "lo-fi beats" | "lo-fi hip hop, vinyl crackle, 85 BPM, muted Rhodes piano, lazy drums" |

---

## Model

This project uses [`facebook/musicgen-small`](https://huggingface.co/facebook/musicgen-small) and [`facebook/musicgen-medium`](https://huggingface.co/facebook/musicgen-medium) from HuggingFace.

MusicGen is an autoregressive transformer model trained by Meta AI on licensed music. Paper: [Simple and Controllable Music Generation (2023)](https://arxiv.org/abs/2306.05284).

---

## Possible extensions

- [ ] Melody conditioning (MusicGen supports a reference audio input)
- [ ] Batch generation with prompt variations for comparison
- [ ] LLM-based prompt expansion instead of rule-based (e.g. call Claude/GPT to rewrite the prompt)
- [ ] History panel with past generations in the session
- [ ] Waveform visualization using `librosa` + `matplotlib`

---

## License

MIT
