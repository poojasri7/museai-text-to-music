import streamlit as st
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import os
import time
from prompt_engine import PromptEnhancer
from model_wrapper import MusicGenWrapper

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MuseAI — Text to Music",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
        background-color: #0e0e0e;
        color: #e8e4dc;
    }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        font-style: italic;
        color: #e8e4dc;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }

    .subtitle {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        color: #666;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 0;
    }

    .stTextArea textarea {
        background: #1a1a1a !important;
        color: #e8e4dc !important;
        border: 1px solid #333 !important;
        border-radius: 4px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.9rem !important;
    }

    .stButton > button {
        background: #e8e4dc;
        color: #0e0e0e;
        border: none;
        border-radius: 2px;
        font-family: 'DM Mono', monospace;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-size: 0.8rem;
        padding: 0.6rem 2rem;
        transition: all 0.2s;
        width: 100%;
    }

    .stButton > button:hover {
        background: #c9c3b5;
        transform: translateY(-1px);
    }

    .enhanced-prompt-box {
        background: #1a1a1a;
        border-left: 2px solid #5a8a6a;
        padding: 1rem 1.2rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.82rem;
        color: #9aaf9a;
        font-family: 'DM Mono', monospace;
        line-height: 1.6;
        margin: 1rem 0;
    }

    .tag-pill {
        display: inline-block;
        background: #222;
        border: 1px solid #444;
        color: #aaa;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        margin: 0.2rem;
        font-family: 'DM Mono', monospace;
    }

    .section-label {
        font-size: 0.65rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 0.5rem;
    }

    .stSlider [data-testid="stSlider"] {
        accent-color: #5a8a6a;
    }

    .stSelectbox select {
        background: #1a1a1a !important;
        color: #e8e4dc !important;
        border: 1px solid #333 !important;
        font-family: 'DM Mono', monospace !important;
    }

    .stSpinner > div {
        border-top-color: #5a8a6a !important;
    }

    .generation-meta {
        font-size: 0.7rem;
        color: #555;
        font-family: 'DM Mono', monospace;
        text-align: right;
    }

    hr {
        border-color: #222 !important;
    }

    .stSidebar {
        background: #111 !important;
    }

    .stSidebar [data-testid="stSidebarContent"] {
        background: #111;
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">MuseAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">text → music generation  ·  powered by MusicGen</p>', unsafe_allow_html=True)
st.markdown("---")


# ─── Sidebar: Settings ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">Generation Settings</p>', unsafe_allow_html=True)

    duration = st.slider("Duration (seconds)", min_value=5, max_value=30, value=10, step=5)

    model_size = st.selectbox(
        "Model",
        options=["facebook/musicgen-small", "facebook/musicgen-medium"],
        index=0,
        help="Small (~300M params) is faster; Medium (~1.5B) is higher quality"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.5, max_value=2.0, value=1.0, step=0.1,
        help="Higher = more creative/experimental. Lower = more predictable."
    )

    use_prompt_enhancement = st.toggle("Prompt Enhancement", value=True,
        help="Automatically expands your prompt with musical vocabulary MusicGen responds better to")

    st.markdown("---")
    st.markdown('<p class="section-label">About</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.72rem; color: #555; line-height: 1.8;">
    MusicGen by Meta AI<br>
    Runs locally — no API key needed<br>
    CPU / CUDA supported<br>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Input ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Describe your music</p>', unsafe_allow_html=True)

user_prompt = st.text_area(
    label="prompt",
    label_visibility="collapsed",
    placeholder="e.g. 'melancholic lo-fi piano with light rain ambiance' or 'energetic jazz with upright bass'",
    height=90,
)

# Quick-pick presets
st.markdown('<p class="section-label" style="margin-top:0.8rem">Quick prompts</p>', unsafe_allow_html=True)
presets = [
    "Lo-fi study beats", "Epic film score", "Jazz café evening",
    "Ambient meditation", "Upbeat indie pop", "Dark synthwave"
]
preset_cols = st.columns(3)
for i, preset in enumerate(presets):
    with preset_cols[i % 3]:
        if st.button(preset, key=f"preset_{i}"):
            user_prompt = preset
            st.session_state["selected_preset"] = preset

# Use preset if selected
if "selected_preset" in st.session_state and not user_prompt:
    user_prompt = st.session_state["selected_preset"]

st.markdown("---")


# ─── Generate Button ────────────────────────────────────────────────────────
generate_clicked = st.button("Generate Music ↗")

if generate_clicked:
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        enhancer = PromptEnhancer()

        if use_prompt_enhancement:
            enhanced = enhancer.enhance(user_prompt)
            st.markdown('<p class="section-label">Enhanced prompt</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="enhanced-prompt-box">{enhanced}</div>', unsafe_allow_html=True)

            tags = enhancer.extract_tags(user_prompt)
            if tags:
                tag_html = "".join([f'<span class="tag-pill">{t}</span>' for t in tags])
                st.markdown(tag_html, unsafe_allow_html=True)
        else:
            enhanced = user_prompt

        st.markdown("---")

        with st.spinner("Generating... this may take 30–90 seconds depending on your hardware"):
            start_time = time.time()
            try:
                wrapper = MusicGenWrapper(model_name=model_size)
                audio_array, sample_rate = wrapper.generate(
                    prompt=enhanced,
                    duration=duration,
                    temperature=temperature,
                )

                elapsed = time.time() - start_time

                # Save to temp file
                output_path = "generated_music.wav"
                wavfile.write(output_path, sample_rate, audio_array)

                st.markdown('<p class="section-label">Output</p>', unsafe_allow_html=True)
                st.audio(output_path, format="audio/wav")
                st.markdown(
                    f'<div class="generation-meta">generated in {elapsed:.1f}s  ·  {duration}s  ·  {sample_rate}Hz  ·  {model_size.split("/")[-1]}</div>',
                    unsafe_allow_html=True
                )

                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download .wav",
                        data=f,
                        file_name=f"museai_{int(time.time())}.wav",
                        mime="audio/wav",
                    )

            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                st.info("If you're hitting memory errors, try switching to 'musicgen-small' in settings, or reduce duration.")
