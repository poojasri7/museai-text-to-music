"""
model_wrapper.py

A thin wrapper around HuggingFace's MusicGen implementation.

Why a wrapper instead of calling the pipeline directly in app.py?
- Keeps model loading logic separate from UI code
- Enables lazy loading (model stays in memory across Streamlit reruns via cache)
- Makes it easy to swap in a different model later (MusicLM, AudioCraft, etc.)
- Easier to unit-test generation logic in isolation
"""

import torch
import numpy as np
from typing import Tuple
import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_model(model_name: str):
    """
    Load and cache the model + processor.

    Using st.cache_resource so the model stays loaded between Streamlit
    reruns — avoids reloading ~1GB weights every time a user changes a
    slider. This is a common pattern in production Streamlit ML apps.
    """
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # Inference mode — disables dropout, batch norm tracking

    return processor, model, device


class MusicGenWrapper:
    """
    Wraps facebook/musicgen-* models from HuggingFace.

    Handles:
    - Device detection (CUDA vs CPU fallback)
    - Token count calculation from desired duration
    - Tensor construction and cleanup
    - Output conversion to numpy for scipy/st.audio compatibility

    MusicGen generates audio at 32kHz. The model's frame rate is 50 tokens/sec,
    so to get N seconds of audio: max_new_tokens = N * 50 (+ small buffer).
    """

    FRAME_RATE = 50  # tokens per second, fixed in MusicGen's architecture
    SAMPLE_RATE = 32000  # Hz — MusicGen always outputs 32kHz

    def __init__(self, model_name: str = "facebook/musicgen-small"):
        self.model_name = model_name
        self.processor, self.model, self.device = _load_model(model_name)

    def generate(
        self,
        prompt: str,
        duration: int = 10,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from a text prompt.

        Args:
            prompt: Enhanced text description of the music
            duration: Target length in seconds (approximate — actual output
                      may vary slightly based on token boundaries)
            temperature: Sampling temperature. 1.0 is the model's default.
                         Lower = more conservative/repetitive.
                         Higher = more creative but potentially incoherent.
            guidance_scale: Classifier-free guidance scale. Higher = the model
                            follows the text prompt more strictly. 3.0 is a
                            reasonable default per the MusicGen paper.

        Returns:
            (audio_array, sample_rate) where audio_array is int16 numpy array
            ready for scipy.io.wavfile.write or st.audio
        """
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # MusicGen's token rate is fixed at 50/sec regardless of model size
        max_new_tokens = int(duration * self.FRAME_RATE * 1.05)  # 5% buffer

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                guidance_scale=guidance_scale,
            )

        # audio_values shape: (batch, channels, samples)
        # We take batch[0], channel[0] since we generate one clip
        audio_np = audio_values[0, 0].cpu().numpy()

        # Normalize to [-1, 1] float range, then convert to int16
        # (required by scipy.io.wavfile for standard WAV output)
        audio_np = audio_np / np.abs(audio_np).max()
        audio_int16 = (audio_np * 32767).astype(np.int16)

        return audio_int16, self.SAMPLE_RATE

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    def __repr__(self):
        return f"MusicGenWrapper(model={self.model_name}, device={self.device})"
