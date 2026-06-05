"""
prompt_engine.py

Prompt enhancement layer for MusicGen.

MusicGen (like most audio generation models) responds much better to
structured, music-theory-aware descriptions than to casual phrases.
This module bridges the gap between what a user types and what the
model actually needs to produce good output.

Design decisions:
- Rule-based enrichment first (fast, deterministic, no extra model needed)
- Mood → instrumentation mappings built from MusicGen paper's examples
- Keeps user's original intent intact; only adds vocabulary, doesn't replace
"""

import re
from typing import List, Optional


# Maps loose mood words to instrumentation + production descriptors
# that MusicGen's training data likely associates with those moods.
MOOD_MAP = {
    # Calm / introspective
    "calm": "ambient, soft pads, reverb-heavy, 70 BPM, minimalist",
    "relaxing": "ambient, slow tempo, soft piano, gentle textures, 65 BPM",
    "meditation": "drone, sine waves, tibetan bowl, 60 BPM, no percussion",
    "peaceful": "acoustic guitar fingerpicking, light reverb, 72 BPM, major key",
    "sleep": "ambient drone, soft strings, 55 BPM, no percussion, very slow attack",

    # Energetic
    "energetic": "120 BPM, driving rhythm, punchy drums, bright mixdown",
    "upbeat": "115 BPM, major key, bright piano chords, punchy snare",
    "hype": "trap hi-hats, 140 BPM, 808 bass, reverb on snare",
    "workout": "130 BPM, electronic, compressed drums, synth stabs",

    # Melancholic / emotional
    "sad": "minor key, slow tempo, 60 BPM, sparse piano, long reverb tail",
    "melancholic": "minor key, 70 BPM, cello, muted piano, rainy atmosphere",
    "nostalgic": "lo-fi, tape saturation, vinyl crackle, 80 BPM, warm chords",
    "emotional": "orchestral strings, slow tempo, minor key, dynamics",

    # Cinematic
    "epic": "orchestral, brass section, timpani, 110 BPM, dramatic swells",
    "cinematic": "orchestral, dynamic range, strings, subtle brass, 80 BPM",
    "tense": "staccato strings, 90 BPM, minor key, sparse, building tension",
    "dark": "minor key, low register, slow, heavy reverb, ominous",

    # Genre-adjacent
    "jazz": "upright bass, brush drums, walking bass line, 110 BPM, swing feel",
    "lo-fi": "lo-fi, vinyl crackle, tape saturation, muted chords, 85 BPM",
    "ambient": "ambient, reverb-heavy, slow attack, pads, 60 BPM, no sharp transients",
    "classical": "orchestral, no drums, dynamic contrast, concert hall reverb",
    "electronic": "synthesizer, four-on-the-floor kick, arpeggiated bass, 120 BPM",
}

# Maps instruments to production context so MusicGen knows their role
INSTRUMENT_CONTEXT = {
    "piano": "acoustic grand piano, studio recording, dynamic range",
    "guitar": "acoustic guitar, natural room reverb, fingerpicking or strummed",
    "electric guitar": "electric guitar, tube amp warmth, light overdrive",
    "violin": "solo violin, vibrato, concert hall reverb",
    "strings": "string quartet, lush, orchestral reverb",
    "drums": "acoustic drum kit, punchy kick, snappy snare",
    "bass": "electric bass, smooth low end, played in pocket",
    "synth": "analog synthesizer, warm oscillators, slight detune",
    "flute": "concert flute, airy, pastoral, light reverb",
    "trumpet": "solo trumpet, bright attack, jazz or orchestral context",
}

# Tags that should propagate directly into the enhanced prompt
PASSTHROUGH_TAGS = [
    "BPM", "minor key", "major key", "4/4", "3/4", "waltz",
    "no vocals", "instrumental", "solo", "duo", "ensemble",
    "acoustic", "electric", "unplugged", "live recording",
]


class PromptEnhancer:
    """
    Transforms a casual user prompt into a structured music description.

    The enhancement strategy:
    1. Detect mood keywords and inject corresponding musical vocabulary
    2. Detect instrument mentions and add production context
    3. Add a quality suffix (always helps model quality)
    4. Preserve user's original phrasing where possible

    This is intentionally rule-based rather than LLM-based because:
    - No extra model or API cost
    - Deterministic — user can trust the same input → similar output
    - Transparent — easy to audit and extend the mappings
    - Fast — zero latency enhancement step
    """

    def __init__(self):
        self.mood_map = MOOD_MAP
        self.instrument_context = INSTRUMENT_CONTEXT

    def enhance(self, raw_prompt: str) -> str:
        prompt_lower = raw_prompt.lower()
        additions = []

        # Step 1: Mood enrichment
        for keyword, descriptor in self.mood_map.items():
            if keyword in prompt_lower:
                additions.append(descriptor)
                break  # Only apply the first match to avoid contradictions

        # Step 2: Instrument context injection
        for instrument, context in self.instrument_context.items():
            if instrument in prompt_lower:
                additions.append(context)

        # Step 3: Avoid duplicate info already in the raw prompt
        filtered = []
        for add in additions:
            # Don't inject BPM if user already mentioned one
            if "BPM" in add and re.search(r'\d+\s*bpm', prompt_lower):
                continue
            # Don't inject key info if user already specified
            if "key" in add and ("minor" in prompt_lower or "major" in prompt_lower):
                continue
            filtered.append(add)

        # Step 4: Assemble enhanced prompt
        if filtered:
            enhancement_str = ", ".join(filtered)
            enhanced = f"{raw_prompt.strip()}, {enhancement_str}"
        else:
            enhanced = raw_prompt.strip()

        # Step 5: Quality suffix — MusicGen responds well to production quality cues
        quality_suffix = "high quality, professional recording, well-mixed"
        enhanced = f"{enhanced}, {quality_suffix}"

        return enhanced

    def extract_tags(self, raw_prompt: str) -> List[str]:
        """
        Pull out short display tags from the raw prompt for the UI.
        These are shown as pills to give the user quick feedback on
        what the system detected from their input.
        """
        tags = []
        prompt_lower = raw_prompt.lower()

        for keyword in self.mood_map:
            if keyword in prompt_lower:
                tags.append(keyword)

        for instrument in self.instrument_context:
            if instrument in prompt_lower:
                tags.append(instrument)

        # BPM detection
        bpm_match = re.search(r'(\d+)\s*bpm', prompt_lower)
        if bpm_match:
            tags.append(f"{bpm_match.group(1)} BPM")

        # Key detection
        if "minor" in prompt_lower:
            tags.append("minor key")
        if "major" in prompt_lower:
            tags.append("major key")

        return list(set(tags))[:6]  # Cap at 6 pills for UI cleanliness
