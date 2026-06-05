"""
tests/test_prompt_engine.py

Unit tests for the PromptEnhancer class.

Run with: pytest tests/
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from prompt_engine import PromptEnhancer


@pytest.fixture
def enhancer():
    return PromptEnhancer()


class TestPromptEnhancement:

    def test_enhance_adds_content(self, enhancer):
        """Enhanced prompt should always be longer than the raw prompt."""
        raw = "sad piano music"
        enhanced = enhancer.enhance(raw)
        assert len(enhanced) > len(raw)

    def test_enhance_preserves_original_text(self, enhancer):
        """The user's original words should appear in the enhanced output."""
        raw = "melancholic cello piece"
        enhanced = enhancer.enhance(raw)
        assert "melancholic cello piece" in enhanced

    def test_quality_suffix_always_present(self, enhancer):
        """Quality suffix should always be appended regardless of mood detection."""
        raw = "something completely unrecognized xyzabc"
        enhanced = enhancer.enhance(raw)
        assert "high quality" in enhanced

    def test_no_duplicate_bpm_injection(self, enhancer):
        """If user already specified BPM, the enhancer should not inject another BPM."""
        raw = "upbeat track at 130 bpm"
        enhanced = enhancer.enhance(raw)
        bpm_count = enhanced.lower().count("bpm")
        assert bpm_count <= 2, f"Too many BPM mentions: {enhanced}"

    def test_no_duplicate_key_injection(self, enhancer):
        """If user already specified key, the enhancer should not inject another key."""
        raw = "sad song in major key"
        enhanced = enhancer.enhance(raw)
        # Should not add 'minor key' because user said 'major key'
        assert enhanced.count("minor key") <= 1

    def test_extract_tags_returns_list(self, enhancer):
        tags = enhancer.extract_tags("calm piano meditation")
        assert isinstance(tags, list)

    def test_extract_tags_cap(self, enhancer):
        """Tag extraction should cap at 6 tags maximum."""
        long_prompt = "calm sad jazz piano meditation relaxing lo-fi"
        tags = enhancer.extract_tags(long_prompt)
        assert len(tags) <= 6

    def test_extract_tags_bpm_detection(self, enhancer):
        tags = enhancer.extract_tags("energetic track at 128 BPM")
        assert "128 BPM" in tags

    def test_empty_prompt_does_not_crash(self, enhancer):
        """Empty prompt should return something (just the quality suffix)."""
        result = enhancer.enhance("")
        assert "high quality" in result

    def test_known_mood_keywords(self, enhancer):
        """Check that common mood words trigger enhancement."""
        moods_to_test = ["calm", "energetic", "melancholic", "epic", "jazz"]
        for mood in moods_to_test:
            enhanced = enhancer.enhance(mood)
            assert len(enhanced) > len(mood) + 20, f"Mood '{mood}' did not trigger enhancement"
