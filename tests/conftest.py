from __future__ import annotations

import pytest

from motifgen.tokenizer import MidiTokenizer


@pytest.fixture
def tokenizer() -> MidiTokenizer:
    return MidiTokenizer()

