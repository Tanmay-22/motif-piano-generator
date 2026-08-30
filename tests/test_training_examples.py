from __future__ import annotations

from motifgen.config import DataConfig
from training.dataset import IGNORE_INDEX, build_baseline_example, build_conditioned_example


def test_conditioning_prompt_predicts_only_continuation(tokenizer):
    config = DataConfig(motif_min_tokens=2, motif_max_tokens=4, continuation_tokens=3)
    motif = [tokenizer.token_to_id["VEL_20"], tokenizer.token_to_id["NOTE_ON_60"]]
    continuation = [
        tokenizer.token_to_id["TIME_SHIFT_10"],
        tokenizer.token_to_id["NOTE_OFF_60"],
        tokenizer.token_to_id["TIME_SHIFT_20"],
    ]
    inputs, labels, padding = build_conditioned_example(motif, continuation, tokenizer, config)
    separator_position = 1 + config.motif_max_tokens
    assert inputs[0].item() == tokenizer.bos_id
    assert inputs[separator_position].item() == tokenizer.sep_id
    assert labels[:separator_position].tolist() == [IGNORE_INDEX] * separator_position
    assert labels[separator_position:].tolist() == continuation
    assert padding.tolist() == [False, False, False, True, True, False, False, False]


def test_baseline_uses_bos_during_training(tokenizer):
    continuation = [10, 11, 12]
    inputs, labels, padding = build_baseline_example(continuation, tokenizer)
    assert inputs.tolist() == [tokenizer.bos_id, 10, 11]
    assert labels.tolist() == continuation
    assert not padding.any()

