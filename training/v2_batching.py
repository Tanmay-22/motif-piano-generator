from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from motifgen.v2 import category_id, motif_feature_vector, texture_id
from training.v2_dataset import PhrasePair


@dataclass(frozen=True)
class V2Batch:
    motif_events: torch.Tensor
    motif_padding_mask: torch.Tensor
    decoder_inputs: torch.Tensor
    decoder_targets: torch.Tensor
    decoder_padding_mask: torch.Tensor
    category_ids: torch.Tensor
    texture_ids: torch.Tensor
    motif_controls: torch.Tensor

    def to(self, device: torch.device | str) -> "V2Batch":
        return V2Batch(**{name: value.to(device) for name, value in self.__dict__.items()})


def _pad_event_sequences(sequences: Sequence[Sequence[object]]) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(sequence) for sequence in sequences)
    result = torch.zeros(len(sequences), maximum, 7, dtype=torch.long)
    padding = torch.ones(len(sequences), maximum, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        values = torch.tensor([event.as_tuple() for event in sequence], dtype=torch.long)
        result[row, : len(sequence)] = values
        padding[row, : len(sequence)] = False
    return result, padding


def collate_v2_phrase_pairs(pairs: Sequence[PhrasePair]) -> V2Batch:
    if not pairs:
        raise ValueError("Cannot collate an empty v2 batch.")
    motif_events, motif_padding = _pad_event_sequences([pair.motif_events for pair in pairs])
    decoder_sequences = [pair.continuation_events[:-1] for pair in pairs]
    target_sequences = [pair.continuation_events[1:] for pair in pairs]
    decoder_inputs, decoder_padding = _pad_event_sequences(decoder_sequences)
    decoder_targets, _ = _pad_event_sequences(target_sequences)
    return V2Batch(
        motif_events=motif_events,
        motif_padding_mask=motif_padding,
        decoder_inputs=decoder_inputs,
        decoder_targets=decoder_targets,
        decoder_padding_mask=decoder_padding,
        category_ids=torch.tensor(
            [category_id(pair.conditioning_category) for pair in pairs], dtype=torch.long
        ),
        texture_ids=torch.tensor(
            [texture_id(pair.motif_features.texture) for pair in pairs], dtype=torch.long
        ),
        motif_controls=torch.tensor(
            [motif_feature_vector(pair.motif_features) for pair in pairs], dtype=torch.float32
        ),
    )
