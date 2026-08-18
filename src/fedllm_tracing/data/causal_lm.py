"""Teacher-forced causal-LM examples with prompt labels masked from loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class TeacherForcedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.input_ids or len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels must be non-empty and equally sized")


def make_teacher_forcing_example(
    prompt_ids: Sequence[int], target_ids: Sequence[int]
) -> TeacherForcedExample:
    if not prompt_ids or not target_ids:
        raise ValueError("Both prompt_ids and target_ids are required")
    input_ids = tuple(int(token) for token in (*prompt_ids, *target_ids))
    labels = tuple([-100] * len(prompt_ids) + [int(token) for token in target_ids])
    return TeacherForcedExample(input_ids, labels)


def encode_teacher_forcing(
    tokenizer: Any,
    prompt: str,
    target: str,
    *,
    max_length: int | None = None,
) -> TeacherForcedExample:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    example = make_teacher_forcing_example(prompt_ids, target_ids)
    if max_length is not None and len(example.input_ids) > max_length:
        raise ValueError("Encoded example exceeds max_length; implicit truncation is disabled")
    return example


def collate_teacher_forcing(
    examples: Sequence[TeacherForcedExample], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    max_length = max(len(example.input_ids) for example in examples)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for example in examples:
        padding = max_length - len(example.input_ids)
        input_rows.append(list(example.input_ids) + [pad_token_id] * padding)
        label_rows.append(list(example.labels) + [-100] * padding)
        mask_rows.append([1] * len(example.input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
    }

