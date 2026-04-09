#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data utilities for batch inference and evaluation.

Provides ``read_test_list()`` to parse JSONL test list files used by
``omnivoice.cli.infer_batch`` and evaluation scripts.
"""

import json
import logging
import os
from pathlib import Path


def read_test_list(path):
    """Read a JSONL test list file.

    Each line should be a JSON object with fields:
        id, text, ref_audio, ref_text, language_id, language_name, duration, speed

    language_id, language_name, duration, and speed are optional (default to None).

    Returns a list of dicts.
    """
    path = Path(path)
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON at line {line_no}: {line}")
                continue

            sample = {
                "id": obj.get("id"),
                "text": obj.get("text"),
                "ref_audio": obj.get("ref_audio"),
                "ref_text": obj.get("ref_text"),
                "language_id": obj.get("language_id"),
                "language_name": obj.get("language_name"),
                "duration": obj.get("duration"),
                "speed": obj.get("speed"),
            }
            samples.append(sample)
    return samples


def read_hf_dataset(dataset_name, split, audio_cache_dir):
    """Load a HuggingFace dataset and convert to the same format as read_test_list.

    Expected dataset fields:
        id, prompt_text, prompt_audio (with array + sampling_rate), target_text.

    Prompt audio arrays are saved as WAV files under ``audio_cache_dir`` so the
    existing batch inference pipeline can load them by path.

    Returns a list of dicts matching the ``read_test_list`` schema.
    """
    import numpy as np
    import torch
    import torchaudio
    from datasets import load_dataset

    logging.info(
        f"Loading HuggingFace dataset '{dataset_name}' split='{split}' ..."
    )
    dataset = load_dataset(dataset_name, split=split)
    logging.info(f"Loaded {len(dataset)} samples from HF dataset")

    os.makedirs(audio_cache_dir, exist_ok=True)

    samples = []
    for item in dataset:
        sample_id = item["id"]
        ref_text = item["prompt_text"]
        target_text = item["target_text"]

        # Save prompt audio to wav file
        audio_array = np.asarray(item["prompt_audio"]["array"], dtype=np.float32)
        sr = item["prompt_audio"]["sampling_rate"]
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
        audio_path = os.path.join(audio_cache_dir, f"{sample_id}.wav")
        torchaudio.save(audio_path, audio_tensor, sr)

        samples.append(
            {
                "id": sample_id,
                "text": target_text,
                "ref_audio": audio_path,
                "ref_text": ref_text,
                "language_id": None,
                "language_name": None,
                "duration": None,
                "speed": None,
            }
        )

    return samples
