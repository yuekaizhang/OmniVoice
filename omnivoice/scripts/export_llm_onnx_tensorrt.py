#!/usr/bin/env python3
# Copyright    2026  Nvidia Corp.        (authors: Yuekai Zhang)
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

"""Export the OmniVoice LLM backbone to ONNX and then to a TensorRT engine.

Usage:

    python3 -m omnivoice.scripts.export_llm_onnx_tensorrt \\
        --model k2-fsa/OmniVoice \\
        --output-dir models/omnivoice_trt \\
        --trt-engine-name llm.fp16.plan

The script:
1. Loads the full OmniVoice model via ``from_pretrained``.
2. Wraps ``model.llm`` so it accepts ``(inputs_embeds, attention_mask)`` and
   returns ``hidden_states``.  A 2-D padding mask is converted to a 4-D
   bidirectional float mask inside the ONNX graph.
3. Exports FP32 ONNX with dynamic batch & sequence axes.
4. Generates calibration data from a reference audio/text pair.
5. Runs modelopt autocast to produce a mixed-precision FP16 ONNX model.
6. Builds a TensorRT FP16 engine with optimisation profiles covering
   batch 2–16 (CFG doubles user batch) and sequence 50–1500.

Steps 4-5 can be skipped with ``--skip-autocast``.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from omnivoice.utils.audio import load_audio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ONNX wrapper
# ---------------------------------------------------------------------------


class LLMForONNXExport(nn.Module):
    """Thin wrapper that makes the HuggingFace LLM ONNX-exportable.

    Inputs:
        inputs_embeds : [B, S, H]  float32
        attention_mask : [B, S]    int64  (1 = valid, 0 = pad)

    Output:
        hidden_states : [B, S, H]  float32
    """

    def __init__(self, llm: nn.Module):
        super().__init__()
        self.llm = llm

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 2-D  [B, S] int64 → 4-D bidirectional float mask [B, 1, S, S]
        # 0.0 = attend,  large-negative = don't attend
        key_pad = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        S = inputs_embeds.size(1)
        mask_4d = key_pad.expand(-1, 1, S, S).to(inputs_embeds.dtype)  # [B,1,S,S]
        # Use -65504 (FP16-safe) instead of torch.finfo(fp32).min (-3.4e38)
        # which overflows FP16 and causes NaN in FP16 TRT engines.
        mask_4d = (1.0 - mask_4d) * (-65504.0)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=mask_4d,
            return_dict=True,
        )
        return outputs[0]  # last_hidden_state


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx(
    model: nn.Module,
    hidden_size: int,
    filename: str,
    opset_version: int = 18,
) -> None:
    """Export ``LLMForONNXExport`` to ONNX."""

    seq_len = 200
    inputs_embeds = torch.randn(2, seq_len, hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(2, seq_len, dtype=torch.int64)

    logger.info("Tracing LLM for ONNX export (seq_len=%d) …", seq_len)

    torch.onnx.export(
        model,
        (inputs_embeds, attention_mask),
        filename,
        opset_version=opset_version,
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["hidden_states"],
        dynamic_axes={
            "inputs_embeds": {0: "B", 1: "S"},
            "attention_mask": {0: "B", 1: "S"},
            "hidden_states": {0: "B", 1: "S"},
        },
    )
    logger.info("Exported ONNX → %s", filename)


# ---------------------------------------------------------------------------
# Calibration data generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_calibration_data(
    model,
    text: str,
    ref_text: str,
    ref_audio_path: str,
    num_target_tokens: int = 200,
    output_path: str = "calibration_data.npz",
) -> str:
    """Generate calibration data for modelopt autocast.

    Produces ``inputs_embeds`` [B, S, H] and ``attention_mask`` [B, S] tensors
    representing a realistic CFG inference batch (B=2: conditional + unconditional).
    """

    # Encode reference audio
    sampling_rate = model.feature_extractor.sampling_rate
    ref_wav = load_audio(ref_audio_path, sampling_rate)

    chunk_size = model.audio_tokenizer.config.hop_length
    clip_size = int(ref_wav.size(-1) % chunk_size)
    ref_wav = ref_wav[:, :-clip_size] if clip_size > 0 else ref_wav
    ref_audio_tokens = model.audio_tokenizer.encode(
        ref_wav.unsqueeze(0).to(model.audio_tokenizer.device),
    ).audio_codes.squeeze(0)  # (C, T)

    # Conditional inputs
    cond = model._prepare_inference_inputs(
        text=text,
        num_target_tokens=num_target_tokens,
        ref_text=ref_text,
        ref_audio_tokens=ref_audio_tokens,
    )
    cond_input_ids = cond["input_ids"]    # [1, C, S_cond]
    cond_audio_mask = cond["audio_mask"]  # [1, S_cond]

    # Unconditional inputs (target tokens only, no conditioning)
    uncond_input_ids = cond_input_ids[:, :, -num_target_tokens:]
    uncond_audio_mask = cond_audio_mask[:, -num_target_tokens:]

    # Get embeddings
    cond_embeds = model._prepare_embed_inputs(cond_input_ids, cond_audio_mask)
    uncond_embeds = model._prepare_embed_inputs(uncond_input_ids, uncond_audio_mask)

    S_cond = cond_embeds.size(1)
    S_uncond = uncond_embeds.size(1)
    S = max(S_cond, S_uncond)
    H = cond_embeds.size(2)

    # Pad and stack into CFG batch [2, S, H]
    batch_embeds = torch.zeros(2, S, H, dtype=torch.float32)
    batch_mask = torch.zeros(2, S, dtype=torch.int64)

    batch_embeds[0, :S_cond] = cond_embeds[0]
    batch_mask[0, :S_cond] = 1

    batch_embeds[1, :S_uncond] = uncond_embeds[0]
    batch_mask[1, :S_uncond] = 1

    np.savez(
        output_path,
        inputs_embeds=batch_embeds.numpy(),
        attention_mask=batch_mask.numpy(),
    )
    logger.info(
        "Saved calibration data → %s  (inputs_embeds=%s, attention_mask=%s)",
        output_path,
        list(batch_embeds.shape),
        list(batch_mask.shape),
    )
    return output_path


# ---------------------------------------------------------------------------
# modelopt autocast (FP32 → FP16 ONNX)
# ---------------------------------------------------------------------------


def autocast_onnx(
    onnx_path: str,
    output_path: str,
    calibration_data: str = None,
    low_precision_type: str = "fp16",
    data_max: float = 512,
) -> None:
    """Run modelopt autocast to produce a mixed-precision ONNX model."""
    import onnx
    from modelopt.onnx.autocast.convert import convert_to_mixed_precision

    logger.info(
        "Running modelopt autocast (%s, data_max=%s) …", low_precision_type, data_max
    )
    model = convert_to_mixed_precision(
        onnx_path=onnx_path,
        low_precision_type=low_precision_type,
        calibration_data=calibration_data,
        data_max=data_max,
    )
    onnx.save(model, output_path)
    logger.info("Saved mixed-precision ONNX → %s", output_path)


# ---------------------------------------------------------------------------
# TensorRT conversion
# ---------------------------------------------------------------------------


def get_trt_profiles(
    hidden_size: int,
    min_batch: int = 2,
    opt_batch: int = 8,
    max_batch: int = 16,
    min_seq: int = 50,
    opt_seq: int = 400,
    max_seq: int = 1500,
) -> Dict:
    return {
        "input_names": ["inputs_embeds", "attention_mask"],
        "min_shape": [
            (min_batch, min_seq, hidden_size),
            (min_batch, min_seq),
        ],
        "opt_shape": [
            (opt_batch, opt_seq, hidden_size),
            (opt_batch, opt_seq),
        ],
        "max_shape": [
            (max_batch, max_seq, hidden_size),
            (max_batch, max_seq),
        ],
    }


def _apply_mixed_precision(network, config, trt, level: int) -> None:
    """Apply layer-level FP32 precision constraints to a TRT network.

    Levels:
        1 — Non-MatMul float ops → FP32 (Softmax, LayerNorm, ElementWise, Unary).
            All MatMul stays in low precision.  (~312 / 11804 layers)
        2 — Level 1  +  attention MatMul → FP32.
            Only FFN MatMul (mlp/ffn keywords in name) stays in low precision.
        3 — ALL layers → FP32 (essentially FP32 compute with BF16 I/O).
    """
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    # Layer types that are integer/shape ops — never touch these
    _SKIP_TYPES = set()
    for name in (
        "SHAPE", "CONSTANT", "IDENTITY", "GATHER", "SLICE",
        "SHUFFLE", "CONCATENATION", "FILL", "CONDITION", "ASSERTION",
        "CONDITIONAL_INPUT", "CONDITIONAL_OUTPUT",
    ):
        if hasattr(trt.LayerType, name):
            _SKIP_TYPES.add(getattr(trt.LayerType, name))

    # Float compute layers that should always be FP32 at level >= 1
    _ALWAYS_FP32_TYPES = {trt.LayerType.SOFTMAX, trt.LayerType.REDUCE}
    if hasattr(trt.LayerType, "NORMALIZATION"):
        _ALWAYS_FP32_TYPES.add(trt.LayerType.NORMALIZATION)

    _ARITHMETIC_OPS = {
        trt.ElementWiseOperation.SUM,
        trt.ElementWiseOperation.SUB,
        trt.ElementWiseOperation.PROD,
        trt.ElementWiseOperation.DIV,
        trt.ElementWiseOperation.POW,
    }

    # FFN keywords — at level 2 these MatMul layers stay in low precision
    _FFN_KW = {"mlp", "feed_forward", "ffn", "gate_proj", "up_proj",
               "down_proj", "fc1", "fc2", "w1", "w2", "w3"}

    fp32_count = 0
    bf16_matmul_names = []

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        ltype = layer.type
        lname = layer.name or ""

        if ltype in _SKIP_TYPES:
            continue

        # --- MatMul / Conv ---
        if ltype in (trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.CONVOLUTION):
            if level >= 3:
                # Level 3: ALL matmul → FP32
                layer.precision = trt.DataType.FLOAT
                fp32_count += 1
            elif level >= 2:
                # Level 2: FFN matmul stays BF16, rest → FP32
                is_ffn = any(kw in lname.lower() for kw in _FFN_KW)
                if is_ffn:
                    bf16_matmul_names.append(lname)
                else:
                    layer.precision = trt.DataType.FLOAT
                    fp32_count += 1
            else:
                # Level 1: all matmul stays in low precision
                bf16_matmul_names.append(lname)
            continue

        # --- Always-FP32 types (Softmax, Reduce, Normalization) ---
        if ltype in _ALWAYS_FP32_TYPES:
            layer.precision = trt.DataType.FLOAT
            fp32_count += 1
            continue

        # --- ElementWise: only arithmetic ops → FP32 ---
        if ltype == trt.LayerType.ELEMENTWISE:
            try:
                if layer.op in _ARITHMETIC_OPS:
                    layer.precision = trt.DataType.FLOAT
                    fp32_count += 1
            except Exception:
                pass
            continue

        # --- Unary (sqrt, rsqrt, exp, …) ---
        if ltype == trt.LayerType.UNARY:
            try:
                layer.precision = trt.DataType.FLOAT
                fp32_count += 1
            except Exception:
                pass
            continue

        # --- Level 3: force remaining float layers too ---
        if level >= 3:
            try:
                layer.precision = trt.DataType.FLOAT
                fp32_count += 1
            except Exception:
                pass

    logger.info(
        "Mixed-precision level %d: %d / %d layers → FP32, "
        "%d MatMul/Conv kept in low precision",
        level, fp32_count, network.num_layers, len(bf16_matmul_names),
    )
    if bf16_matmul_names and level >= 2:
        logger.info(
            "Low-precision MatMul layers (FFN): %s",
            bf16_matmul_names[:10],
        )


def _apply_onnx_mixed_precision(onnx_path: str, output_path: str) -> None:
    """Modify an FP32 ONNX graph so that FFN MatMul ops use BF16 weights.

    For each FFN MatMul (gate_proj, up_proj, down_proj):
      - Convert the weight initializer from FP32 to BF16
      - Insert Cast(to=BFLOAT16) before the activation input
      - Insert Cast(to=FLOAT) after the MatMul output

    This gives TensorRT an explicit mixed-precision signal at the ONNX level,
    avoiding the global BF16 builder flag that quantises *all* weights.
    """
    import onnx
    import onnx_graphsurgeon as gs
    from ml_dtypes import bfloat16

    logger.info("Applying ONNX-level mixed precision to FFN MatMul nodes …")
    graph = gs.import_onnx(onnx.load(onnx_path))

    _FFN_KW = {"gate_proj", "up_proj", "down_proj"}
    modified = 0

    for node in list(graph.nodes):
        if node.op != "MatMul":
            continue
        # Match FFN MatMul by checking if any FFN keyword appears in the node name
        if not any(kw in node.name for kw in _FFN_KW):
            continue

        # Identify weight (Constant) and activation inputs
        weight_idx = None
        act_idx = None
        for i, inp in enumerate(node.inputs):
            if isinstance(inp, gs.Constant) and inp.values is not None:
                weight_idx = i
            else:
                act_idx = i
        if weight_idx is None or act_idx is None:
            logger.warning("Skipping %s: cannot identify weight/activation inputs", node.name)
            continue

        weight_tensor = node.inputs[weight_idx]
        act_tensor = node.inputs[act_idx]

        # 1. Convert weight from FP32 to BF16
        fp32_vals = weight_tensor.values
        bf16_vals = fp32_vals.astype(bfloat16)
        weight_bf16 = gs.Constant(
            name=weight_tensor.name + "_bf16",
            values=bf16_vals,
        )

        # 2. Insert Cast(to=BFLOAT16) on the activation input
        act_bf16 = gs.Variable(
            name=act_tensor.name + f"_bf16_{node.name}",
            dtype=None,
        )
        cast_to_bf16 = gs.Node(
            op="Cast",
            name=f"{node.name}_cast_act_to_bf16",
            inputs=[act_tensor],
            outputs=[act_bf16],
            attrs={"to": onnx.TensorProto.BFLOAT16},
        )

        # 3. The MatMul now operates in BF16 — create a new BF16 output
        matmul_out_bf16 = gs.Variable(
            name=node.outputs[0].name + "_bf16",
            dtype=None,
        )
        original_output = node.outputs[0]

        # 4. Insert Cast(to=FLOAT) after MatMul
        cast_to_fp32 = gs.Node(
            op="Cast",
            name=f"{node.name}_cast_out_to_fp32",
            inputs=[matmul_out_bf16],
            outputs=[original_output],
            attrs={"to": onnx.TensorProto.FLOAT},
        )

        # Rewire: MatMul inputs become [act_bf16, weight_bf16] (preserve order)
        if act_idx == 0:
            node.inputs = [act_bf16, weight_bf16]
        else:
            node.inputs = [weight_bf16, act_bf16]
        # MatMul output becomes the BF16 intermediate
        node.outputs = [matmul_out_bf16]

        # Insert new nodes into the graph
        graph.nodes.append(cast_to_bf16)
        graph.nodes.append(cast_to_fp32)
        modified += 1

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output_path)
    logger.info(
        "ONNX mixed precision: modified %d FFN MatMul nodes → %s",
        modified, output_path,
    )


def convert_onnx_to_trt(
    trt_path: str,
    onnx_path: str,
    profiles: Dict,
    dtype: torch.dtype = torch.float16,
    mixed_precision: int = 0,
) -> None:
    import tensorrt as trt

    logger.info("Converting ONNX → TRT …")
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()

    # Level 3: no BF16/FP16 flag — pure FP32 compute/storage.
    # Level 4: BF16 flag ON — FFN weights are already BF16 in the ONNX graph,
    #          and we apply level 2 constraints (attention FP32) via TRT API.
    if mixed_precision == 3:
        logger.info(
            "mixed-precision level 3: skipping %s builder flag (all layers FP32)",
            dtype,
        )
    elif mixed_precision == 4:
        if dtype == torch.bfloat16:
            config.set_flag(trt.BuilderFlag.BF16)
        elif dtype == torch.float16:
            config.set_flag(trt.BuilderFlag.FP16)
        logger.info(
            "mixed-precision level 4: %s builder flag ON + ONNX-level FFN BF16 + "
            "level 2 TRT constraints (attention FP32)",
            dtype,
        )
    elif dtype == torch.float16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif dtype == torch.bfloat16:
        config.set_flag(trt.BuilderFlag.BF16)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX model {onnx_path}")

    if mixed_precision == 4:
        # Level 4: apply level 2 TRT constraints (attention FP32, FFN unconstrained)
        _apply_mixed_precision(network, config, trt, level=2)
    elif mixed_precision > 0 and dtype in (torch.float16, torch.bfloat16):
        _apply_mixed_precision(network, config, trt, level=mixed_precision)

    profile = builder.create_optimization_profile()
    for idx, name in enumerate(profiles["input_names"]):
        profile.set_shape(
            name,
            profiles["min_shape"][idx],
            profiles["opt_shape"][idx],
            profiles["max_shape"][idx],
        )
    config.add_optimization_profile(profile)

    # Set I/O dtypes — use FP32 I/O when mixed-precision is enabled
    # to avoid BF16/FP16 quantization of inputs_embeds and hidden_states.
    if mixed_precision > 0:
        io_dtype = trt.DataType.FLOAT
    elif dtype == torch.float16:
        io_dtype = trt.DataType.HALF
    elif dtype == torch.bfloat16:
        io_dtype = trt.DataType.BF16
    else:
        io_dtype = trt.DataType.FLOAT

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if inp.name == "attention_mask":
            inp.dtype = trt.DataType.INT64
        else:
            inp.dtype = io_dtype
    for i in range(network.num_outputs):
        network.get_output(i).dtype = io_dtype

    engine_bytes = builder.build_serialized_network(network, config)
    assert engine_bytes is not None, "TRT engine build failed"

    with open(trt_path, "wb") as f:
        f.write(engine_bytes)
    logger.info("Saved TRT engine → %s", trt_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export OmniVoice LLM to ONNX + TensorRT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="models/omnivoice_trt",
        help="Directory to save ONNX and TRT files.",
    )
    p.add_argument(
        "--trt-engine-name",
        type=str,
        default="llm.fp16.plan",
        help="Filename for the TensorRT engine.",
    )
    p.add_argument("--opset-version", type=int, default=18)
    p.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Precision for the TRT engine.",
    )
    p.add_argument("--min-batch", type=int, default=2)
    p.add_argument("--opt-batch", type=int, default=8)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--min-seq", type=int, default=50)
    p.add_argument("--opt-seq", type=int, default=400)
    p.add_argument("--max-seq", type=int, default=1500)

    # Autocast (FP32 → mixed-precision ONNX) options
    p.add_argument(
        "--skip-autocast",
        action="store_true",
        help="Skip modelopt autocast step and build TRT from FP32 ONNX directly.",
    )
    p.add_argument(
        "--autocast-precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Low precision type for modelopt autocast.",
    )
    p.add_argument(
        "--calibration-ref-audio",
        type=str,
        default="prompt_audio.wav",
        help="Reference audio file for calibration data generation.",
    )
    p.add_argument(
        "--calibration-ref-text",
        type=str,
        default="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
        help="Reference text for calibration data generation.",
    )
    p.add_argument(
        "--calibration-text",
        type=str,
        default="身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
        help="Target text for calibration data generation.",
    )
    p.add_argument(
        "--data-max",
        type=float,
        default=512,
        help="data_max parameter for modelopt autocast.",
    )
    p.add_argument(
        "--mixed-precision",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Layer-level FP32 precision constraints. "
        "0: off (pure BF16/FP16). "
        "1: non-MatMul → FP32 (Softmax, LayerNorm, ElementWise). "
        "2: level 1 + attention MatMul → FP32 (only FFN MatMul stays BF16). "
        "3: ALL layers → FP32 (BF16 only for I/O, quality ≈ FP32 TRT). "
        "4: ONNX-level FFN BF16 + BF16 flag + attention FP32 constraints.",
    )
    return p


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    logger.info("Loading OmniVoice from %s …", args.model)
    from omnivoice.models.omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(
        args.model,
        device_map="cpu",
        dtype=torch.float32,
        train=True,  # skip loading audio tokenizer etc.
    )
    model.eval()

    # Manually load tokenizers for calibration data generation
    if not args.skip_autocast:
        from transformers import (
            AutoFeatureExtractor,
            AutoTokenizer,
            HiggsAudioV2TokenizerModel,
        )

        model.text_tokenizer = AutoTokenizer.from_pretrained(args.model)

        if os.path.isdir(args.model):
            resolved_path = args.model
        else:
            from huggingface_hub import snapshot_download

            resolved_path = snapshot_download(args.model)

        audio_tokenizer_path = os.path.join(resolved_path, "audio_tokenizer")
        if not os.path.isdir(audio_tokenizer_path):
            audio_tokenizer_path = "eustlb/higgs-audio-v2-tokenizer"

        model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
            audio_tokenizer_path, device_map="cpu"
        )
        model.feature_extractor = AutoFeatureExtractor.from_pretrained(
            audio_tokenizer_path
        )
        model.sampling_rate = model.feature_extractor.sampling_rate
        logger.info("Loaded text_tokenizer + audio_tokenizer for calibration")

    hidden_size = model.config.llm_config.hidden_size
    logger.info("LLM hidden_size = %d", hidden_size)

    # 2. Export FP32 ONNX
    wrapper = LLMForONNXExport(model.llm)
    wrapper.eval()

    onnx_path = str(out_dir / "llm.onnx")
    export_onnx(
        wrapper,
        hidden_size=hidden_size,
        filename=onnx_path,
        opset_version=args.opset_version,
    )

    # 3. Generate calibration data + autocast to FP16 ONNX
    if not args.skip_autocast:
        calibration_path = str(out_dir / "calibration_data.npz")
        generate_calibration_data(
            model,
            text=args.calibration_text,
            ref_text=args.calibration_ref_text,
            ref_audio_path=args.calibration_ref_audio,
            output_path=calibration_path,
        )

        suffix = args.autocast_precision  # "fp16" or "bf16"
        mixed_onnx_path = str(out_dir / f"llm.{suffix}.onnx")
        autocast_onnx(
            onnx_path=onnx_path,
            output_path=mixed_onnx_path,
            calibration_data=calibration_path,
            low_precision_type=args.autocast_precision,
            data_max=args.data_max,
        )
        trt_input_onnx = mixed_onnx_path
    else:
        trt_input_onnx = onnx_path

    # 3b. ONNX-level mixed precision for level 4
    if args.mixed_precision == 4:
        onnx_mp_path = str(out_dir / "llm.onnx_mp4.onnx")
        _apply_onnx_mixed_precision(trt_input_onnx, onnx_mp_path)
        trt_input_onnx = onnx_mp_path

    # 4. Build TRT engine
    profiles = get_trt_profiles(
        hidden_size=hidden_size,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        min_seq=args.min_seq,
        opt_seq=args.opt_seq,
        max_seq=args.max_seq,
    )
    trt_path = str(out_dir / args.trt_engine_name)
    _dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    trt_dtype = _dtype_map[args.dtype]
    convert_onnx_to_trt(
        trt_path, trt_input_onnx, profiles,
        dtype=trt_dtype, mixed_precision=args.mixed_precision,
    )

    logger.info("Done!  TRT engine: %s", trt_path)


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, force=True)
    main()
