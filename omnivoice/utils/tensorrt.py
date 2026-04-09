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

"""TensorRT runtime utilities for OmniVoice LLM backbone."""

import logging
import os
import queue
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class _TrtConfigProxy:
    """Minimal proxy so ``getattr(model.llm.config, ...)`` doesn't crash."""

    _attn_implementation = "trt"


class TrtLLMWrapper:
    """Drop-in replacement for ``model.llm`` that runs TensorRT inference.

    The wrapper matches the calling convention used in
    ``OmniVoice.forward()``:

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        hidden_states = llm_outputs[0]

    So ``__call__`` returns a tuple whose first element is the hidden-state
    tensor.
    """

    _is_trt = True

    _TRT_TO_TORCH = {
        "FLOAT": torch.float32,
        "HALF": torch.float16,
        "BF16": torch.bfloat16,
    }

    def __init__(
        self,
        trt_engine: Any,
        hidden_size: int,
        trt_concurrent: int = 1,
        device: str = "cuda:0",
        input_embeddings: Optional[torch.nn.Module] = None,
    ):
        self.trt_context_pool: queue.Queue = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        self.hidden_size = hidden_size
        self.device = device
        self.config = _TrtConfigProxy()
        self._input_embeddings = input_embeddings

        # Detect I/O dtype from the engine itself
        trt_dt = str(trt_engine.get_tensor_dtype("inputs_embeds")).split(".")[-1]
        self._io_dtype = self._TRT_TO_TORCH.get(trt_dt, torch.float16)
        logger.info("TRT engine I/O dtype: %s → torch %s", trt_dt, self._io_dtype)

        for _ in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            trt_stream = torch.cuda.stream(torch.cuda.Stream(torch.device(device)))
            assert trt_context is not None, (
                "Failed to create TRT context — possibly not enough GPU memory. "
                f"Try reducing trt_concurrent (currently {trt_concurrent})."
            )
            self.trt_context_pool.put((trt_context, trt_stream))

        assert not self.trt_context_pool.empty(), "No available TRT context."

    # ------------------------------------------------------------------

    def get_input_embeddings(self) -> Optional[torch.nn.Module]:
        return self._input_embeddings

    # ------------------------------------------------------------------

    def _acquire(self) -> Tuple[Any, Any]:
        return self.trt_context_pool.get()

    def _release(self, context: Any, stream: Any) -> None:
        self.trt_context_pool.put((context, stream))

    # ------------------------------------------------------------------

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        """Run the TRT engine.

        Args:
            inputs_embeds: ``[B, S, H]`` float tensor.
            attention_mask: ``[B, S]`` bool/int tensor (True/1 = valid).
            **kwargs: Ignored (``return_dict``, ``position_ids``, …).

        Returns:
            ``(hidden_states,)`` — a 1-tuple so that ``result[0]`` gives the
            hidden-state tensor of shape ``[B, S, H]``.
        """
        B, S, H = inputs_embeds.shape

        # Build 2-D int mask  [B, S]  (1 = valid, 0 = pad)
        if attention_mask is None:
            mask_2d = torch.ones(B, S, dtype=torch.int64, device=inputs_embeds.device)
        elif attention_mask.dim() == 2:
            mask_2d = attention_mask.to(torch.int64)
        else:
            # 4-D bool mask [B, 1, S, S] — collapse to 2-D
            mask_2d = attention_mask[:, 0, 0, :].to(torch.int64)

        inputs_casted = inputs_embeds.contiguous().to(self._io_dtype)
        mask_2d = mask_2d.contiguous()

        output = torch.empty(B, S, H, dtype=self._io_dtype, device=inputs_embeds.device)

        estimator, stream = self._acquire()
        torch.cuda.current_stream().synchronize()

        with stream:
            estimator.set_input_shape("inputs_embeds", (B, S, H))
            estimator.set_input_shape("attention_mask", (B, S))

            estimator.set_tensor_address(
                "inputs_embeds", inputs_casted.data_ptr()
            )
            estimator.set_tensor_address("attention_mask", mask_2d.data_ptr())

            # Output tensor — last IO tensor
            num_io = self.trt_engine.num_io_tensors
            out_name = self.trt_engine.get_tensor_name(num_io - 1)
            estimator.set_tensor_address(out_name, output.data_ptr())

            ok = estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            assert ok, "TRT execute_async_v3 failed"
            torch.cuda.current_stream().synchronize()

        self._release(estimator, stream)

        return (output.to(inputs_embeds.dtype),)


def load_llm_trt(
    model: Any,
    trt_engine_path: str,
    trt_concurrent: int = 1,
    device: Optional[str] = None,
) -> None:
    """Replace ``model.llm`` with a :class:`TrtLLMWrapper`.

    Args:
        model: An ``OmniVoice`` instance.
        trt_engine_path: Path to the serialised TRT engine (``.plan``).
        trt_concurrent: Number of concurrent TRT execution contexts.
        device: CUDA device string.  Defaults to ``model.device``.
    """
    assert os.path.exists(trt_engine_path), (
        f"TRT engine not found at {trt_engine_path}. "
        "Run export_llm_onnx_tensorrt.py first."
    )

    import tensorrt as trt

    logger.info("Loading TRT engine from %s …", trt_engine_path)
    with open(trt_engine_path, "rb") as f:
        engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(
            f.read()
        )
    assert engine is not None, f"Failed to deserialise TRT engine: {trt_engine_path}"

    hidden_size = model.config.llm_config.hidden_size
    if device is None:
        device = str(model.device)

    input_embeddings = model.llm.get_input_embeddings()
    del model.llm
    model.llm = TrtLLMWrapper(
        engine,
        hidden_size=hidden_size,
        trt_concurrent=trt_concurrent,
        device=device,
        input_embeddings=input_embeddings,
    )
    logger.info("LLM replaced with TRT engine (hidden_size=%d).", hidden_size)
