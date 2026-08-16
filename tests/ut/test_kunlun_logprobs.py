from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_module():
    outputs = types.ModuleType("vllm.v1.outputs")

    class LogprobsTensors:
        def __init__(
            self,
            logprob_token_ids,
            logprobs,
            selected_token_ranks,
            cu_num_generated_tokens,
        ):
            self.logprob_token_ids = logprob_token_ids
            self.logprobs = logprobs
            self.selected_token_ranks = selected_token_ranks
            self.cu_num_generated_tokens = cu_num_generated_tokens

    outputs.LogprobsTensors = LogprobsTensors
    saved = sys.modules.get("vllm.v1.outputs")
    sys.modules["vllm.v1.outputs"] = outputs
    try:
        path = (
            Path(__file__).parents[2]
            / "vllm_kunlun/v1/worker/gpu/sample/logprob.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_test_kunlun_logprob", path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is None:
            sys.modules.pop("vllm.v1.outputs", None)
        else:
            sys.modules["vllm.v1.outputs"] = saved


def test_compute_token_logprobs_matches_torch_log_softmax():
    module = _load_module()
    logits = torch.tensor([[1.0, 3.0, -2.0], [4.0, 0.0, 2.0]])
    token_ids = torch.tensor([[1, 0], [2, 1]])

    actual = module.compute_token_logprobs(logits, token_ids)
    expected = logits.log_softmax(dim=-1).gather(1, token_ids)

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


def test_compute_topk_logprobs_returns_sampled_token_and_ranks():
    module = _load_module()
    logits = torch.tensor([[1.0, 3.0, -2.0], [4.0, 0.0, 2.0]])
    sampled = torch.tensor([1, 2])

    output = module.compute_topk_logprobs(logits, 1, sampled)

    assert output.logprob_token_ids.tolist() == [[1, 1], [2, 0]]
    assert output.selected_token_ranks.tolist() == [1, 2]
    assert torch.isfinite(output.logprobs).all()
    expected = logits.log_softmax(dim=-1).gather(
        1, output.logprob_token_ids
    )
    torch.testing.assert_close(output.logprobs, expected)
