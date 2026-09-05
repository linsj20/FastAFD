import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from minisgl.distributed.info import DistributedInfo
from minisgl.models.config import ModelConfig, RotaryConfig
from minisgl.models.qwen35_moe_afd import (
    Qwen3_5AttnOnlyAttentionStage,
    Qwen3_5AttnOnlyForCausalLM,
)
from minisgl.utils import torch_dtype


def _config() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        num_qo_heads=32,
        num_kv_heads=2,
        head_dim=256,
        hidden_size=4096,
        vocab_size=248320,
        intermediate_size=0,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(256, 64, 64, 10_000_000.0, None),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=1024,
        norm_topk_prob=True,
        use_routing_bias=False,
        scoring_func="softmax",
        model_type="qwen3_5_moe_text",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        layer_types=("linear_attention", "full_attention"),
        attn_output_gate=True,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=64,
        shared_expert_intermediate_size=1024,
    )


class Qwen35AttnOnlyStructureTest(unittest.TestCase):
    def test_linear_attention_weights_exist_only_on_attention_stage(self) -> None:
        with (
            patch(
                "minisgl.distributed.info._TP_INFO",
                DistributedInfo(rank=0, size=1),
            ),
            patch(
                "minisgl.distributed.info._EP_INFO",
                DistributedInfo(rank=0, size=4),
            ),
            patch(
                "minisgl.core._GLOBAL_CTX",
                SimpleNamespace(
                    server_args=SimpleNamespace(
                        afd_moe_runner_backend="auto",
                        afd_moe_a2a_backend="none",
                    )
                ),
            ),
            patch("torch.cuda.is_available", return_value=False),
            patch("minisgl.layers.rotary._ROPE_DEVICE", torch.device("cpu")),
            torch.device("meta"),
            torch_dtype(torch.bfloat16),
        ):
            attention_keys = Qwen3_5AttnOnlyAttentionStage(_config()).state_dict()
            model_keys = Qwen3_5AttnOnlyForCausalLM(_config()).state_dict()

        self.assertTrue(any(".linear_attn." in key for key in attention_keys))
        self.assertFalse(any(".self_attn." in key for key in attention_keys))
        self.assertFalse(any(".mlp." in key for key in attention_keys))
        self.assertFalse(any(".linear_attn." in key for key in model_keys))
        self.assertTrue(any(".self_attn.qkv_proj." in key for key in model_keys))
        self.assertTrue(any(".self_attn.o_proj." in key for key in model_keys))
        self.assertTrue(any(".mlp.experts." in key for key in model_keys))


if __name__ == "__main__":
    unittest.main()
