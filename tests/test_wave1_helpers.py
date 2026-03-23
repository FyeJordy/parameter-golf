import math
import unittest
from pathlib import Path

import torch

import train_gpt


class Wave1HelperTests(unittest.TestCase):
    def test_pair_hash_embedding_starts_as_noop(self):
        module = train_gpt.PairHashEmbedding(num_buckets=128, embed_dim=8, model_dim=16)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        out = module(input_ids, bos_id=1)
        self.assertEqual(out.shape, (1, 3, 16))
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_make_pair_hash_ids_uses_bos_prefix(self):
        input_ids = torch.tensor([[5, 7, 11]], dtype=torch.int64)
        hashed = train_gpt.make_pair_hash_ids(input_ids, bos_id=1, num_buckets=8192)
        expected = torch.tensor(
            [[(1 * 1009 + 5) % 8192, (5 * 1009 + 7) % 8192, (7 * 1009 + 11) % 8192]],
            dtype=torch.int64,
        )
        self.assertTrue(torch.equal(hashed, expected))

    def test_build_eval_attn_mask_applies_causal_and_bos_reset(self):
        input_ids = torch.tensor([[1, 10, 11, 1, 12]], dtype=torch.int64)
        mask = train_gpt.build_eval_attn_mask(input_ids, bos_id=1, doc_isolated=True)
        self.assertEqual(mask.shape, (1, 1, 5, 5))
        self.assertEqual(mask.dtype, torch.float32)

        self.assertEqual(mask[0, 0, 4, 3].item(), 0.0)
        self.assertTrue(math.isinf(mask[0, 0, 4, 2].item()))
        self.assertTrue(math.isinf(mask[0, 0, 2, 4].item()))

    def test_export_tensor_spec_routes_mixed_lowbit_names(self):
        self.assertEqual(
            train_gpt.export_tensor_spec("blocks.0.mlp.fc.weight", train_gpt.MIXED_LOWBITS_MODE).mode,
            "int5_row",
        )
        self.assertEqual(
            train_gpt.export_tensor_spec("blocks.0.attn.c_q.weight", train_gpt.MIXED_LOWBITS_MODE).mode,
            "int6_row",
        )
        self.assertEqual(
            train_gpt.export_tensor_spec("pair_hash.embed.weight", train_gpt.MIXED_LOWBITS_MODE).mode,
            "fp16_passthrough",
        )
        self.assertEqual(
            train_gpt.export_tensor_spec("pair_hash.pair_alpha", train_gpt.MIXED_LOWBITS_MODE).mode,
            "fp32_passthrough",
        )

    def test_export_codec_roundtrip_supports_zlib_and_zstd(self):
        payload = b"wave1-export-payload"
        for codec in ("zlib", "zstd"):
            blob = train_gpt.compress_export_payload(payload, codec=codec)
            restored = train_gpt.decompress_export_payload(blob, codec=codec)
            self.assertEqual(restored, payload)

    def test_fake_quantize_per_row_respects_requested_levels(self):
        w = torch.tensor([[0.10, 0.25, -0.33, 0.90]], dtype=torch.float32)
        int8ish = train_gpt.fake_quantize_per_row(w, levels=127.0)
        int5ish = train_gpt.fake_quantize_per_row(w, levels=train_gpt.INT5_MAX)
        self.assertFalse(torch.allclose(int8ish, int5ish))

    def test_mixed_lowbit_quantize_roundtrip_preserves_tensor_shapes(self):
        state_dict = {
            "blocks.0.mlp.fc.weight": torch.randn(4, 4, dtype=torch.bfloat16),
            "blocks.0.attn.c_q.weight": torch.randn(4, 4, dtype=torch.bfloat16),
            "pair_hash.embed.weight": torch.randn(8, 3, dtype=torch.bfloat16),
            "pair_hash.pair_alpha": torch.tensor(0.25, dtype=torch.float32),
        }
        obj, stats = train_gpt.quantize_state_dict_for_export(state_dict, train_gpt.MIXED_LOWBITS_MODE)
        restored = train_gpt.dequantize_state_dict_for_export(obj)

        self.assertEqual(obj["export_mode"], train_gpt.MIXED_LOWBITS_MODE)
        self.assertGreater(stats["export_payload_bytes"], 0)
        for name, tensor in state_dict.items():
            self.assertEqual(restored[name].shape, tensor.shape)

    def test_gpt_forward_logits_supports_pairhash_and_eval_mask(self):
        model = train_gpt.GPT(
            vocab_size=32,
            num_layers=2,
            model_dim=16,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.02,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.0,
            bos_id=1,
            pairhash_enabled=True,
            pairhash_buckets=128,
            pairhash_dim=8,
        ).bfloat16()
        input_ids = torch.tensor([[1, 5, 6, 1, 7]], dtype=torch.int64)
        attn_mask = train_gpt.build_eval_attn_mask(input_ids, bos_id=1, doc_isolated=True).to(dtype=torch.bfloat16)
        logits = model.forward_logits(input_ids, attn_mask=attn_mask)
        self.assertEqual(logits.shape, (1, 5, 32))

    def test_configure_qat_levels_matches_export_tensor_spec(self):
        model = train_gpt.GPT(
            vocab_size=32,
            num_layers=2,
            model_dim=16,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.02,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.0,
            bos_id=1,
            pairhash_enabled=True,
            pairhash_buckets=128,
            pairhash_dim=8,
        )
        train_gpt.configure_qat_levels(model, train_gpt.MIXED_LOWBITS_MODE)
        self.assertEqual(model.blocks[0].mlp.fc._qat_levels, train_gpt.INT5_MAX)
        self.assertEqual(model.blocks[0].attn.c_q._qat_levels, train_gpt.INT6_MAX)
        self.assertEqual(model.pair_hash.proj._qat_levels, train_gpt.INT6_MAX)

    def test_build_token_param_groups_disables_embedding_weight_decay(self):
        model = train_gpt.GPT(
            vocab_size=32,
            num_layers=2,
            model_dim=16,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.02,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.0,
            bos_id=1,
            pairhash_enabled=True,
            pairhash_buckets=128,
            pairhash_dim=8,
        )
        groups = train_gpt.build_token_param_groups(model, token_lr=0.05)
        self.assertEqual(groups[0]["weight_decay"], 0.0)
        self.assertTrue(any(param is model.tok_emb.weight for param in groups[0]["params"]))
        self.assertTrue(any(param is model.pair_hash.embed.weight for param in groups[0]["params"]))

    def test_docs_and_validation_script_use_export_metric_names(self):
        repo_root = Path(train_gpt.__file__).resolve().parent
        run_full_validation = (repo_root / "run_full_validation.sh").read_text()
        handoff = (repo_root / "HANDOFF.md").read_text()
        self.assertIn("final_export_roundtrip_exact", run_full_validation)
        self.assertIn("Total submission size export", run_full_validation)
        self.assertIn("final_export_roundtrip_exact", handoff)
        self.assertNotIn("final_int8_zlib_roundtrip_exact", run_full_validation)
        self.assertNotIn("Total submission size int8", run_full_validation)
        self.assertNotIn("final_int8_zlib_roundtrip_exact", handoff)


if __name__ == "__main__":
    unittest.main()
