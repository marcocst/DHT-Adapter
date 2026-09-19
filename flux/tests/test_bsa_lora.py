import unittest

import torch
import torch.nn as nn
from diffusers import FluxTransformer2DModel
from diffusers.loaders import AttnProcsLayers

from model.lora import (
    BSALinearLayer,
    FluxLoraAttnProcessor,
    LoRALinearLayer,
    set_flux_transformer_attn_processor,
)


class BSALinearLayerTests(unittest.TestCase):
    def test_random_initialization_is_noop_but_s_has_gradient(self):
        torch.manual_seed(7)
        layer = BSALinearLayer(6, 5, rank=3, init_mode="random")
        inputs = torch.randn(4, 6)
        target = torch.randn(4, 5)

        output = layer(inputs)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

        loss = torch.mean((output - target) ** 2)
        loss.backward()

        self.assertFalse(layer.down.weight.requires_grad)
        self.assertFalse(layer.up.weight.requires_grad)
        self.assertTrue(layer.middle.weight.requires_grad)
        self.assertIsNone(layer.down.weight.grad)
        self.assertIsNone(layer.up.weight.grad)
        self.assertIsNotNone(layer.middle.weight.grad)
        self.assertGreater(layer.middle.weight.grad.norm().item(), 0.0)

    def test_svd_initialization_decomposes_w0_and_is_noop(self):
        torch.manual_seed(11)
        original = nn.Linear(5, 4, bias=False)
        layer = BSALinearLayer(
            5,
            4,
            rank=3,
            original_layer=original,
            init_mode="svd",
            svd_oversample=2,
            svd_niter=8,
            svd_device="cpu",
        )

        inputs = torch.randn(2, 5)
        self.assertTrue(torch.equal(layer(inputs), torch.zeros(2, 4)))
        self.assertTrue(
            torch.equal(layer.middle.weight, layer.middle_reference)
        )

        reconstructed = (
            layer.up.weight @ layer.middle_reference @ layer.down.weight
        )
        _, singular_values, _ = torch.linalg.svd(original.weight, full_matrices=False)
        expected_error = torch.sqrt(torch.sum(singular_values[3:] ** 2))
        actual_error = torch.linalg.norm(original.weight - reconstructed)
        self.assertTrue(
            torch.allclose(actual_error, expected_error, atol=1e-4, rtol=1e-4)
        )

    def test_timestep_mask_limits_s_to_top_left_block(self):
        layer = BSALinearLayer(4, 4, rank=4, init_mode="checkpoint")
        with torch.no_grad():
            layer.down.weight.copy_(torch.eye(4))
            layer.up.weight.copy_(torch.eye(4))
            layer.middle.weight.fill_(1.0)
            layer.middle_reference.zero_()

        inputs = torch.ones(1, 4)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        output = layer(inputs, mask=mask)
        self.assertTrue(
            torch.equal(output, torch.tensor([[2.0, 2.0, 0.0, 0.0]]))
        )

        output.sum().backward()
        gradient = layer.middle.weight.grad
        self.assertGreater(gradient[:2, :2].abs().sum().item(), 0.0)
        self.assertEqual(gradient[2:, :].abs().sum().item(), 0.0)
        self.assertEqual(gradient[:, 2:].abs().sum().item(), 0.0)

    def test_checkpoint_mode_restores_all_bsa_tensors(self):
        torch.manual_seed(23)
        source = BSALinearLayer(5, 7, rank=3, init_mode="random")
        with torch.no_grad():
            source.middle.weight.normal_()
            source.middle_reference.normal_()

        restored = BSALinearLayer(5, 7, rank=3, init_mode="checkpoint")
        restored.load_state_dict(source.state_dict(), strict=True)

        inputs = torch.randn(2, 5)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        self.assertTrue(
            torch.equal(source(inputs, mask=mask), restored(inputs, mask=mask))
        )

    def test_legacy_lora_initialization_remains_unchanged(self):
        torch.manual_seed(31)
        layer = LoRALinearLayer(5, 7, rank=3)
        self.assertGreater(layer.down.weight.norm().item(), 0.0)
        self.assertEqual(layer.up.weight.norm().item(), 0.0)

    def test_transformer_checkpoint_layout_restores_without_recomputing_svd(self):
        def make_transformer():
            return FluxTransformer2DModel(
                patch_size=1,
                in_channels=4,
                num_layers=1,
                num_single_layers=1,
                attention_head_dim=8,
                num_attention_heads=2,
                joint_attention_dim=16,
                pooled_projection_dim=8,
                guidance_embeds=True,
                axes_dims_rope=(2, 2, 4),
            )

        source_transformer = make_transformer()
        source_transformer.requires_grad_(False)
        set_flux_transformer_attn_processor(
            source_transformer,
            set_attn_proc_func=lambda name, dh, nh, ap, original_layer: FluxLoraAttnProcessor(
                hidden_size=source_transformer.inner_dim,
                rank=2,
                lora_linear_layer=lambda *args, **kwargs: BSALinearLayer(
                    *args,
                    **kwargs,
                    init_mode="svd",
                    svd_oversample=1,
                    svd_niter=2,
                    svd_device="cpu",
                ),
                original_layer=original_layer,
            ),
        )
        source_layers = AttnProcsLayers(source_transformer.attn_processors)
        source_state = source_layers.state_dict()

        restored_transformer = make_transformer()
        restored_transformer.requires_grad_(False)
        set_flux_transformer_attn_processor(
            restored_transformer,
            set_attn_proc_func=lambda name, dh, nh, ap, original_layer: FluxLoraAttnProcessor(
                hidden_size=restored_transformer.inner_dim,
                rank=2,
                lora_linear_layer=lambda *args, **kwargs: BSALinearLayer(
                    *args,
                    **kwargs,
                    init_mode="checkpoint",
                ),
                original_layer=original_layer,
            ),
        )
        restored_layers = AttnProcsLayers(restored_transformer.attn_processors)
        restored_layers.load_state_dict(source_state, strict=True)

        self.assertEqual(
            set(source_state.keys()),
            set(restored_layers.state_dict().keys()),
        )
        trainable_names = [
            name for name, parameter in restored_layers.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable_names)
        self.assertTrue(all("middle.weight" in name for name in trainable_names))


if __name__ == "__main__":
    unittest.main()
