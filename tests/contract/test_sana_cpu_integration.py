from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import SanaTransformer2DModel

from ratemem.adapters.dynamic_atom_linear import DynamicAtomLinear
from ratemem.adapters.sana_layout import install_sana_dynamic_atoms


def _tiny_sana() -> SanaTransformer2DModel:
    transformer = SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=4,
        cross_attention_dim=8,
        caption_channels=8,
        mlp_ratio=1.0,
        sample_size=4,
        patch_size=1,
        qk_norm=None,
    )
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer


@dataclass(frozen=True)
class _Inputs:
    hidden: torch.Tensor
    captions: torch.Tensor
    mask: torch.Tensor
    timesteps: torch.Tensor

    def sample(self, index: int) -> _Inputs:
        return _Inputs(
            hidden=self.hidden[index : index + 1],
            captions=self.captions[index : index + 1],
            mask=self.mask[index : index + 1],
            timesteps=self.timesteps[index : index + 1],
        )


def _inputs(batch: int) -> _Inputs:
    return _Inputs(
        hidden=torch.randn(batch, 4, 4, 4),
        captions=torch.randn(batch, 3, 8),
        mask=torch.ones(batch, 3),
        timesteps=torch.linspace(100.0, 700.0, batch),
    )


def _forward(transformer: SanaTransformer2DModel, values: _Inputs) -> torch.Tensor:
    return transformer(
        hidden_states=values.hidden,
        encoder_hidden_states=values.captions,
        encoder_attention_mask=values.mask,
        timestep=values.timesteps,
        return_dict=True,
    ).sample


def test_tiny_diffusers_sana_zero_code_is_bit_exact_and_wraps_only_qkv() -> None:
    torch.manual_seed(31)
    transformer = _tiny_sana()
    values = _inputs(2)
    non_qkv = tuple(
        block.attn1.to_out[0] for block in transformer.transformer_blocks
    )
    with torch.no_grad():
        expected = _forward(transformer, values)

    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    zero_code = torch.zeros(2, bank.layout.code_dim)
    with torch.no_grad(), bank.activate(zero_code):
        actual = _forward(transformer, values)

    assert torch.equal(actual, expected)
    assert actual.shape == values.hidden.shape
    assert len(bank.wrappers) == 12
    assert sum(
        isinstance(module, DynamicAtomLinear) for module in transformer.modules()
    ) == 12
    assert non_qkv == tuple(
        block.attn1.to_out[0] for block in transformer.transformer_blocks
    )


def test_tiny_diffusers_sana_batched_codes_do_not_cross_talk() -> None:
    torch.manual_seed(37)
    transformer = _tiny_sana()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    values = _inputs(2)
    codes = torch.randn(2, bank.layout.code_dim)

    with torch.no_grad(), bank.activate(codes):
        batched = _forward(transformer, values)
    independent = []
    for index in range(2):
        with torch.no_grad(), bank.activate(codes[index]):
            independent.append(_forward(transformer, values.sample(index)))

    torch.testing.assert_close(
        batched,
        torch.cat(independent),
        rtol=1e-5,
        atol=1e-6,
    )


def test_tiny_diffusers_sana_gradients_reach_code_and_every_atom_but_no_base() -> None:
    torch.manual_seed(41)
    transformer = _tiny_sana()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    values = _inputs(2)
    code = torch.randn(2, bank.layout.code_dim, requires_grad=True)

    with bank.activate(code):
        output = _forward(transformer, values)
        output.square().mean().backward()

    assert output.shape == values.hidden.shape
    assert code.grad is not None and torch.count_nonzero(code.grad) > 0
    for wrapper in bank.wrappers:
        assert wrapper.atom_down.grad is not None
        assert torch.count_nonzero(wrapper.atom_down.grad) > 0
        assert wrapper.atom_up.grad is not None
        assert torch.count_nonzero(wrapper.atom_up.grad) > 0
        assert all(parameter.grad is None for parameter in wrapper.base.parameters())
    assert {
        id(parameter)
        for parameter in transformer.parameters()
        if parameter.requires_grad
    } == {id(parameter) for parameter in bank.parameters()}
