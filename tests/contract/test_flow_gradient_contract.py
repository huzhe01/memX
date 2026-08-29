from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, SanaTransformer2DModel
from torch import Tensor, nn

from ratemem.adapters.sana_layout import (
    SanaDynamicAdapterBank,
    install_sana_dynamic_atoms,
)
from ratemem.sana.flow import (
    FlowBatch,
    FlowDraw,
    OneTimestepFlowTrainer,
)
from ratemem.support.amortizer import SupportAmortizer


def _schedule() -> tuple[tuple[float, ...], tuple[float, ...]]:
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    return (
        tuple(float(value) for value in scheduler.timesteps),
        tuple(float(value) for value in scheduler.sigmas),
    )


def _tiny_sana() -> SanaTransformer2DModel:
    transformer = SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
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


def _batch(batch_size: int = 2, *, device: torch.device | None = None) -> FlowBatch:
    target = device or torch.device("cpu")
    return FlowBatch(
        clean_latents=torch.randn(batch_size, 4, 4, 4, device=target),
        prompt_embeddings=torch.randn(batch_size, 3, 8, device=target),
        prompt_attention_mask=torch.tensor(
            [[1, 1, 0], [1, 1, 1]], dtype=torch.int64, device=target
        )[:batch_size],
        support_features=torch.randn(batch_size, 2, 6, device=target),
        support_mask=torch.tensor(
            [[True, False], [True, True]], dtype=torch.bool, device=target
        )[:batch_size],
        description_features=torch.randn(batch_size, 8, device=target),
    )


def _build(
    *,
    seed: int = 17,
    device: torch.device | None = None,
    with_frozen_buffer: bool = False,
) -> tuple[
    OneTimestepFlowTrainer,
    SanaTransformer2DModel,
    SanaDynamicAdapterBank,
    SupportAmortizer,
    torch.optim.AdamW,
]:
    torch.manual_seed(seed)
    target = device or torch.device("cpu")
    transformer = _tiny_sana().to(target)
    if with_frozen_buffer:
        transformer.register_buffer(
            "integrity_probe",
            torch.tensor([1.0, 2.0], device=target),
        )
    bank = install_sana_dynamic_atoms(
        transformer,
        rank=2,
        atom_count=4,
        expected_blocks=1,
    )
    transformer.enable_gradient_checkpointing()
    amortizer = SupportAmortizer(
        support_dim=6,
        description_dim=8,
        hidden_dim=16,
        projection_count=6,
        atom_count=4,
        layers=1,
        heads=4,
    ).to(target).train()
    optimizer = torch.optim.AdamW(
        [*bank.parameters(), *amortizer.parameters()],
        lr=1e-3,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    timesteps, sigmas = _schedule()
    trainer = OneTimestepFlowTrainer(
        transformer,
        bank,
        amortizer,
        timesteps,
        sigmas,
        optimizer,
        expected_amortizer_signature=amortizer.architecture_signature,
        autocast_dtype=torch.bfloat16 if target.type == "cuda" else None,
    )
    return trainer, transformer, bank, amortizer, optimizer


def _trainable_clones(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _assert_changed(before: dict[str, Tensor], module: nn.Module) -> None:
    after = dict(module.named_parameters())
    assert any(not torch.equal(value, after[name]) for name, value in before.items())


def _assert_bank_inactive(bank: SanaDynamicAdapterBank) -> None:
    assert all(wrapper._coefficients is None for wrapper in bank.wrappers)
    assert all(wrapper._activation_token is None for wrapper in bank.wrappers)


def test_train_step_uses_one_top_level_pass_and_every_trainable_gets_finite_gradient() -> None:
    trainer, transformer, bank, amortizer, _optimizer = _build()
    batch = _batch()
    forward_calls = 0
    seen_gradients: dict[int, Tensor] = {}

    def count_forward(_module: nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
        nonlocal forward_calls
        forward_calls += 1

    handle = transformer.register_forward_hook(count_forward)
    def capture(identity: int) -> Callable[[Tensor], Tensor]:
        def save(gradient: Tensor) -> Tensor:
            seen_gradients.setdefault(identity, gradient.detach().clone())
            return gradient

        return save

    gradient_handles = [
        parameter.register_hook(capture(id(parameter)))
        for parameter in (*bank.parameters(), *amortizer.parameters())
    ]
    atom_before = {
        name: parameter.detach().clone() for name, parameter in bank.named_parameters()
    }
    amortizer_before = _trainable_clones(amortizer)
    result = trainer.train_step(
        batch,
        generator=torch.Generator(device="cpu").manual_seed(19),
    )
    handle.remove()
    for gradient_handle in gradient_handles:
        gradient_handle.remove()

    expected = tuple(bank.parameters()) + tuple(amortizer.parameters())
    assert forward_calls == result.transformer_pass_count == 1
    assert result.loss > 0 and result.timestep_count == 2
    assert len(result.timestep_indices) == len(result.timesteps) == len(result.sigmas) == 2
    assert set(seen_gradients) == {id(parameter) for parameter in expected}
    assert all(torch.isfinite(gradient).all() for gradient in seen_gradients.values())
    assert result.gradients.code_l2 > 0
    assert result.gradients.atom_l2 > 0
    assert result.gradients.amortizer_l2 > 0
    assert result.gradients.atom_tensor_count == len(tuple(bank.parameters()))
    assert result.gradients.amortizer_tensor_count == len(tuple(amortizer.parameters()))
    assert all(parameter.grad is None for parameter in transformer.parameters())
    assert all(parameter.grad is None for parameter in amortizer.parameters())
    assert all(
        not parameter.requires_grad
        for wrapper in bank.wrappers
        for parameter in wrapper.base.parameters()
    )
    assert all(not child.training for child in transformer.modules())
    assert all(child.training for child in amortizer.modules())
    assert transformer.is_gradient_checkpointing
    assert all(
        atom_before[name]._version == 0
        or not torch.equal(atom_before[name], parameter)
        for name, parameter in bank.named_parameters()
    )
    _assert_changed(amortizer_before, amortizer)
    assert any(
        not torch.equal(atom_before[name], parameter)
        for name, parameter in bank.named_parameters()
    )
    _assert_bank_inactive(bank)


def test_train_step_is_deterministically_replayable_from_equal_state_and_generator() -> None:
    first, first_transformer, _first_bank, first_amortizer, _ = _build(seed=23)
    second, second_transformer, _second_bank, second_amortizer, _ = _build(seed=23)
    torch.manual_seed(29)
    batch = _batch()
    first_result = first.train_step(
        batch,
        generator=torch.Generator(device="cpu").manual_seed(31),
    )
    second_result = second.train_step(
        batch,
        generator=torch.Generator(device="cpu").manual_seed(31),
    )
    assert first_result == second_result
    for first_value, second_value in zip(
        first_transformer.state_dict().values(),
        second_transformer.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(first_value, second_value, rtol=0.0, atol=0.0)
    for first_value, second_value in zip(
        first_amortizer.state_dict().values(),
        second_amortizer.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(first_value, second_value, rtol=0.0, atol=0.0)


def test_evaluate_loss_uses_fixed_draw_once_and_never_updates_or_touches_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, transformer, bank, amortizer, optimizer = _build()
    batch = _batch()
    draw = FlowDraw(
        noise=torch.randn(batch.clean_latents.shape, generator=torch.Generator().manual_seed(41)),
        timestep_indices=torch.tensor([17, 901], dtype=torch.int64),
    )
    before_transformer = {
        name: value.detach().clone() for name, value in transformer.state_dict().items()
    }
    before_amortizer = {
        name: value.detach().clone() for name, value in amortizer.state_dict().items()
    }
    before_optimizer = optimizer.state_dict()
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_hook(count)

    def forbidden_step(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("evaluation called optimizer.step")

    monkeypatch.setattr(optimizer, "step", forbidden_step)
    first = trainer.evaluate_loss(batch, draw=draw)
    second = trainer.evaluate_loss(batch, draw=draw)
    handle.remove()
    assert first == second and first > 0
    assert calls == 2
    for name, value in transformer.state_dict().items():
        torch.testing.assert_close(value, before_transformer[name], rtol=0.0, atol=0.0)
    for name, value in amortizer.state_dict().items():
        torch.testing.assert_close(value, before_amortizer[name], rtol=0.0, atol=0.0)
    assert optimizer.state_dict() == before_optimizer
    assert all(parameter.grad is None for parameter in transformer.parameters())
    assert all(parameter.grad is None for parameter in amortizer.parameters())
    _assert_bank_inactive(bank)


def test_invalid_batch_and_generator_fail_before_zero_grad_rng_or_forward_and_do_not_poison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, transformer, _bank, _amortizer, optimizer = _build()
    batch = _batch()
    calls: list[str] = []
    original_zero_grad = optimizer.zero_grad

    def counted_zero_grad(*args: object, **kwargs: object) -> None:
        calls.append("zero")
        original_zero_grad(*args, **kwargs)

    monkeypatch.setattr(optimizer, "zero_grad", counted_zero_grad)
    handle = transformer.register_forward_pre_hook(
        lambda *_args: calls.append("forward")
    )
    with pytest.raises(TypeError, match="generator"):
        trainer.train_step(batch, generator=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="float32"):
        trainer.train_step(
            replace(batch, clean_latents=batch.clean_latents.double()),
            generator=torch.Generator().manual_seed(1),
        )
    assert calls == []
    result = trainer.train_step(
        batch,
        generator=torch.Generator().manual_seed(1),
    )
    handle.remove()
    assert result.loss > 0 and calls == ["zero", "forward"]


def test_disabled_grad_mode_is_rejected_before_zero_grad_and_does_not_poison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _transformer, _bank, _amortizer, optimizer = _build()
    batch = _batch()
    calls = 0
    original_zero_grad = optimizer.zero_grad

    def counted_zero_grad(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        original_zero_grad(*args, **kwargs)

    monkeypatch.setattr(optimizer, "zero_grad", counted_zero_grad)
    with torch.no_grad(), pytest.raises(RuntimeError, match="gradients enabled"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(79))
    assert calls == 0
    assert trainer.train_step(
        batch, generator=torch.Generator().manual_seed(79)
    ).loss > 0


def test_successful_optimizer_step_that_mutates_mode_is_caught_and_poisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, transformer, _bank, _amortizer, optimizer = _build()
    batch = _batch()
    original_step = optimizer.step

    def mutating_step(*args: object, **kwargs: object) -> object:
        result = original_step(*args, **kwargs)
        transformer.transformer_blocks[0].train()
        return result

    monkeypatch.setattr(optimizer, "step", mutating_step)
    with pytest.raises(RuntimeError, match="eval mode"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(83))
    transformer.eval()
    monkeypatch.setattr(optimizer, "step", original_step)
    with pytest.raises(RuntimeError, match="permanently poisoned"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(83))


def test_optimizer_version_bumps_without_value_changes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _transformer, bank, amortizer, optimizer = _build()
    batch = _batch()
    trainable = tuple(bank.parameters()) + tuple(amortizer.parameters())
    originals = tuple(parameter.detach().clone() for parameter in trainable)
    original_step = optimizer.step

    def restore_after_real_step(*args: object, **kwargs: object) -> object:
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            for parameter, original in zip(trainable, originals, strict=True):
                parameter.copy_(original)
        return result

    monkeypatch.setattr(optimizer, "step", restore_after_real_step)
    with pytest.raises(RuntimeError, match="did not change any atom parameter value"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(87))
    monkeypatch.setattr(optimizer, "step", original_step)
    with pytest.raises(RuntimeError, match="permanently poisoned"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(87))


def test_mutated_optimizer_moment_dtype_is_rejected_before_next_forward() -> None:
    trainer, transformer, _bank, _amortizer, optimizer = _build()
    batch = _batch()
    trainer.train_step(batch, generator=torch.Generator().manual_seed(89))
    first_parameter = optimizer.param_groups[0]["params"][0]
    optimizer.state[first_parameter]["exp_avg"] = optimizer.state[first_parameter][
        "exp_avg"
    ].double()
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_pre_hook(count)
    with pytest.raises(RuntimeError, match="dtype"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(89))
    handle.remove()
    assert calls == 0


@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("betas", (0.8, 0.999)),
        ("eps", 1e-7),
        ("weight_decay", 0.01),
        ("amsgrad", True),
        ("maximize", True),
        ("foreach", None),
        ("capturable", True),
        ("differentiable", True),
        ("fused", None),
        ("decoupled_weight_decay", False),
    ],
)
def test_every_pinned_adamw_hyperparameter_mutation_fails_before_forward(
    name: str,
    bad_value: object,
) -> None:
    trainer, transformer, _bank, _amortizer, optimizer = _build()
    batch = _batch()
    optimizer.param_groups[0][name] = bad_value
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_pre_hook(count)
    with pytest.raises(RuntimeError, match=f"AdamW {name}|optimizer hyperparameters"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(91))
    handle.remove()
    assert calls == 0


def test_disjoint_optimizer_views_sharing_one_storage_are_rejected_preflight() -> None:
    trainer, transformer, _bank, _amortizer, optimizer = _build()
    batch = _batch()
    trainer.train_step(batch, generator=torch.Generator().manual_seed(97))
    parameters = optimizer.param_groups[0]["params"]
    first = parameters[0]
    second = next(parameter for parameter in parameters[1:] if parameter.shape == first.shape)
    element_count = first.numel()
    shared = torch.zeros(element_count * 2, dtype=first.dtype, device=first.device)
    optimizer.state[first]["exp_avg"] = shared[:element_count].view_as(first)
    optimizer.state[second]["exp_avg"] = shared[element_count:].view_as(second)
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_pre_hook(count)
    with pytest.raises(RuntimeError, match="storage aliases"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(97))
    handle.remove()
    assert calls == 0


def test_clearing_initialized_optimizer_state_is_rejected_before_next_forward() -> None:
    trainer, transformer, _bank, _amortizer, optimizer = _build()
    batch = _batch()
    trainer.train_step(batch, generator=torch.Generator().manual_seed(101))
    optimizer.state.clear()
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_pre_hook(count)
    with pytest.raises(RuntimeError, match="state is incomplete"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(101))
    handle.remove()
    assert calls == 0


def test_constructor_rejects_nonempty_optimizer_state_even_when_schema_is_complete() -> None:
    trainer, _transformer, _bank, _amortizer, _optimizer = _build()
    batch = _batch()
    trainer.train_step(batch, generator=torch.Generator().manual_seed(102))
    with pytest.raises(ValueError, match="optimizer state must be empty"):
        OneTimestepFlowTrainer(
            trainer.transformer,
            trainer.adapter_bank,
            trainer.amortizer,
            trainer.training_timesteps,
            trainer.training_sigmas,
            trainer.optimizer,
            expected_amortizer_signature=trainer.amortizer.architecture_signature,
        )


def test_optimizer_state_data_mutation_is_rejected_before_forward() -> None:
    trainer, transformer, _bank, _amortizer, optimizer = _build()
    batch = _batch()
    trainer.train_step(batch, generator=torch.Generator().manual_seed(104))
    parameter = optimizer.param_groups[0]["params"][0]
    moment = optimizer.state[parameter]["exp_avg"]
    version = moment._version
    moment.data.add_(0.5)
    assert moment._version == version
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_pre_hook(count)
    with pytest.raises(RuntimeError, match="optimizer state.*digest"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(104))
    handle.remove()
    assert calls == 0


def test_optimizer_moment_aliasing_frozen_storage_is_rejected_globally() -> None:
    trainer, transformer, _bank, _amortizer, optimizer = _build()
    batch = _batch()
    trainer.train_step(batch, generator=torch.Generator().manual_seed(106))
    parameter = optimizer.param_groups[0]["params"][0]
    frozen = next(
        value
        for value in trainer.frozen_parameters
        if value.dtype == parameter.dtype and value.numel() >= parameter.numel()
    )
    optimizer.state[parameter]["exp_avg"] = frozen.view(-1)[
        : parameter.numel()
    ].view_as(parameter)
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_pre_hook(count)
    with pytest.raises(RuntimeError, match="global tensor storage alias"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(106))
    handle.remove()
    assert calls == 0


def test_valid_looking_optimizer_moment_mutation_is_rejected_before_forward() -> None:
    trainer, transformer, _bank, _amortizer, optimizer = _build()
    batch = _batch()
    trainer.train_step(batch, generator=torch.Generator().manual_seed(103))
    parameter = optimizer.param_groups[0]["params"][0]
    optimizer.state[parameter]["exp_avg"].zero_()
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_pre_hook(count)
    with pytest.raises(RuntimeError, match="optimizer state changed outside"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(103))
    handle.remove()
    assert calls == 0


def test_trainable_parameter_mutation_outside_trainer_is_rejected_preflight() -> None:
    trainer, transformer, bank, _amortizer, _optimizer = _build()
    batch = _batch()
    parameter = next(iter(bank.parameters()))
    parameter.detach().zero_()
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_pre_hook(count)
    with pytest.raises(RuntimeError, match="trainable parameter versions changed"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(107))
    handle.remove()
    assert calls == 0


@pytest.mark.parametrize("failure_kind", ["forward", "output", "optimizer"])
def test_any_failure_after_side_effect_clears_state_and_permanently_poisons(
    failure_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, transformer, bank, amortizer, optimizer = _build()
    batch = _batch()
    original_forward = transformer.forward
    original_step = optimizer.step
    calls = 0

    if failure_kind == "forward":
        def failed_forward(*_args: object, **_kwargs: object) -> object:
            nonlocal calls
            calls += 1
            raise LookupError("injected forward failure")

        monkeypatch.setattr(transformer, "forward", failed_forward)
        expected = "forward failure"
    elif failure_kind == "output":
        def bad_output(*args: object, **kwargs: object) -> tuple[Tensor, Tensor]:
            nonlocal calls
            calls += 1
            output = original_forward(*args, **kwargs)
            assert type(output) is tuple
            return output[0], output[0]

        monkeypatch.setattr(transformer, "forward", bad_output)
        expected = "exact one-tensor tuple"
    else:
        first_parameter = next(iter(bank.parameters()))

        def partial_step(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            with torch.no_grad():
                first_parameter.add_(1.0)
            raise ArithmeticError("injected partial optimizer failure")

        monkeypatch.setattr(optimizer, "step", partial_step)
        expected = "partial optimizer failure"

    with pytest.raises(Exception, match=expected):
        trainer.train_step(
            batch,
            generator=torch.Generator().manual_seed(47),
        )
    assert calls == 1
    assert all(parameter.grad is None for parameter in transformer.parameters())
    assert all(parameter.grad is None for parameter in amortizer.parameters())
    _assert_bank_inactive(bank)
    monkeypatch.setattr(transformer, "forward", original_forward)
    monkeypatch.setattr(optimizer, "step", original_step)
    with pytest.raises(RuntimeError, match="permanently poisoned"):
        trainer.train_step(
            batch,
            generator=torch.Generator().manual_seed(47),
        )
    assert calls == 1


def test_model_output_on_wrong_device_is_rejected_before_loss_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, transformer, bank, amortizer, _optimizer = _build()
    batch = _batch()

    def wrong_device_output(*_args: object, **_kwargs: object) -> tuple[Tensor]:
        return (torch.empty(batch.clean_latents.shape, device="meta"),)

    monkeypatch.setattr(transformer, "forward", wrong_device_output)
    with pytest.raises(ValueError, match="prediction device"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(109))
    _assert_bank_inactive(bank)
    with pytest.raises(RuntimeError, match="permanently poisoned"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(109))
    assert all(parameter.grad is None for parameter in transformer.parameters())
    assert all(parameter.grad is None for parameter in amortizer.parameters())
    _assert_bank_inactive(bank)


@pytest.mark.parametrize("wrong_dtype", [torch.float64, torch.float16])
def test_cpu_model_output_must_be_exact_float32(
    wrong_dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, transformer, bank, _amortizer, _optimizer = _build()
    batch = _batch()
    original_forward = transformer.forward

    def wrong_dtype_output(*args: object, **kwargs: object) -> tuple[Tensor]:
        output = original_forward(*args, **kwargs)
        assert type(output) is tuple and len(output) == 1
        return (output[0].to(dtype=wrong_dtype),)

    monkeypatch.setattr(transformer, "forward", wrong_dtype_output)
    with pytest.raises(TypeError, match="prediction dtype.*float32"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(113))
    _assert_bank_inactive(bank)


@pytest.mark.parametrize("target", ["atom", "amortizer", "frozen"])
def test_finite_data_mutation_is_detected_before_forward_without_version_counter(
    target: str,
) -> None:
    trainer, transformer, bank, amortizer, _optimizer = _build()
    batch = _batch()
    if target == "atom":
        parameter = next(iter(bank.parameters()))
    elif target == "amortizer":
        parameter = next(iter(amortizer.parameters()))
    else:
        parameter = trainer.frozen_parameters[0]
    version = parameter._version
    parameter.data.add_(0.125)
    assert parameter._version == version
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_pre_hook(count)
    with pytest.raises(RuntimeError, match=f"{target}.*digest|frozen.*digest"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(127))
    handle.remove()
    assert calls == 0


def test_evaluate_forward_data_mutation_is_detected_and_permanently_poisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, transformer, bank, _amortizer, _optimizer = _build()
    batch = _batch()
    draw = FlowDraw(
        noise=torch.randn_like(batch.clean_latents),
        timestep_indices=torch.tensor([11, 19], dtype=torch.int64),
    )
    frozen = trainer.frozen_parameters[0]
    original_forward = transformer.forward

    def mutating_forward(*args: object, **kwargs: object) -> object:
        output = original_forward(*args, **kwargs)
        frozen.data.add_(0.25)
        return output

    monkeypatch.setattr(transformer, "forward", mutating_forward)
    with pytest.raises(RuntimeError, match="frozen.*digest"):
        trainer.evaluate_loss(batch, draw=draw)
    _assert_bank_inactive(bank)
    monkeypatch.setattr(transformer, "forward", original_forward)
    with pytest.raises(RuntimeError, match="permanently poisoned"):
        trainer.evaluate_loss(batch, draw=draw)


def test_frozen_buffer_data_mutation_is_detected_before_forward() -> None:
    trainer, transformer, _bank, _amortizer, _optimizer = _build(
        with_frozen_buffer=True
    )
    batch = _batch()
    buffer = dict(transformer.named_buffers())["integrity_probe"]
    version = buffer._version
    buffer.data.mul_(2.0)
    assert buffer._version == version
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_pre_hook(count)
    with pytest.raises(RuntimeError, match="frozen.*digest"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(131))
    handle.remove()
    assert calls == 0


def test_optimizer_hyperparameter_or_parameter_topology_mutation_fails_before_side_effect() -> None:
    trainer, transformer, _bank, _amortizer, optimizer = _build()
    batch = _batch()
    calls = 0

    def count(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = transformer.register_forward_pre_hook(count)
    optimizer.param_groups[0]["lr"] = 0.5
    with pytest.raises(RuntimeError, match="AdamW lr|optimizer hyperparameters"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(53))
    assert calls == 0
    optimizer.param_groups[0]["lr"] = 1e-3
    optimizer.param_groups[0]["params"].append(nn.Parameter(torch.zeros(())))
    with pytest.raises(RuntimeError, match="optimizer parameter ownership"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(53))
    assert calls == 0
    optimizer.param_groups[0]["params"].pop()
    assert trainer.train_step(
        batch, generator=torch.Generator().manual_seed(53)
    ).loss > 0
    handle.remove()


@pytest.mark.parametrize("broken_contract", ["transformer_mode", "amortizer_mode", "gc"])
def test_constructor_requires_eval_transformer_train_amortizer_and_checkpointing(
    broken_contract: str,
) -> None:
    torch.manual_seed(59)
    transformer = _tiny_sana()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=1
    )
    transformer.enable_gradient_checkpointing()
    amortizer = SupportAmortizer(
        support_dim=6,
        description_dim=8,
        hidden_dim=16,
        projection_count=6,
        atom_count=4,
        layers=1,
        heads=4,
    ).train()
    if broken_contract == "transformer_mode":
        transformer.transformer_blocks[0].train()
    elif broken_contract == "amortizer_mode":
        amortizer.encoder.layers[0].eval()
    else:
        transformer.disable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        [*bank.parameters(), *amortizer.parameters()],
        lr=1e-3,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    timesteps, sigmas = _schedule()
    with pytest.raises((TypeError, ValueError, RuntimeError), match="eval|train|checkpoint"):
        OneTimestepFlowTrainer(
            transformer,
            bank,
            amortizer,
            timesteps,
            sigmas,
            optimizer,
            expected_amortizer_signature=amortizer.architecture_signature,
        )


def test_constructor_rejects_wrong_bank_optimizer_or_amortizer_signature() -> None:
    trainer, _transformer, _bank, _amortizer, _optimizer = _build()
    other_transformer = _tiny_sana()
    other_bank = install_sana_dynamic_atoms(
        other_transformer, rank=2, atom_count=4, expected_blocks=1
    )
    other_transformer.enable_gradient_checkpointing()
    with pytest.raises(ValueError, match="same transformer"):
        OneTimestepFlowTrainer(
            trainer.transformer,
            other_bank,
            trainer.amortizer,
            trainer.training_timesteps,
            trainer.training_sigmas,
            trainer.optimizer,
            expected_amortizer_signature=trainer.amortizer.architecture_signature,
        )
    with pytest.raises(ValueError, match="architecture signature"):
        OneTimestepFlowTrainer(
            trainer.transformer,
            trainer.adapter_bank,
            trainer.amortizer,
            trainer.training_timesteps,
            trainer.training_sigmas,
            trainer.optimizer,
            expected_amortizer_signature="0" * 64,
        )

    optimizer = torch.optim.AdamW(
        list(trainer.adapter_bank.parameters()),
        lr=1e-3,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    with pytest.raises(ValueError, match="optimizer parameter ownership"):
        OneTimestepFlowTrainer(
            trainer.transformer,
            trainer.adapter_bank,
            trainer.amortizer,
            trainer.training_timesteps,
            trainer.training_sigmas,
            optimizer,
            expected_amortizer_signature=trainer.amortizer.architecture_signature,
        )


def test_production_transformer_contract_rejects_nonproduction_amortizer_architecture() -> None:
    transformer = _tiny_sana()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=1
    )
    transformer.enable_gradient_checkpointing()
    transformer.register_to_config(
        in_channels=32,
        out_channels=32,
        sample_size=32,
        caption_channels=2304,
        num_layers=20,
    )
    amortizer = SupportAmortizer(
        support_dim=6,
        description_dim=8,
        hidden_dim=16,
        projection_count=6,
        atom_count=4,
        layers=1,
        heads=4,
    ).train()
    optimizer = torch.optim.AdamW(
        [*bank.parameters(), *amortizer.parameters()],
        lr=1e-3,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    timesteps, sigmas = _schedule()
    with pytest.raises(ValueError, match="production amortizer architecture"):
        OneTimestepFlowTrainer(
            transformer,
            bank,
            amortizer,
            timesteps,
            sigmas,
            optimizer,
            expected_amortizer_signature=amortizer.architecture_signature,
        )


def test_reentry_is_rejected_and_poisoned_without_a_second_transformer_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, transformer, _bank, _amortizer, _optimizer = _build()
    batch = _batch()
    draw = FlowDraw(
        noise=torch.randn_like(batch.clean_latents),
        timestep_indices=torch.tensor([3, 7], dtype=torch.int64),
    )
    original_forward = transformer.forward
    calls = 0

    def reenter(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        trainer.evaluate_loss(batch, draw=draw)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(transformer, "forward", reenter)
    with pytest.raises(RuntimeError, match="already executing"):
        trainer.train_step(batch, generator=torch.Generator().manual_seed(61))
    assert calls == 1
    with pytest.raises(RuntimeError, match="permanently poisoned"):
        trainer.evaluate_loss(batch, draw=draw)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bfloat16_train_step_preserves_fp32_objective_and_gradients() -> None:
    device = torch.device("cuda:0")
    trainer, transformer, bank, amortizer, _optimizer = _build(device=device)
    transformer.to(dtype=torch.bfloat16)
    # Moving the transformer also moves the installed atom parameters; rebuild the trainer
    # so its immutable placement snapshot describes the BF16 production-style state.
    optimizer = torch.optim.AdamW(
        [*bank.parameters(), *amortizer.parameters()],
        lr=1e-3,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    timesteps, sigmas = _schedule()
    trainer = OneTimestepFlowTrainer(
        transformer,
        bank,
        amortizer,
        timesteps,
        sigmas,
        optimizer,
        expected_amortizer_signature=amortizer.architecture_signature,
        autocast_dtype=torch.bfloat16,
    )
    result = trainer.train_step(
        _batch(device=device),
        generator=torch.Generator(device=device).manual_seed(67),
    )
    assert result.loss > 0
    assert result.gradients.code_l2 > 0
    assert result.gradients.atom_l2 > 0
    assert result.gradients.amortizer_l2 > 0
