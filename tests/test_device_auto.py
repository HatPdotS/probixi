from __future__ import annotations

import types

import click
import pytest
import torch

import probixi.cli as cli
from probixi.multigpu import resolve_devices
from probixi.probixi import auto_device


def _fake_mps(available: bool):
    return types.SimpleNamespace(is_available=lambda: available)


def test_auto_device_returns_a_concrete_device():
    dev = auto_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "mps", "cpu"}


def test_auto_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _fake_mps(True), raising=False)
    assert auto_device().type == "cuda"


def test_auto_device_falls_back_to_mps_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _fake_mps(True), raising=False)
    assert auto_device().type == "mps"


def test_auto_device_falls_back_to_cpu_without_an_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _fake_mps(False), raising=False)
    assert auto_device().type == "cpu"


def test_auto_device_tolerates_a_torch_build_without_mps(monkeypatch):
    # torch.backends.mps is absent on some builds; that must not raise
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delattr(torch.backends, "mps", raising=False)
    assert auto_device().type == "cpu"


def test_resolve_devices_uses_auto_when_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _fake_mps(True), raising=False)
    assert [d.type for d in resolve_devices(None)] == ["mps"]


@pytest.mark.parametrize("spec", [None, "auto", "AUTO", " auto "])
def test_cli_auto_leaves_the_single_device_path(spec):
    # None means "resolve in Probixi"; an explicit "auto" must mean the same
    assert cli._resolve_cli_devices(spec, None, None) is None


def test_cli_auto_does_not_conflict_with_multi_gpu_flags():
    got = cli._resolve_cli_devices("auto", "cuda:0,cuda:1", None)
    assert [str(d) for d in got] == ["cuda:0", "cuda:1"]


def test_cli_an_explicit_device_still_conflicts_with_multi_gpu_flags():
    with pytest.raises(click.UsageError):
        cli._resolve_cli_devices("cuda", "cuda:0,cuda:1", None)


def test_cli_named_device_is_honoured():
    assert [str(d) for d in cli._resolve_cli_devices("cpu", None, None)] == ["cpu"]
