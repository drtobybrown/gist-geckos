"""Tests for GENERAL workspace floating-point policy (_auxiliary.workspace_dtype)."""
import os

import numpy as np

from ngistPipeline.auxiliary import _auxiliary


def _cfg(**general_kwargs):
    return {"GENERAL": dict(general_kwargs)}


def test_workspace_dtype_defaults_to_float64():
    assert _auxiliary.workspace_dtype({"GENERAL": {}}) == np.dtype(np.float64)


def test_workspace_dtype_use_float32():
    assert _auxiliary.workspace_dtype(_cfg(USE_FLOAT32=True)) == np.dtype(np.float32)


def test_workspace_dtype_use_float32_string():
    assert _auxiliary.workspace_dtype(_cfg(USE_FLOAT32="true")) == np.dtype(np.float32)


def test_workspace_dtype_array_dtype_float32():
    assert _auxiliary.workspace_dtype(_cfg(ARRAY_DTYPE="float32")) == np.dtype(np.float32)


def test_workspace_dtype_array_dtype_float64():
    assert _auxiliary.workspace_dtype(_cfg(ARRAY_DTYPE="float64")) == np.dtype(np.float64)


def test_workspace_dtype_env_forces_float64(monkeypatch):
    monkeypatch.setenv("NGIST_USE_FLOAT64", "1")
    assert _auxiliary.workspace_dtype(_cfg(USE_FLOAT32=True)) == np.dtype(np.float64)


def test_workspace_dtype_array_dtype_overrides_use_float32():
    assert _auxiliary.workspace_dtype(
        _cfg(USE_FLOAT32=True, ARRAY_DTYPE="float64")
    ) == np.dtype(np.float64)
