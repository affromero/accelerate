# Copyright 2026 Andres Romero. All rights reserved.
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

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from accelerate import load_checkpoint_in_model


def write_checkpoint(directory, safe=False):
    directory.mkdir(parents=True, exist_ok=True)
    model = torch.nn.Linear(2, 1)
    state = {"weight": torch.tensor([[2.0, 3.0]]), "bias": torch.tensor([7.0])}
    suffix = ".safetensors" if safe else ".bin"
    names = []
    for key, value in state.items():
        name = key + suffix
        names.append(name)
        if safe:
            save_file({key: value}, str(directory / name), metadata={"format": "pt"})
        else:
            torch.save({key: value}, directory / name)
    index = directory / "model.index.json"
    index.write_text(json.dumps({"weight_map": dict(zip(state, names))}))
    return model, index


def assert_loaded(model):
    torch.testing.assert_close(model(torch.tensor([[11.0, 13.0]])), torch.tensor([[68.0]]))


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("direct_index", [False, True])
def test_valid_sharded_checkpoints_preserve_inference(tmp_path, safe, direct_index):
    model, index = write_checkpoint(tmp_path / "model", safe)
    load_checkpoint_in_model(model, str(index if direct_index else index.parent), device_map={"": "cpu"})
    assert_loaded(model)


@pytest.mark.parametrize("filename", ["../outside.bin", "/tmp/outside.bin", "C:/outside.bin", "..\\outside.bin"])
def test_index_cannot_select_paths_outside_the_checkpoint(tmp_path, filename):
    model, index = write_checkpoint(tmp_path / "model")
    index.write_text(json.dumps({"weight_map": {"weight": filename}}))
    with pytest.raises(ValueError, match="inside the checkpoint directory"):
        load_checkpoint_in_model(model, str(index.parent), device_map={"": "cpu"})


@pytest.mark.parametrize("mapping", [[], None, {}, {"weight": []}, {"weight": 17}, {"weight": ""}])
def test_malformed_weight_maps_fail_before_loading(tmp_path, mapping):
    model, index = write_checkpoint(tmp_path / "model")
    index.write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError):
        load_checkpoint_in_model(model, str(index.parent), device_map={"": "cpu"})


@pytest.mark.parametrize("index_symlink", [False, True])
def test_checkpoint_symlinks_cannot_escape_the_model_directory(tmp_path, index_symlink):
    model, index = write_checkpoint(tmp_path / "model")
    selected = index if index_symlink else index.parent / "weight.bin"
    outside = tmp_path / selected.name
    selected.rename(outside)
    selected.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        load_checkpoint_in_model(model, str(index.parent), device_map={"": "cpu"})


@pytest.mark.parametrize("safe", [False, True])
def test_huggingface_index_and_shard_blob_symlinks_remain_supported(tmp_path, safe):
    root = tmp_path / "models--owner--model"
    snapshot = root / "snapshots" / ("a" * 40)
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    model, index = write_checkpoint(snapshot, safe)
    for number, path in enumerate(list(snapshot.iterdir())):
        blob = blobs / (str(number) * 64)
        path.rename(blob)
        path.symlink_to(blob)
    load_checkpoint_in_model(model, str(index.parent), device_map={"": "cpu"})
    assert_loaded(model)


def test_huggingface_layout_cannot_allow_an_escaped_blobs_directory(tmp_path):
    root = tmp_path / "models--owner--model"
    snapshot = root / "snapshots" / ("b" * 40)
    model, index = write_checkpoint(snapshot)
    outside = tmp_path / "external-blobs"
    outside.mkdir()
    (root / "blobs").symlink_to(outside, target_is_directory=True)
    shard = index.parent / "weight.bin"
    blob = outside / "weight"
    shard.rename(blob)
    shard.symlink_to(blob)
    with pytest.raises(ValueError, match="outside"):
        load_checkpoint_in_model(model, str(snapshot), device_map={"": "cpu"})


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO")
@pytest.mark.parametrize("name", ["pytorch_model.bin", "model.safetensors", "model.index.json"])
def test_directory_selected_fifo_is_rejected_without_blocking(tmp_path, name):
    os.mkfifo(tmp_path / name)
    code = """
import sys, torch
from accelerate import load_checkpoint_in_model
try:
    load_checkpoint_in_model(torch.nn.Linear(2, 1), sys.argv[1], device_map={'': 'cpu'})
except ValueError as error:
    assert 'regular file' in str(error), str(error)
else:
    raise AssertionError('FIFO was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("safe", [False, True])
def test_replacing_a_checkpoint_path_does_not_replace_the_opened_weights(tmp_path, monkeypatch, safe):
    model, index = write_checkpoint(tmp_path / "model", safe)
    replacement = tmp_path / "outside"
    replacement.mkdir()
    original_open = os.open
    replacement_attempted = False

    def replace_after_open(path, flags, *args, **kwargs):
        nonlocal replacement_attempted
        descriptor = original_open(path, flags, *args, **kwargs)
        name = Path(os.fspath(path)).name
        if name == ("weight.safetensors" if safe else "weight.bin") and not replacement_attempted:
            selected = index.parent / name
            outside = replacement / name
            if safe:
                save_file({"weight": torch.tensor([[99.0, 99.0]])}, str(outside), metadata={"format": "pt"})
            else:
                torch.save({"weight": torch.tensor([[99.0, 99.0]])}, outside)
            try:
                selected.unlink()
            except PermissionError:
                if os.name != "nt":
                    raise
            else:
                selected.symlink_to(outside)
            replacement_attempted = True
        return descriptor

    monkeypatch.setattr(os, "open", replace_after_open)
    load_checkpoint_in_model(model, str(index.parent), device_map={"": "cpu"})
    assert replacement_attempted
    assert_loaded(model)


def test_private_safetensors_copy_is_removed_without_invalidating_loaded_weights(tmp_path, monkeypatch):
    model, index = write_checkpoint(tmp_path / "model", safe=True)
    original_exists = os.path.exists
    original_directory = tempfile.TemporaryDirectory
    copied_directories = []

    def descriptor_paths_are_unavailable(path):
        if str(path).startswith(("/proc/self/fd/", "/dev/fd/")):
            return False
        return original_exists(path)

    def private_directory(*args, **kwargs):
        directory = original_directory(*args, **kwargs)
        copied_directories.append(directory.name)
        return directory

    monkeypatch.setattr(os.path, "exists", descriptor_paths_are_unavailable)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", private_directory)
    load_checkpoint_in_model(model, str(index.parent), device_map={"": "cpu"})
    assert_loaded(model)
    assert copied_directories
    assert all(not original_exists(directory) for directory in copied_directories)
