import os
import sys
from pathlib import Path
import json

import numpy as np

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent))
import scale_pyramid


def get_metadata(container, ds_name):
    with open(Path(container) / ds_name / ".zarray") as f:
        return json.load(f)


def test_s0(args_s0):
    container, ds_name = args_s0
    max_scale = 3
    scales = ";".join("2" for _ in range(max_scale))
    scale_pyramid.main([container, ds_name, scales])
    prev_size = None
    items = os.listdir(container)
    for scale in range(max_scale + 1):
        assert f"s{scale}" in items
        current_size = np.array(get_metadata(container, f"s{scale}")["shape"])
        if prev_size is not None:
            assert np.allclose(prev_size // 2, current_size)
        prev_size = current_size


def test_s1(args_s1):
    container, ds_name = args_s1
    max_scale = 3
    scales = ";".join("2" for _ in range(max_scale))
    scale_pyramid.main([container, ds_name, scales])
    items = os.listdir(container)
    prev_size = None
    for scale in range(1, max_scale + 2):
        assert f"s{scale}" in items
        current_size = np.array(get_metadata(container, f"s{scale}")["shape"])
        if prev_size is not None:
            assert np.allclose(prev_size // 2, current_size)
        prev_size = current_size


def test_not_s(args_not_s):
    container, ds_name = args_not_s
    max_scale = 3
    scales = ";".join("2" for _ in range(max_scale))
    scale_pyramid.main([container, ds_name, scales])
    items = os.listdir(os.path.join(container, ds_name))
    prev_size = None
    for scale in range(max_scale + 1):
        assert f"s{scale}" in items
        current_size = np.array(get_metadata(os.path.join(container, ds_name), f"s{scale}")["shape"])
        if prev_size is not None:
            assert np.allclose(prev_size // 2, current_size)
        prev_size = current_size
