import pytest
import zarr
import numpy as np


@pytest.fixture(scope="session")
def ds_args():
    shape = (32, 32, 32)
    chunks = (16, 16, 16)
    size = np.product(shape)
    repeats = size // 256
    data = np.concatenate((np.arange(256, dtype="uint8"),) * repeats).reshape(shape)
    return {"data": data, "chunks": chunks}


def make_dataset(ds_name, tmp_path, ds_args):
    container = tmp_path / "my_data.zarr"
    container.mkdir()
    store = zarr.DirectoryStore(container)
    group = zarr.group(store, overwrite=True)

    group.create_dataset(ds_name, **ds_args)
    return str(container), ds_name


@pytest.fixture
def args_s0(tmp_path, ds_args):
    return make_dataset("s0", tmp_path, ds_args)


@pytest.fixture
def args_s1(tmp_path, ds_args):
    return make_dataset("s1", tmp_path, ds_args)


@pytest.fixture
def args_not_s(tmp_path, ds_args):
    return make_dataset("potato", tmp_path, ds_args)
