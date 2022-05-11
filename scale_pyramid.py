#!/usr/bin/env python
import argparse
import daisy
import numpy as np
import skimage.measure
import zarr

# monkey-patch os.mkdirs, due to bug in zarr
import os
import logging

logger = logging.getLogger(__name__)

prev_makedirs = os.makedirs
DEFAULT_WORKERS = 60


def extract_scale(s):
    if s[0] != "s":
        return None
    try:
        return int(s[1:])
    except ValueError:
        return None


def makedirs(name, mode=0o777, exist_ok=False):
    # always ok if exists
    return prev_makedirs(name, mode, exist_ok=True)


os.makedirs = makedirs


def downscale_block(in_array, out_array, factor, block):

    dims = len(factor)
    in_data = in_array.to_ndarray(block.read_roi, fill_value=0)

    in_shape = daisy.Coordinate(in_data.shape[-dims:])
    assert in_shape.is_multiple_of(factor)

    n_channels = len(in_data.shape) - dims
    if n_channels >= 1:
        factor = (1,) * n_channels + factor

    if in_data.dtype == np.uint64:
        slices = tuple(slice(k // 2, None, k) for k in factor)
        out_data = in_data[slices]
    else:
        out_data = skimage.measure.block_reduce(in_data, factor, np.mean)

    try:
        out_array[block.write_roi] = out_data
    except Exception:
        logger.critical("Failed to write to %s", block.write_roi)
        raise

    return 0


def downscale(in_array, out_array, factor, write_size, workers=DEFAULT_WORKERS):

    logger.info("Downsampling by factor %s", factor)

    dims = in_array.roi.dims()
    block_roi = daisy.Roi((0,) * dims, write_size)

    logger.info("Processing ROI %s with blocks %s", out_array.roi, block_roi)

    daisy.run_blockwise(
        out_array.roi,
        block_roi,
        block_roi,
        process_function=lambda b: downscale_block(in_array, out_array, factor, b),
        read_write_conflict=False,
        num_workers=workers,
        max_retries=0,
        fit="shrink",
    )


def create_scale_pyramid(
    in_file, in_ds_name, scales, chunk_shape, workers=DEFAULT_WORKERS
):

    ds = zarr.open(in_file)

    # make sure in_ds_name points to a dataset
    try:
        daisy.open_ds(in_file, in_ds_name)
    except Exception:
        raise RuntimeError("%s does not seem to be a dataset" % in_ds_name)

    lowest_scale = extract_scale(in_ds_name.split("/")[-1])

    if lowest_scale is None:

        ds_name = in_ds_name + "/s0"

        logger.info("Moving %s to %s", in_ds_name, ds_name)
        ds.store.rename(in_ds_name, in_ds_name + "__tmp")
        ds.store.rename(in_ds_name + "__tmp", ds_name)

        lowest_scale = 0

    else:

        ds_name = in_ds_name
        in_ds_name = "/".join(in_ds_name.split("/")[:-1])

    logger.info("Scaling %s by a factor of %s", in_file, scales)

    prev_array = daisy.open_ds(in_file, ds_name)

    if prev_array.n_channel_dims == 0:
        num_channels = 1
        ndims = len(prev_array.shape)
    elif prev_array.n_channel_dims == 1:
        num_channels = prev_array.shape[0]
        ndims = len(prev_array.shape) - 1
    else:
        raise RuntimeError(">1 channel dimension not yet implemented")

    if chunk_shape is None:
        chunk_shape = daisy.Coordinate(prev_array.data.chunks)
        logger.info("Reusing chunk shape of %s for new datasets" % (chunk_shape,))
    elif prev_array.n_channel_dims:
        if np.isscalar(chunk_shape):
            chunk_shape = daisy.Coordinate((num_channels,) + (chunk_shape,) * ndims)
        elif len(chunk_shape) == prev_array.shape:
            if chunk_shape[0] != num_channels:
                raise ValueError(f"Chunk shape does not match channel count: first dim must be {num_channels} or omitted")
            chunk_shape = daisy.Coordinate(chunk_shape)
        elif len(chunk_shape) == ndims:
            chunk_shape = daisy.Coordinate((num_channels, *chunk_shape))
        else:
            raise ValueError("Chunk shape must be a scalar, or match the array dimensions, or match the non-channel array dimensions")
    else:
        if np.isscalar(chunk_shape):
            chunk_shape = daisy.Coordinate((chunk_shape,) * ndims)
        elif len(chunk_shape) == ndims:
            chunk_shape = daisy.Coordinate(chunk_shape)
        else:
            raise ValueError("Chunk shape must be a scalar, or match the array dimensions")

    for scale_num, scale in enumerate(scales, lowest_scale + 1):
        if np.isscalar(scale):
            scale = daisy.Coordinate((scale,) * ndims)
        elif len(scale) == ndims:
            scale = daisy.Coordinate(scale)
        else:
            raise ValueError("Scale must a scalar, or list with the same length as the non-channel dimensions")

        next_voxel_size = prev_array.voxel_size * scale
        next_total_roi = prev_array.roi.snap_to_grid(next_voxel_size, mode="grow")
        next_write_size = chunk_shape * next_voxel_size

        logger.info("Next voxel size: %s", next_voxel_size)
        logger.info("Next total ROI: %s", next_total_roi)
        logger.info("Next chunk size: %s", next_write_size)

        next_ds_name = in_ds_name + "/s" + str(scale_num)
        logger.info("Preparing %s", next_ds_name)

        next_array = daisy.prepare_ds(
            in_file,
            next_ds_name,
            total_roi=next_total_roi,
            voxel_size=next_voxel_size,
            write_size=next_write_size,
            dtype=prev_array.dtype,
            num_channels=num_channels,
        )

        downscale(prev_array, next_array, scale, next_write_size, workers)

        prev_array = next_array


def parse_scales(s):
    return [parse_chunk(lvl) for lvl in s.split(";")]


def parse_chunk(s):
    try:
        return int(s)
    except ValueError:
        return [int(c.strip()) for c in s.split(",")]


def main(in_args=None):
    parser = argparse.ArgumentParser(
        description="Create a scale pyramid for a zarr/N5 container."
    )

    parser.add_argument("file", help="The input container")
    parser.add_argument(
        "array", help="The path to the array (/dataset) within the container"
    )
    parser.add_argument(
        "scales",
        type=parse_scales,
        help=(
            "Scale levels. Each scale level is separated by colons, "
            "and can be given as a single integer (for isotropic scaling) "
            "or a comma-separated list of integers for anisotropic. "
            "Scale levels refer to the voxel size of the previous scale, "
            "not the base scale. "
            "e.g. '2,2,1;2,2,1;2;2;2;2;2'"
        ),
    )
    parser.add_argument(
        "--chunk-shape",
        "-c",
        type=parse_chunk,
        help=(
            "The size of a chunk in voxels in the output. "
            "By default, re-uses the source chunking."
        ),
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of workers, default {DEFAULT_WORKERS}",
    )
    parser.add_argument("--log-file", "-l", help="Log file path (appends if exists).")

    args = parser.parse_args(in_args)
    log_kwargs = {"level": logging.INFO}
    if args.log_file:
        log_kwargs["filename"] = args.log_file

    logging.basicConfig(**log_kwargs)

    create_scale_pyramid(
        args.file, args.array, args.scales, args.chunk_shape, args.workers
    )


if __name__ == "__main__":
    main()
