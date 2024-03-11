import numpy as np
import xarray as xr

from scipy.ndimage import (
    distance_transform_edt,
    binary_erosion,
    label,
    generate_binary_structure,
)
from openeo.udf import XarrayDataCube
from xarray.ufuncs import isnan as ufuncs_isnan


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:

    def sieve_filter(array, threshold):
        # Label connected components
        structure = generate_binary_structure(2, 1)  # 2D structure for 4-connectivity
        labeled_array, num = label(array, structure)

        component_sizes = np.bincount(labeled_array.ravel())

        large_components = np.where(component_sizes > threshold)[0]

        # Create a mask to keep only large components
        mask = np.isin(labeled_array, large_components)

        # Apply the mask to original array
        sieved_array = array * mask

        return sieved_array

    cube_array: xr.DataArray = cube.get_array()
    cube_array = cube_array.transpose("t", "bands", "y", "x")

    # clouds = np.logical_or(np.logical_or(np.logical_and(cube_array <= 11, cube_array >= 8), cube_array == 3), cube_array <= 1).isel(bands=0)
    clouds = np.logical_or(
        np.logical_and(cube_array < 11, cube_array >= 8), cube_array == 3
    ).isel(bands=0)

    # Calculate the Distance To Cloud score
    # Erode
    # Define a function to apply binary erosion
    def erode(image):
        return ~binary_erosion(image, iterations=3, border_value=1)

    # Use apply_ufunc to apply the erosion operation
    eroded = xr.apply_ufunc(
        erode,  # function to apply
        clouds,  # input DataArray
        input_core_dims=[["y", "x"]],  # dimensions over which to apply function
        output_core_dims=[["y", "x"]],  # dimensions of the output
        vectorize=True,  # vectorize the function over non-core dimensions
        dask="parallelized",  # enable dask parallelization
        output_dtypes=[np.int32],  # data type of the output
        # kwargs={'selem': er}  # additional keyword arguments to pass to erode
    )

    # Distance to cloud
    d = xr.apply_ufunc(
        distance_transform_edt,
        eroded,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32],
    )
    d = xr.where(d > 100, 100, d)
    d = xr.where(d < 10, 0, d)
    d = d / 100

    d_da = xr.DataArray(
        d,
        coords={
            "t": cube_array.coords["t"],
            "y": cube_array.coords["y"],
            "x": cube_array.coords["x"],
        },
        dims=["t", "y", "x"],
    )

    d_da = d_da.expand_dims(
        dim={
            "bands": cube_array.coords["bands"],
        },
    )

    d_da = d_da.transpose("t", "bands", "y", "x")

    return XarrayDataCube(d_da)
