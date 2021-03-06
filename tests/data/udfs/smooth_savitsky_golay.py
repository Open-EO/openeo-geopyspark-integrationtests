import xarray

from openeo.udf import XarrayDataCube


def apply_hypercube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    from scipy.signal import savgol_filter

    array: xarray.DataArray = cube.get_array()
    filled = array.interpolate_na(dim='t')
    smoothed_array = savgol_filter(filled.values, 5, 2, axis=0)
    return XarrayDataCube(xarray.DataArray(smoothed_array, dims=array.dims, coords=array.coords))
