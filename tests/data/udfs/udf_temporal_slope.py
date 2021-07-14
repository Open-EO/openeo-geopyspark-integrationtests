import xarray

from openeo.udf import XarrayDataCube


def apply_hypercube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    """Reduce the time dimension for each tile and compute min, mean, max and sum for each pixel over time."""
    array: xarray.DataArray = cube.get_array()
    result = xarray.concat(
        [array.min(dim='t'), array.max(dim='t'), array.sum(dim='t'), array.mean(dim='t')],
        dim='bands'
    )
    return XarrayDataCube(result)
