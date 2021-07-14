import xarray

from openeo_udf.api.datacube import DataCube


def apply_hypercube(cube: DataCube, context: dict) -> DataCube:
    """Reduce the time dimension for each tile and compute min, mean, max and sum for each pixel
    over time.
    Each raster tile in the udf data object will be reduced by time. Minimum, maximum, mean and sum are
    computed for each pixel over time.
    Args:
        udf_data (UdfData): The UDF data object that contains raster and vector tiles
    Returns:
        This function will not return anything, the UdfData object "udf_data" must be used to store the resulting
        data.
    """
    # The list of tiles that were created
    array: xarray.DataArray = cube.get_array()
    result = xarray.concat(
        [array.min(dim='t'), array.max(dim='t'), array.sum(dim='t'), array.mean(dim='t')],
        dim='bands'
    )
    return DataCube(result)
