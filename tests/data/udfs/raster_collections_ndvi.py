import xarray


def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:
    """Compute the NDVI based on sentinel2 tiles"""
    red = cube.sel(bands="B04")
    nir = cube.sel(bands="B08")
    ndvi = (nir - red) / (nir + red)

    return ndvi
