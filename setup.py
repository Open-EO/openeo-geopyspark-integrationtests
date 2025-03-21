from setuptools import setup

setup(
    name="openeo-geopyspark-integrationtests",
    test_suite="tests",
    install_requires=[
        "requests",
        "pytest",
        "numpy",
        "scipy",
        # Avoid wheel-less rasterio releases on Python 3.8 and lower https://github.com/rasterio/rasterio/issues/3168
        "rasterio; python_version>='3.9'",
        "rasterio<1.3.11; python_version<'3.9'",
        "schema",
        "pytest-timeout",
        "shapely",
        "openeo>=0.39.0",
        "openeo_driver>=0.39.1.dev",
        "pytest-xdist",
        "xarray>=2022.0.0",
        "pyproj",
        "h5netcdf",
        "geopandas",
        "pystac>=1.0.0",
        "hvac>=1.0.2",
        "netCDF4",
        "rioxarray",
        "cftime",
        "jsonschema",
        "rio_cogeo",
        "pydantic~=1.0",
    ],
)
