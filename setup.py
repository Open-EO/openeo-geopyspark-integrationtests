from setuptools import setup

setup(
    name="openeo-geopyspark-integrationtests",
    test_suite="tests",
    install_requires=[
        "requests",
        "pytest",
        "numpy",
        "scipy",
        "rasterio",
        "schema",
        "pytest-timeout",
        "shapely",
        "openeo>=0.31.0.a2.dev",  # TODO: drop ".a2.dev" suffix once openeo 0.31.0 is released
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
    ],
)
