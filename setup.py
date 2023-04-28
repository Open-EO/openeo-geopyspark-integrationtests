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
        "openeo>=0.17.0a3.*",
        "pytest-xdist",
        "xarray<2022.6.0",
        "pyproj",
        "h5netcdf",
        "geopandas",
        "pystac>=1.0.0",
        "hvac>=1.0.2",
    ],
)
