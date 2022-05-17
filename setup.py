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
        "openeo>=0.8.3a2.*",
        "pytest-xdist",
        "xarray",
        "pyproj",
        "h5netcdf",
        "geopandas",
        "pystac>=1.0.0"
    ],
)
