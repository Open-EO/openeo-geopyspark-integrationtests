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
        "openeo",
        "pytest-xdist",
        "xarray",
        "pyproj",
        "h5netcdf",
    ],
)
