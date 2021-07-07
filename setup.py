from setuptools import setup

tests_require = [
    'requests',
    'pytest',
    'numpy',
    'scipy',
    'rasterio',
    'schema',
    'pytest-timeout',
    'shapely',
    'openeo',
    'pytest-xdist',
]

setup(
    name='openeo-integration-tests',
    test_suite='tests',
    extras_require={
        "dev": tests_require,
    },
)
