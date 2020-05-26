from setuptools import setup

tests_require = [
    'requests',
    'pytest',
    'numpy',
    'scipy',
    'rasterio',
    'scema',
    'pytest-timeout',
    'shapely',
    'openeo',
]

setup(
    name='openeo-integration-tests',
    test_suite='tests',
    setup_requires=['pytest-runner'],
    tests_require=tests_require,
    extras_require={
        "dev": tests_require,
    },
)
