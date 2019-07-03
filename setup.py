from setuptools import setup

setup(
    name='openeo-integration-tests',
    test_suite='tests',
    setup_requires=['pytest-runner'],
    tests_require=['requests','openeo-udf','pytest'],
)
