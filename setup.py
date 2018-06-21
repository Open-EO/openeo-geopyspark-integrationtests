from setuptools import setup

setup(
    name='openeo-integration-tests',
    test_suite='tests',
    tests_require=['requests','openeo-udf']
)
