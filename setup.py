from setuptools import setup

setup(
    name='openeo-integration-tests',
    test_suite='tests',
    setup_requires=['pytest-runner'],
    tests_require=['requests','openeo-udf>=0.0.9.post0','pytest<5.1.0','scipy'],
)
