

# openeo-geopyspark-integrationtests

Integration test suite that uses the openeo-python-client
to test various use cases against an OpenEO backend.

## Installation

TODO

## Usage

To run the test suite against a certain OpenEO backend:
specify the backend base URL in environment variable `ENDPOINT`
and run the tests. 
For example:

    ENDPOINT=http://localhost:8080/
    pytest

