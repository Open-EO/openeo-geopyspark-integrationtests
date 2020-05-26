

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

    export ENDPOINT=http://localhost:8080/
    pytest


Pytest provides [various options](https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests) to run a subset or just a single test.
For example, select by substring of the name of a test with the `-k` option:

    pytest -k test_health


### Debugging and troubleshooting tips

- The `tmp_path` fixture provides a [fresh temporary folder for a test to work in](https://docs.pytest.org/en/latest/tmpdir.html). 
It is cleaned up automatically, except for the last 3 runs, so you can inspect
generated files post-mortem. The temp folders are typically situated under `/tmp/pytest-of-$USERNAME`.

- To disable pytest's default log/output capturing, to better see what is going on in "real time", add these options:

        --capture=no --log-cli-level=INFO