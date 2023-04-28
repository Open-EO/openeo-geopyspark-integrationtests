

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

### Authentication

Most tests in the test suite require an authenticated connection to the back-end.
The test suite setup/fixtures tries to do the right thing automatically:
- In an automated/Jenkins/CI context: use OIDC client credentials auth with a "service account".
  Credentials should be provided through environment variables:
  look for `OPENEO_JENKINS_SERVICE_ACCOUNT_` in `conftest.py` for inspiration
- Running locally as developer: using refresh tokens is supported.
  Make sure you have valid refresh tokens in you environment.
  If not, the device code flow will be used as fallback,
  but with an extremely short poll timeout by default,
  making it humanly impossible to actually complete the device code flow.
  The timeout is short by default to avoid the confusion
  with tests hanging inexplicably for minutes before failing.
  Temporarily increase the timeout with env var
  `OPENEO_OIDC_DEVICE_CODE_MAX_POLL_TIME=300`.

### Filtering

Pytest provides [various options](https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests)
to run a subset or just a single test.
Some examples (that can be combined):

-   select by substring of the name of a test with the `-k` option:

        pytest -k test_health

-   run tests that do not involve batch jobs (which are decorated with `@pytest.mark.batchjob`)

        pytest -m "not batchjob"

### Debugging and troubleshooting tips

- The `tmp_path` fixture provides a [fresh temporary folder for a test to work in](https://docs.pytest.org/en/latest/tmpdir.html).
It is cleaned up automatically, except for the last 3 runs, so you can inspect
generated files post-mortem. The temp folders are typically situated under `/tmp/pytest-of-$USERNAME`.

- To disable pytest's default log/output capturing, to better see what is going on in "real time", add these options:

        --capture=no --log-cli-level=INFO
