import logging
import os
from typing import Any

import pytest
import requests

import openeo
from openeo.capabilities import ComparableVersion

_log = logging.getLogger(__name__)

def get_openeo_base_url(version: str = "1.1.0"):
    try:
        endpoint = os.environ["ENDPOINT"].rstrip("/")
    except Exception:
        raise RuntimeError("Environment variable 'ENDPOINT' should be set"
                           " with URL pointing to OpenEO backend to test against"
                           " (e.g. 'http://localhost:8080/' or 'http://openeo-dev.vgt.vito.be/')")
    return "{e}/openeo/{v}".format(e=endpoint.rstrip("/"), v=version)


def get_tsservice_base_url():
    tsservice_endpoint = os.environ.get("TSSERVICE_ENDPOINT")

    if not tsservice_endpoint:
        raise RuntimeError("Environment variable 'TSSERVICE_ENDPOINT' should be set"
                           " with URL pointing to OpenEO/tsservice backend to test against"
                           " (e.g. 'http://localhost:8155/')")

    return tsservice_endpoint


@pytest.fixture(params=[
    "1.1.0",
])
def api_version(request) -> ComparableVersion:
    return ComparableVersion(request.param)


@pytest.fixture
def api_base_url(api_version):
    return get_openeo_base_url(str(api_version))


@pytest.fixture
def requests_session(request) -> requests.Session:
    """
    Fixture to create a `requests.Session` that automatically injects a query parameter in API URLs
    referencing the currently running test.
    Simplifies cross-referencing between integration tests and flask/YARN logs
    """
    session = requests.Session()
    session.params["_origin"] = f"{request.session.name}/{request.node.name}"
    return session


@pytest.fixture
def connection(api_base_url, requests_session) -> openeo.Connection:
    return openeo.connect(api_base_url, session=requests_session)


@pytest.fixture
def connection100(requests_session) -> openeo.Connection:
    return openeo.connect(get_openeo_base_url("1.0.0"), session=requests_session)



def _redact(x: Any) -> Any:
    """Helper to redact sensitive items in nested dictionaries."""

    def is_sensitive_key(key: Any) -> bool:
        return isinstance(key, str) and any(s in key.lower() for s in {"secret", "token", "password"})

    if isinstance(x, dict):
        return {k: "-redacted-" if is_sensitive_key(k) and v else _redact(v) for k, v in x.items()}
    else:
        return x


# TODO #6 larger scope than "function" for this fixture?
# TODO #6 better name for this fixture?
@pytest.fixture
def auth_connection(connection) -> openeo.Connection:
    """
    Fixture to authenticate the connection,
    attempting different methods to support multiple run modes:
    automated jenkins/CI run (client credentials), developer running locally (device flow/refresh tokens)
    """
    # Try to extract Jenkins service account credentials from env (e.g. set from Jenkinsfile)
    service_account_creds = {
        "provider_id": os.environ.get("OPENEO_JENKINS_SERVICE_ACCOUNT_PROVIDER_ID"),
        "client_id": os.environ.get("OPENEO_JENKINS_SERVICE_ACCOUNT_CLIENT_ID"),
        "client_secret": os.environ.get("OPENEO_JENKINS_SERVICE_ACCOUNT_CLIENT_SECRET"),
    }
    _log.info(f"Extracted Jenkins service account credentials: {_redact(service_account_creds)}")
    if all(service_account_creds.values()):
        _log.info(f"Using client credentials auth with Jenkins service account: {_redact(service_account_creds)}")
        connection.authenticate_oidc_client_credentials(**service_account_creds, store_refresh_token=False)
        return connection

    # Try classic OIDC refresh tokens + device code flow:
    # allows developers to run the integration tests locally with own user.
    _log.info("Trying auth `connection.authenticate_oidc()` with refresh tokens + device code flow")
    connection.authenticate_oidc(
        # Note the really short default max poll time to fail fast by default
        # (when nobody is watching the device code flow instructions).
        max_poll_time=int(os.environ.get("OPENEO_OIDC_DEVICE_CODE_MAX_POLL_TIME") or 1),
        store_refresh_token=True,
    )
    return connection



@pytest.fixture
def tsservice_base_url():
    return get_tsservice_base_url()
