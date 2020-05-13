import os


def get_openeo_base_url(version="0.4.0"):
    try:
        endpoint = os.environ["ENDPOINT"].rstrip("/")
    except Exception:
        raise RuntimeError("Environment variable 'ENDPOINT' should be set"
                           " with URL pointing to OpenEO backend to test against"
                           " (e.g. 'http://localhost:8080/' or 'http://openeo-dev.vgt.vito.be/')")
    return "{e}/openeo/{v}".format(e=endpoint.rstrip("/"), v=version)
