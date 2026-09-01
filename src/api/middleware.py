import time

from fastapi import Request

from src.monitoring.config import get_serving_settings
from src.monitoring.serving import normalize_path, observe_request, should_ignore_path

SERVING_CFG = get_serving_settings()

def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) *1000, 2)

async def serving_monitoring_middleware(request: Request, call_next):
    """
    Record latency, status and exception metrics for serving requests.

    Monitoring endpoints and other configured paths are excluded to prevent
    self-observation from distorting serving metrics.
    """
    if not SERVING_CFG.get("enabled", True):
        return await call_next(request)

    raw_path = request.url.path

    if should_ignore_path(raw_path, SERVING_CFG.get("ignored_paths")):
        return await call_next(request)

    method = request.method
    path = normalize_path(raw_path, SERVING_CFG.get("track_paths"))
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        latency_seconds = time.perf_counter() - start
        observe_request(
            method=method,
            path=path,
            status_code=status_code,
            latency_seconds=latency_seconds,
        )


