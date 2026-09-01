import os

from fastapi.security.api_key import APIKeyHeader
from fastapi import HTTPException, Security
from starlette.status import HTTP_403_FORBIDDEN



# Define the header name for the API Key
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Validate the API key supplied in the X-API-KEY header.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If the configured or supplied API key is missing or invalid.
    """
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Could not validate API Key",
    )
