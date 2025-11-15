from fastapi import Request
from fastapi.responses import JSONResponse
from core.config import settings

async def check_wms_auth(request: Request, call_next):
    """Middleware to check API key authentication."""
    
    # Allow root path without auth
    if request.url.path in ["/", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    # Check for API key in headers
    auth_key = request.headers.get("x-api-key")
    
    if auth_key and auth_key == settings.WMS_API_KEY:
        response = await call_next(request)
        return response
    else:
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized or missing API key."}
        )