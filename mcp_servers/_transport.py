# mcp_servers/_transport.py
# Shared transport for all 4 MCP servers — stdio (default, for local MCP
# client subprocess use, e.g. Claude Desktop) or streamable-HTTP (for
# running as an independent, network-reachable Docker service). Selected
# by the MCP_TRANSPORT env var so no server-specific code has to change.

import contextlib
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route


async def run_server(server: Server, name: str) -> None:
    """Runs `server` over stdio (default) or streamable-HTTP, per MCP_TRANSPORT."""
    transport = os.getenv('MCP_TRANSPORT', 'stdio')

    if transport == 'stdio':
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
        return

    if transport != 'http':
        raise ValueError(f"Unknown MCP_TRANSPORT: {transport!r} (expected 'stdio' or 'http')")

    import uvicorn

    session_manager = StreamableHTTPSessionManager(app=server, stateless=True)

    async def healthz(request):
        return JSONResponse({'status': 'ok', 'server': name})

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with session_manager.run():
            yield

    app = Starlette(
        routes=[
            Route('/healthz', healthz),
            Mount('/mcp', app=session_manager.handle_request),
        ],
        lifespan=lifespan,
    )

    port = int(os.getenv('MCP_PORT', '8800'))
    config = uvicorn.Config(app, host='0.0.0.0', port=port, log_level='info')
    await uvicorn.Server(config).serve()
