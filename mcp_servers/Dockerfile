# mcp_servers/Dockerfile
# Shared image for all 4 APEX MCP servers (paper-mcp, graph-mcp, sim-mcp,
# patent-mcp) — same base, same deps, same repo-wide import layout. Which
# server actually runs is selected per-container via MCP_SERVER_MODULE, so
# each still starts, scales, and fails independently as its own service.
#
# Build context is the repo root (see docker-compose.yml), since these
# servers import sibling packages like `database.*`.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MCP_TRANSPORT=http
CMD ["sh", "-c", "python -m ${MCP_SERVER_MODULE}"]
