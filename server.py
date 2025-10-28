"""Minimal FastMCP server exposing example tools."""

from __future__ import annotations

import os

from fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

APP_NAME = os.getenv("APP_NAME", "lamtat-mcp-server")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "6565"))

mcp = FastMCP(
    name=APP_NAME,
    host=HOST,
    port=PORT,
    stateless_http=True,
    json_response=True,
)




@mcp.custom_route('/', methods=['GET'])
async def root(_: Request) -> JSONResponse:
    return JSONResponse({'status': 'ok'})


@mcp.custom_route('/health', methods=['GET'])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({'status': 'ok'})

@mcp.tool
async def process_data(uri: str, ctx: Context):
    """Fetch content from a resource URI, summarize via client, and return summary."""

    await ctx.info(f"Processing {uri}...")

    data = await ctx.read_resource(uri)

    summary = await ctx.sample(f"Summarize: {data.content[:500]}")

    return summary.text


@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numeric values and return the product."""

    return a * b


def run() -> None:
    """Start the FastMCP server."""

    mcp.run("streamable-http")


if __name__ == "__main__":
    run()
