"""Minimal FastMCP server exposing example tools."""

from __future__ import annotations

import base64
import binascii
import os
from datetime import datetime, timezone
from typing import Any

import boto3
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

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_CLIENT = boto3.client("s3") if S3_BUCKET_NAME else None




@mcp.custom_route('/', methods=['GET'])
async def root(_: Request) -> JSONResponse:
    return JSONResponse({'status': 'ok'})


@mcp.custom_route('/health', methods=['GET'])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({'status': 'ok'})



def _store_bytes(team: str, filename: str, raw_bytes: bytes, content_type: str | None = None) -> dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_filename = filename or "document.bin"
    key = f"raw/{team}/{timestamp}_{safe_filename}"

    metadata = {
        "team": team,
        "original_filename": safe_filename,
        "length_bytes": str(len(raw_bytes)),
    }

    put_kwargs: dict[str, Any] = {
        "Bucket": S3_BUCKET_NAME,
        "Key": key,
        "Body": raw_bytes,
        "Metadata": metadata,
    }
    if content_type:
        put_kwargs["ContentType"] = content_type

    S3_CLIENT.put_object(**put_kwargs)

    return {
        "bucket": S3_BUCKET_NAME,
        "key": key,
        "team": team,
        "size_bytes": len(raw_bytes),
        "content_type": content_type,
    }



@mcp.tool
async def store_docs(
    team: str,
    files: list[dict[str, Any]],
    ctx: Context,
) -> list[dict[str, Any]]:
    """Persist one or more base64 documents for a team."""

    if S3_CLIENT is None or not S3_BUCKET_NAME:
        raise RuntimeError("S3 bucket is not configured (set S3_BUCKET_NAME env var)")
    if not team:
        raise ValueError("team is required")
    if not files:
        raise ValueError("files must be a non-empty list")

    results: list[dict[str, Any]] = []
    for idx, file_info in enumerate(files):
        if not isinstance(file_info, dict):
            raise ValueError(f"files[{idx}] must be an object with filename/content_base64")

        filename = file_info.get("filename")
        content_base64 = file_info.get("content_base64")
        content_type = file_info.get("content_type")

        if not filename or not isinstance(filename, str):
            raise ValueError(f"files[{idx}].filename is required")
        if not content_base64 or not isinstance(content_base64, str):
            raise ValueError(f"files[{idx}].content_base64 is required")

        try:
            raw_bytes = base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"files[{idx}].content_base64 is not valid base64") from exc

        result = _store_bytes(team, filename, raw_bytes, content_type)
        results.append(result)

    await ctx.info(f"Stored {len(results)} document(s) for team {team}")
    return results


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
