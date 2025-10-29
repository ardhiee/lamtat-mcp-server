
"""LamTat MCP server with document ingestion, Bedrock (Cohere) embeddings, and OpenSearch indexing."""

from __future__ import annotations

import base64
import binascii
import os
import uuid
import json
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import boto3
import logging
from fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency
    DocumentConverter = None  # type: ignore[assignment]

try:
    from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
except ImportError:  # pragma: no cover - optional dependency
    AWSV4SignerAuth = None  # type: ignore[assignment]
    OpenSearch = None  # type: ignore[assignment]
    RequestsHttpConnection = None  # type: ignore[assignment]


APP_NAME = os.getenv("APP_NAME", "lamtat-mcp-server")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "6565"))

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name=APP_NAME,
    host=HOST,
    port=PORT,
    stateless_http=True,
    json_response=True,
)

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_CLIENT = boto3.client("s3") if S3_BUCKET_NAME else None

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
BEDROCK_REGION = os.getenv("BEDROCK_REGION") or AWS_REGION
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
BEDROCK_INPUT_TYPE = os.getenv("BEDROCK_INPUT_TYPE")
_embedding_types_env = os.getenv("BEDROCK_EMBEDDING_TYPES")
if _embedding_types_env:
    BEDROCK_EMBEDDING_TYPES = [part.strip() for part in _embedding_types_env.split(",") if part.strip()]
elif "amazon.titan-embed" in (BEDROCK_MODEL_ID or ""):
    BEDROCK_EMBEDDING_TYPES = ["float"]
else:
    BEDROCK_EMBEDDING_TYPES: list[str] = []
_output_dimension_env = os.getenv("BEDROCK_OUTPUT_DIMENSION")
if _output_dimension_env:
    try:
        BEDROCK_OUTPUT_DIMENSION: int | None = int(_output_dimension_env)
    except ValueError as exc:  # pragma: no cover - invalid configuration
        raise ValueError("BEDROCK_OUTPUT_DIMENSION must be an integer") from exc
else:
    BEDROCK_OUTPUT_DIMENSION = None
BEDROCK_TRUNCATE = os.getenv("BEDROCK_TRUNCATE")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

DOC_CONVERTER = DocumentConverter() if DocumentConverter is not None else None

if BEDROCK_REGION:
    try:
        BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        BEDROCK_ERROR = None
    except Exception as exc:  # pragma: no cover - configuration issue
        BEDROCK_CLIENT = None
        BEDROCK_ERROR = f"Failed to initialise Bedrock client: {exc}"
else:
    BEDROCK_CLIENT = None
    BEDROCK_ERROR = "BEDROCK_REGION is not configured"


if (
    AWS_REGION
    and OPENSEARCH_ENDPOINT
    and OpenSearch is not None
    and AWSV4SignerAuth is not None
    and RequestsHttpConnection is not None
):
    session = boto3.Session(region_name=AWS_REGION)
    credentials = session.get_credentials()
    if credentials is not None:
        host = OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", "")
        auth = AWSV4SignerAuth(credentials, AWS_REGION, service="aoss")
        OPENSEARCH_CLIENT = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
        OPENSEARCH_ERROR = None
    else:  # pragma: no cover - credentials misconfiguration
        OPENSEARCH_CLIENT = None
        OPENSEARCH_ERROR = "Unable to obtain AWS credentials for OpenSearch"
else:
    OPENSEARCH_CLIENT = None
    OPENSEARCH_ERROR = "OpenSearch dependencies or configuration missing"



@mcp.custom_route("/", methods=["GET"])
async def root(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


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


def _extract_text(raw_bytes: bytes, filename: str) -> str:
    if DOC_CONVERTER is not None:
        try:
            document = DOC_CONVERTER.read(BytesIO(raw_bytes), file_name=filename)
            text_content = getattr(document, "text_content", None)
            if isinstance(text_content, str) and text_content.strip():
                return text_content
            if hasattr(document, "export_to_text"):
                exported = document.export_to_text()
                if isinstance(exported, str) and exported.strip():
                    return exported
        except Exception:  # pragma: no cover - fallback on failure
            pass
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("utf-8", errors="ignore")


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    size = max(size, 1)
    overlap = max(0, min(overlap, size - 1))
    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + size, length)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks


def _embed_chunks(chunks: list[str]) -> list[list[float]]:
    if BEDROCK_CLIENT is None:
        raise RuntimeError(BEDROCK_ERROR or "Bedrock client unavailable")
    if not chunks:
        return []

    body = {
        "texts": chunks,
    }
    if BEDROCK_INPUT_TYPE:
        body["input_type"] = BEDROCK_INPUT_TYPE

    logger.info("Invoking Bedrock", extra={
        "bedrock_model_id": BEDROCK_MODEL_ID,
        "bedrock_request": body,
    })

    invoke_kwargs = {
        "body": json.dumps(body).encode("utf-8"),
        "contentType": "application/json",
        "accept": "*/*",
    }
    if not BEDROCK_MODEL_ID:
        raise RuntimeError("BEDROCK_MODEL_ID is not configured")
    if ":inference-profile/" in BEDROCK_MODEL_ID:
        invoke_kwargs["inferenceProfileArn"] = BEDROCK_MODEL_ID
    else:
        invoke_kwargs["modelId"] = BEDROCK_MODEL_ID

    response = BEDROCK_CLIENT.invoke_model(**invoke_kwargs)
    payload = json.loads(response["body"].read())
    embeddings = payload.get("embeddings")
    if embeddings is None:
        raise RuntimeError("Bedrock response missing embeddings")
    return embeddings


def _index_chunks(
    *,
    team: str,
    doc_id: str,
    filename: str,
    s3_key: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if OPENSEARCH_CLIENT is None:
        raise RuntimeError(OPENSEARCH_ERROR or "OpenSearch client unavailable")
    if not OPENSEARCH_INDEX:
        raise RuntimeError("OPENSEARCH_INDEX is not configured")

    timestamp = datetime.now(timezone.utc).isoformat()
    indexed: list[dict[str, Any]] = []
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{doc_id}-{idx:04d}"
        body = {
            "tenant_id": team,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": chunk,
            "vector": vector,
            "source": "mcp",
            "source_uri": s3_key,
            "repo": metadata.get("repo") if metadata else None,
            "path": metadata.get("path") if metadata else filename,
            "owner_team": team,
            "allowed_teams": metadata.get("allowed_teams") if metadata else [team],
            "uploaded_by": metadata.get("uploaded_by") if metadata else None,
            "tags": metadata.get("tags") if metadata else None,
            "created_at": timestamp,
        }
        OPENSEARCH_CLIENT.index(index=OPENSEARCH_INDEX, id=chunk_id, body=body)
        indexed.append({
            "chunk_id": chunk_id,
            "vector_dimensions": len(vector),
        })
    return indexed


def _process_document(
    team: str,
    filename: str,
    raw_bytes: bytes,
    s3_key: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc_id = str(uuid.uuid4())
    text = _extract_text(raw_bytes, filename)
    chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        return {
            "doc_id": doc_id,
            "chunks_indexed": 0,
            "chunk_ids": [],
        }
    embeddings = _embed_chunks(chunks)
    indexed = _index_chunks(
        team=team,
        doc_id=doc_id,
        filename=filename,
        s3_key=s3_key,
        chunks=chunks,
        embeddings=embeddings,
        metadata=metadata,
    )
    return {
        "doc_id": doc_id,
        "chunks_indexed": len(indexed),
        "chunks": indexed,
    }


@mcp.tool
async def store_docs(
    team: str,
    files: list[dict[str, Any]],
    ctx: Context,
) -> list[dict[str, Any]]:
    """Persist base64 documents, generate Bedrock embeddings, and index into OpenSearch."""

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

        s3_result = _store_bytes(team, filename, raw_bytes, content_type)

        ingestion_metadata = {
            "uploaded_by": file_info.get("uploaded_by"),
            "allowed_teams": file_info.get("allowed_teams"),
            "tags": file_info.get("tags"),
            "repo": file_info.get("repo"),
            "path": file_info.get("path"),
        }

        opensearch_result = _process_document(
            team=team,
            filename=filename,
            raw_bytes=raw_bytes,
            s3_key=s3_result["key"],
            metadata=ingestion_metadata,
        )

        results.append(
            {
                "s3": s3_result,
                "opensearch": opensearch_result,
            }
        )

    await ctx.info(f"Stored {len(results)} document(s) for team {team} and indexed chunks")
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
