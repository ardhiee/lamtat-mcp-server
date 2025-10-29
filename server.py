"""LamTat MCP server with document ingestion, Bedrock (Cohere) embeddings, and OpenSearch indexing."""

from __future__ import annotations

import base64
import binascii
import os
import uuid
import json
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Optional, List, Dict

import boto3
import logging
from fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# ---------- Optional dependencies ----------
try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover
    DocumentConverter = None  # type: ignore

try:
    from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
except ImportError:  # pragma: no cover
    AWSV4SignerAuth = None  # type: ignore
    OpenSearch = None  # type: ignore
    RequestsHttpConnection = None  # type: ignore

# ---------- App / server ----------
APP_NAME = os.getenv("APP_NAME", "lamtat-mcp-server")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "6565"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

mcp = FastMCP(
    name=APP_NAME,
    host=HOST,
    port=PORT,
    stateless_http=True,
    json_response=True,
)

# ---------- S3 ----------
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_CLIENT = boto3.client("s3") if S3_BUCKET_NAME else None

# ---------- AWS / Bedrock ----------
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
BEDROCK_REGION = os.getenv("BEDROCK_REGION") or AWS_REGION
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")  # e.g. global.cohere.embed-v4:0 or your profile ARN
# Default to document embeddings for chunk storage
BEDROCK_INPUT_TYPE = (os.getenv("BEDROCK_INPUT_TYPE") or "search_document").strip()

# Embedding types: keep single type unless you handle multi-type responses
_embedding_types_env = os.getenv("BEDROCK_EMBEDDING_TYPES")
if _embedding_types_env:
    BEDROCK_EMBEDDING_TYPES: List[str] = [p.strip() for p in _embedding_types_env.split(",") if p.strip()]
else:
    BEDROCK_EMBEDDING_TYPES = ["float"]

# Output dimension: default 1024 (override via env). Set to None to omit.
_output_dimension_env = os.getenv("BEDROCK_OUTPUT_DIMENSION")
if _output_dimension_env:
    try:
        BEDROCK_OUTPUT_DIMENSION: Optional[int] = int(_output_dimension_env)
    except ValueError as exc:  # pragma: no cover
        raise ValueError("BEDROCK_OUTPUT_DIMENSION must be an integer") from exc
else:
    # If your index is 1536 set this env; otherwise default 1024 (common for cohere v4 in OS).
    BEDROCK_OUTPUT_DIMENSION = 1024

BEDROCK_TRUNCATE = (os.getenv("BEDROCK_TRUNCATE") or "RIGHT").strip()  # RIGHT | LEFT | NONE

if BEDROCK_REGION:
    try:
        BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        BEDROCK_ERROR = None
    except Exception as exc:  # pragma: no cover
        BEDROCK_CLIENT = None
        BEDROCK_ERROR = f"Failed to initialise Bedrock client: {exc}"
else:
    BEDROCK_CLIENT = None
    BEDROCK_ERROR = "BEDROCK_REGION is not configured"

# ---------- OpenSearch ----------
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")  # https://xxxxx.aoss.amazonaws.com
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX")
if AWS_REGION and OPENSEARCH_ENDPOINT and OpenSearch and AWSV4SignerAuth and RequestsHttpConnection:
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
    else:  # pragma: no cover
        OPENSEARCH_CLIENT = None
        OPENSEARCH_ERROR = "Unable to obtain AWS credentials for OpenSearch"
else:
    OPENSEARCH_CLIENT = None
    OPENSEARCH_ERROR = "OpenSearch dependencies or configuration missing"

# ---------- Chunking ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

DOC_CONVERTER = DocumentConverter() if DocumentConverter is not None else None


# ---------- Routes ----------
@mcp.custom_route("/", methods=["GET"])
async def root(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ---------- Helpers ----------
def _require_envs():
    if S3_CLIENT is None or not S3_BUCKET_NAME:
        raise RuntimeError("S3 bucket is not configured (set S3_BUCKET_NAME)")
    if BEDROCK_CLIENT is None:
        raise RuntimeError(BEDROCK_ERROR or "Bedrock client unavailable")
    if not BEDROCK_MODEL_ID:
        raise RuntimeError("BEDROCK_MODEL_ID is not configured (use global.cohere.embed-v4:0 or your profile ARN)")
    if OPENSEARCH_CLIENT is None:
        raise RuntimeError(OPENSEARCH_ERROR or "OpenSearch client unavailable")
    if not OPENSEARCH_INDEX:
        raise RuntimeError("OPENSEARCH_INDEX is not configured")


def _store_bytes(team: str, filename: str, raw_bytes: bytes, content_type: Optional[str] = None) -> dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_filename = filename or "document.bin"
    key = f"raw/{team}/{timestamp}_{safe_filename}"

    metadata = {
        "team": team,
        "original_filename": safe_filename,
        "length_bytes": str(len(raw_bytes)),
    }

    put_kwargs: Dict[str, Any] = {
        "Bucket": S3_BUCKET_NAME,
        "Key": key,
        "Body": raw_bytes,
        "Metadata": metadata,
    }
    if content_type:
        put_kwargs["ContentType"] = content_type

    S3_CLIENT.put_object(**put_kwargs)  # type: ignore[arg-type]

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
        except Exception:
            pass
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("utf-8", errors="ignore")


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    size = max(size, 1)
    overlap = max(0, min(overlap, size - 1))
    chunks: List[str] = []
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
    """Embed text chunks via Bedrock (Cohere Embed v4) with safe batching."""
    if BEDROCK_CLIENT is None:
        raise RuntimeError(BEDROCK_ERROR or "Bedrock client unavailable")
    if not BEDROCK_MODEL_ID:
        raise RuntimeError("BEDROCK_MODEL_ID is not configured")
    if not chunks:
        return []

    # 1) Clean & cap each chunk to avoid payload rejections
    MAX_CHARS = int(os.getenv("EMBEDDING_MAX_CHARS", "4000"))  # conservative cap
    cleaned = [c.strip()[:MAX_CHARS] for c in chunks if isinstance(c, str) and c.strip()]
    if not cleaned:
        return []

    # 2) Batch to avoid “invalid parameter combination” on oversized arrays
    BATCH = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    out: list[list[float]] = []

    for i in range(0, len(cleaned), BATCH):
        batch = cleaned[i:i + BATCH]

        body = {
            "input_type": "search_document",           # REQUIRED
            "texts": batch,                            # text-only
            "output_dimension": int(os.getenv("BEDROCK_OUTPUT_DIMENSION", "1024")),
            # keep minimal; add embedding_types or truncate later only if needed
        }

        logger.info(
            "Invoking Bedrock v4",
            extra={
                "modelId": BEDROCK_MODEL_ID,
                "batch_size": len(batch),
                "first_text_len": len(batch[0]) if batch else 0,
                "output_dimension": body["output_dimension"],
            },
        )

        try:
            resp = BEDROCK_CLIENT.invoke_model(
                modelId=BEDROCK_MODEL_ID,                  # e.g. global.cohere.embed-v4:0 (profile)
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="*/*",
            )
        except Exception as e:
            # surface a precise error to the tool without killing the stream
            raise RuntimeError(f"Bedrock invoke_model failed (batch {i//BATCH}): {e}") from e

        try:
            payload = json.loads(resp["body"].read())
        except Exception as e:
            raise RuntimeError(f"Failed to parse Bedrock response (batch {i//BATCH}): {e}") from e

        embs = payload.get("embeddings")
        if not isinstance(embs, list):
            raise RuntimeError(f"Unexpected Bedrock response (no 'embeddings'): {payload}")
        out.extend(embs)

    return out

def _index_chunks(
    *,
    team: str,
    doc_id: str,
    filename: str,
    s3_key: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if OPENSEARCH_CLIENT is None:
        raise RuntimeError(OPENSEARCH_ERROR or "OpenSearch client unavailable")
    if not OPENSEARCH_INDEX:
        raise RuntimeError("OPENSEARCH_INDEX is not configured")

    if len(chunks) != len(embeddings):
        raise RuntimeError(f"Chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}")

    timestamp = datetime.now(timezone.utc).isoformat()
    indexed: List[Dict[str, Any]] = []

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
            "repo": (metadata or {}).get("repo"),
            "path": (metadata or {}).get("path") or filename,
            "owner_team": team,
            "allowed_teams": (metadata or {}).get("allowed_teams") or [team],
            "uploaded_by": (metadata or {}).get("uploaded_by"),
            "tags": (metadata or {}).get("tags"),
            "created_at": timestamp,
        }
        # For opensearch-py 2.x the param is `body=...`
        OPENSEARCH_CLIENT.index(index=OPENSEARCH_INDEX, id=chunk_id, body=body)
        indexed.append({"chunk_id": chunk_id, "vector_dimensions": len(vector)})

    return indexed


def _process_document(
    team: str,
    filename: str,
    raw_bytes: bytes,
    s3_key: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    doc_id = str(uuid.uuid4())
    text = _extract_text(raw_bytes, filename)
    chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        return {"doc_id": doc_id, "chunks_indexed": 0, "chunk_ids": []}
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
    return {"doc_id": doc_id, "chunks_indexed": len(indexed), "chunks": indexed}


# ---------- Tools ----------
@mcp.tool
async def store_docs(
    team: str,
    files: List[Dict[str, Any]],
    ctx: Context,
) -> List[Dict[str, Any]]:
    """Persist base64 documents, generate Bedrock embeddings, and index into OpenSearch."""
    try:
        _require_envs()
        if not team:
            raise ValueError("team is required")
        if not files:
            raise ValueError("files must be a non-empty list")

        results: List[Dict[str, Any]] = []

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

            results.append({"s3": s3_result, "opensearch": opensearch_result})

        await ctx.info(f"Stored {len(results)} document(s) for team {team} and indexed chunks")
        return results

    except Exception as e:
        # Return a clean tool error without closing streams abruptly
        await ctx.error(f"store_docs_failed: {e}")
        # Surface structured error result to the caller
        return [{"error": str(e), "ok": False}]


@mcp.tool
async def process_data(uri: str, ctx: Context):
    """Fetch content from a resource URI, summarize via client, and return summary."""
    try:
        await ctx.info(f"Processing {uri}...")
        data = await ctx.read_resource(uri)
        summary = await ctx.sample(f"Summarize: {data.content[:500]}")
        return summary.text
    except Exception as e:
        await ctx.error(f"process_data_failed: {e}")
        return {"ok": False, "error": str(e)}


@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numeric values and return the product."""
    return a * b


# ---------- Entrypoint ----------
def run() -> None:
    """Start the FastMCP server."""
    mcp.run("streamable-http")


if __name__ == "__main__":
    run()
