"""LamTat MCP server — tenant-protected ingest + search (keyword & KNN)
with checksum de-dup (pre-upload) and robust PDF text extraction:
- Use Docling if installed
- Else fallback to PyPDF to reliably extract text

Env vars (typical):
  S3_BUCKET_NAME
  AWS_REGION
  BEDROCK_REGION
  BEDROCK_MODEL_ID                 e.g. global.cohere.embed-v4:0 (or inference profile ARN)
  BEDROCK_OUTPUT_DIMENSION=1024
  OPENSEARCH_ENDPOINT              https://<id>.<region>.aoss.amazonaws.com
  OPENSEARCH_INDEX                 e.g. knowledge-chunks

Optional:
  CHUNK_SIZE=1000
  CHUNK_OVERLAP=100
  EMBEDDING_MAX_CHARS=4000
  EMBEDDING_BATCH_SIZE=64
  LOG_LEVEL=INFO
"""

from __future__ import annotations
import base64, binascii, hashlib, json, logging, os, uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional

import boto3
from fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# ---------- optional (text extraction) ----------
DOC_EXTRACTOR = "none"
try:
    from docling.document_converter import DocumentConverter  # type: ignore
    DOC_EXTRACTOR = "docling"
except Exception:
    DocumentConverter = None  # type: ignore
    try:
        # lightweight fallback
        from pypdf import PdfReader  # type: ignore
        DOC_EXTRACTOR = "pypdf"
    except Exception:
        PdfReader = None  # type: ignore

# ---------- optional (OpenSearch client) ----------
try:
    from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
except ImportError:
    AWSV4SignerAuth = OpenSearch = RequestsHttpConnection = None  # type: ignore

# ---------- app / server ----------
APP_NAME = os.getenv("APP_NAME", "lamtat-mcp-server")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "6565"))
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

mcp = FastMCP(name=APP_NAME, host=HOST, port=PORT, stateless_http=True, json_response=True)

# ---------- env / clients ----------
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_CLIENT = boto3.client("s3") if S3_BUCKET_NAME else None

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
BEDROCK_REGION = os.getenv("BEDROCK_REGION") or AWS_REGION
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
BEDROCK_OUTPUT_DIMENSION = int(os.getenv("BEDROCK_OUTPUT_DIMENSION", "1024"))
BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION) if BEDROCK_REGION else None

OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX")

if AWS_REGION and OPENSEARCH_ENDPOINT and OpenSearch and AWSV4SignerAuth and RequestsHttpConnection:
    session = boto3.Session(region_name=AWS_REGION)
    creds = session.get_credentials()
    if creds:
        host = OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", "")
        auth = AWSV4SignerAuth(creds, AWS_REGION, service="aoss")
        OPENSEARCH_CLIENT = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
    else:
        OPENSEARCH_CLIENT = None
else:
    OPENSEARCH_CLIENT = None

# ---------- config ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
EMBEDDING_MAX_CHARS = int(os.getenv("EMBEDDING_MAX_CHARS", "4000"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))


# ---------- routes ----------
@mcp.custom_route("/", methods=["GET"])
async def root(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "extractor": DOC_EXTRACTOR})

@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ---------- utils & guards ----------
def _require_envs() -> None:
    if not S3_CLIENT or not S3_BUCKET_NAME:
        raise RuntimeError("S3 bucket/config missing")
    if not BEDROCK_CLIENT or not BEDROCK_MODEL_ID:
        raise RuntimeError("Bedrock client/model not configured")
    if not OPENSEARCH_CLIENT or not OPENSEARCH_INDEX:
        raise RuntimeError("OpenSearch client/index not configured")


def ensure_index() -> None:
    """Create KNN index w/ mapping if missing (idempotent) + ensure `checksum` keyword field exists."""
    dim = BEDROCK_OUTPUT_DIMENSION
    body = {
        "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 100}},
        "mappings": {
            "properties": {
                "tenant_id":     {"type": "keyword"},
                "doc_id":        {"type": "keyword"},
                "chunk_id":      {"type": "keyword"},
                "text":          {"type": "text"},
                "vector":        {"type": "knn_vector", "dimension": dim},
                "source":        {"type": "keyword"},
                "source_uri":    {"type": "keyword"},
                "repo":          {"type": "keyword"},
                "path":          {"type": "keyword"},
                "commit_sha":    {"type": "keyword"},
                "owner_team":    {"type": "keyword"},
                "allowed_teams": {"type": "keyword"},
                "uploaded_by":   {"type": "keyword"},
                "tags":          {"type": "keyword"},
                "created_at":    {"type": "date"},
                "checksum":      {"type": "keyword"},
            }
        },
    }
    try:
        OPENSEARCH_CLIENT.indices.create(index=OPENSEARCH_INDEX, body=body)  # type: ignore
        logger.info(f"Created index: {OPENSEARCH_INDEX}")
    except Exception as e:
        msg = str(e)
        if "resource_already_exists_exception" in msg:
            logger.info(f"Index exists: {OPENSEARCH_INDEX}")
            # ensure checksum exists
            try:
                mapping = OPENSEARCH_CLIENT.indices.get_mapping(index=OPENSEARCH_INDEX)  # type: ignore
                props = mapping[OPENSEARCH_INDEX]["mappings"].get("properties", {})
                if "checksum" not in props:
                    OPENSEARCH_CLIENT.indices.put_mapping(  # type: ignore
                        index=OPENSEARCH_INDEX, body={"properties": {"checksum": {"type": "keyword"}}}
                    )
                    logger.info("Added `checksum` to mapping.")
            except Exception as ie:
                raise RuntimeError(f"Failed to validate/extend mapping: {ie}")
        else:
            raise


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _find_existing_doc(team: str, checksum: str) -> Dict[str, Any] | None:
    """Return first (tenant_id, checksum) match or None."""
    body = {
        "size": 1,
        "query": {"bool": {"filter": [{"term": {"tenant_id": team}}, {"term": {"checksum": checksum}}]}},
        "_source": ["doc_id", "chunk_id", "source_uri"],
    }
    res = OPENSEARCH_CLIENT.search(index=OPENSEARCH_INDEX, body=body)  # type: ignore
    hits = res.get("hits", {}).get("hits", [])
    return hits[0] if hits else None


def _store_bytes(team: str, filename: str, raw: bytes, content_type: Optional[str]) -> Dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = filename or "document.bin"
    key = f"raw/{team}/{ts}_{safe}"
    put = {"Bucket": S3_BUCKET_NAME, "Key": key, "Body": raw, "Metadata": {"team": team, "original_filename": safe}}
    if content_type:
        put["ContentType"] = content_type
    S3_CLIENT.put_object(**put)  # type: ignore
    return {"bucket": S3_BUCKET_NAME, "key": key, "team": team, "size_bytes": len(raw), "content_type": content_type}


def _extract_text(raw: bytes, filename: str) -> str:
    # 1) Docling (if available)
    if DocumentConverter is not None:
        try:
            doc = DocumentConverter().read(BytesIO(raw), file_name=filename)
            t = getattr(doc, "text_content", None)
            if isinstance(t, str) and t.strip():
                logger.info("Extracted text via Docling", extra={"len": len(t)})
                return t
            if hasattr(doc, "export_to_text"):
                t = doc.export_to_text()
                if isinstance(t, str) and t.strip():
                    logger.info("Extracted text via Docling.export_to_text", extra={"len": len(t)})
                    return t
        except Exception as e:
            logger.warning(f"Docling extraction failed: {e}")

    # 2) PyPDF fallback (if available)
    if DOC_EXTRACTOR == "pypdf":
        try:
            from pypdf import PdfReader  # lazy import in case env changes
            import io
            reader = PdfReader(io.BytesIO(raw))
            t = "".join(page.extract_text() or "" for page in reader.pages)
            if t.strip():
                logger.info("Extracted text via PyPDF", extra={"len": len(t)})
                return t
        except Exception as e:
            logger.warning(f"PyPDF extraction failed: {e}")

    # 3) Last-resort binary decode (may be garbage)
    try:
        t = raw.decode("utf-8")
        logger.info("Extracted text via utf-8 decode", extra={"len": len(t)})
        return t
    except UnicodeDecodeError:
        t = raw.decode("utf-8", errors="ignore")
        logger.info("Extracted text via utf-8 ignore errors", extra={"len": len(t)})
        return t


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    t = text.strip()
    if not t:
        return []
    size = max(size, 1)
    overlap = max(0, min(overlap, size - 1))
    out, i, n = [], 0, len(t)
    while i < n:
        j = min(i + size, n)
        ch = t[i:j].strip()
        if ch:
            out.append(ch)
        if j == n:
            break
        i = j - overlap
    return out


def _embed_chunks(chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    cleaned = [c.strip()[:EMBEDDING_MAX_CHARS] for c in chunks if c and c.strip()]
    if not cleaned:
        return []
    vecs: List[List[float]] = []
    for i in range(0, len(cleaned), EMBEDDING_BATCH_SIZE):
        batch = cleaned[i:i+EMBEDDING_BATCH_SIZE]
        req = {"input_type": "search_document", "texts": batch, "output_dimension": BEDROCK_OUTPUT_DIMENSION}
        resp = BEDROCK_CLIENT.invoke_model(  # type: ignore
            modelId=BEDROCK_MODEL_ID, body=json.dumps(req).encode("utf-8"),
            contentType="application/json", accept="*/*"
        )
        payload = json.loads(resp["body"].read())
        embs = payload.get("embeddings")
        if isinstance(embs, dict):  # handle {"embeddings":{"float":[...]}}
            embs = embs.get("float") or next((v for v in embs.values() if isinstance(v, list)), None)
        if not isinstance(embs, list):
            raise RuntimeError(f"Bedrock response missing embeddings: {payload}")
        vecs.extend(embs)
    return vecs


def _index_chunks(
    *, team: str, doc_id: str, filename: str, s3_key: str,
    chunks: List[str], embeddings: List[List[float]],
    metadata: Optional[Dict[str, Any]], checksum: str
) -> List[Dict[str, Any]]:
    if len(chunks) != len(embeddings):
        raise RuntimeError("Chunks/embeddings mismatch")
    ts = datetime.now(timezone.utc).isoformat()
    results: List[Dict[str, Any]] = []
    for idx, (text, vec) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{doc_id}-{idx:04d}"
        body = {
            "tenant_id": team,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": text,
            "vector": vec,
            "source": "mcp",
            "source_uri": s3_key,
            "repo": (metadata or {}).get("repo"),
            "path": (metadata or {}).get("path") or filename,
            "owner_team": team,
            "allowed_teams": (metadata or {}).get("allowed_teams") or [team],
            "uploaded_by": (metadata or {}).get("uploaded_by"),
            "tags": (metadata or {}).get("tags"),
            "created_at": ts,
            "checksum": checksum,
        }
        resp = OPENSEARCH_CLIENT.index(index=OPENSEARCH_INDEX, body=body)  # type: ignore
        if resp.get("result") not in ("created", "updated"):
            raise RuntimeError(f"Indexing failed for {chunk_id}: {resp}")
        results.append({"chunk_id": chunk_id, "server_id": resp.get("_id"), "vector_dimensions": len(vec)})
    try:
        OPENSEARCH_CLIENT.indices.refresh(index=OPENSEARCH_INDEX)  # type: ignore
    except Exception:
        pass
    return results


def _team_filter(team: str) -> Dict[str, Any]:
    return {"bool": {"should": [{"term": {"tenant_id": team}}, {"term": {"allowed_teams": team}}], "minimum_should_match": 1}}


def _presign_s3(bucket: str, key: str, seconds: int = 3600) -> Optional[str]:
    if not bucket or not key or not S3_CLIENT:
        return None
    try:
        return S3_CLIENT.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=seconds)
    except Exception:
        return None


def _make_hit(hit: Dict[str, Any], requester_team: str) -> Dict[str, Any]:
    src = hit.get("_source", {}) or {}
    key = src.get("source_uri")
    owner = src.get("tenant_id")
    access = "owner" if requester_team == owner else ("shared" if requester_team in (src.get("allowed_teams") or []) else "unknown")
    preview = (src.get("text") or "")
    if len(preview) > 200:
        preview = preview[:200] + "…"
    return {
        "score": hit.get("_score"),
        "doc_id": src.get("doc_id"),
        "chunk_id": src.get("chunk_id"),
        "text_preview": preview,
        "citation": f"{src.get('doc_id')}::{src.get('chunk_id')}",
        "link": _presign_s3(S3_BUCKET_NAME, key) if key else None,
        "metadata": {
            "team_owner": owner, "access": access,
            "allowed_teams": src.get("allowed_teams"), "uploaded_by": src.get("uploaded_by"),
            "repo": src.get("repo"), "path": src.get("path"), "tags": src.get("tags"),
            "created_at": src.get("created_at"), "checksum": src.get("checksum"),
        },
    }


# ---------- processing ----------
def _process_document(team: str, filename: str, raw: bytes, s3_key: str, meta: Optional[Dict[str, Any]], checksum: str) -> Dict[str, Any]:
    doc_id = str(uuid.uuid4())
    text = _extract_text(raw, filename)
    chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        logger.warning("No chunks extracted; skipping embeddings/index", extra={"filename": filename})
        return {"doc_id": doc_id, "chunks_indexed": 0, "chunks": []}
    vecs = _embed_chunks(chunks)
    idxd = _index_chunks(team=team, doc_id=doc_id, filename=filename, s3_key=s3_key, chunks=chunks, embeddings=vecs, metadata=meta, checksum=checksum)
    return {"doc_id": doc_id, "chunks_indexed": len(idxd), "chunks": idxd}


# ---------- tools ----------
@mcp.tool
async def store_docs(team: str, files: List[Dict[str, Any]], ctx: Context) -> List[Dict[str, Any]]:
    """Persist base64 docs to S3, embed via Bedrock, index into AOSS (KNN) with checksum de-dup per team (pre-upload)."""
    try:
        _require_envs()
        ensure_index()
        if not team:
            raise ValueError("team is required")
        if not files:
            raise ValueError("files must be a non-empty list")

        results: List[Dict[str, Any]] = []
        for i, f in enumerate(files):
            if not isinstance(f, dict):
                raise ValueError(f"files[{i}] must be an object with filename/content_base64")
            filename = f.get("filename")
            content_b64 = f.get("content_base64")
            ctype = f.get("content_type")
            if not filename or not isinstance(filename, str):
                raise ValueError(f"files[{i}].filename is required")
            if not content_b64 or not isinstance(content_b64, str):
                raise ValueError(f"files[{i}].content_base64 is required")

            try:
                raw = base64.b64decode(content_b64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"files[{i}].content_base64 is not valid base64") from exc

            checksum = _sha256_hex(raw)

            # ---- de-dup BEFORE S3 upload
            existing = _find_existing_doc(team, checksum)
            if existing:
                src = existing.get("_source", {}) or {}
                results.append({
                    "duplicate": True, "reason": "checksum match", "team": team, "filename": filename, "checksum": checksum,
                    "existing_doc_id": src.get("doc_id"), "existing_chunk_id": src.get("chunk_id"), "existing_source_uri": src.get("source_uri"),
                })
                continue

            s3res = _store_bytes(team, filename, raw, ctype)
            meta = {
                "uploaded_by": f.get("uploaded_by"),
                "allowed_teams": f.get("allowed_teams"),
                "tags": f.get("tags"),
                "repo": f.get("repo"),
                "path": f.get("path"),
            }
            osres = _process_document(team=team, filename=filename, raw=raw, s3_key=s3res["key"], meta=meta, checksum=checksum)
            results.append({"duplicate": False, "s3": s3res, "opensearch": osres})

        await ctx.info(f"Processed {len(results)} file(s) for team {team}")
        return results
    except Exception as e:
        logger.error(f"store_docs_failed: {e}")
        return [{"ok": False, "error": str(e)}]


@mcp.tool
async def search_text(query: str, team: str, size: int = 5) -> Dict[str, Any]:
    """Keyword search limited to docs owned by or shared with `team` (mandatory)."""
    if not team:
        return {"ok": False, "error": "team is required"}
    body = {"size": size, "query": {"bool": {"must": [{"match": {"text": query}}], "filter": [_team_filter(team)]}}}
    res = OPENSEARCH_CLIENT.search(index=OPENSEARCH_INDEX, body=body)  # type: ignore
    hits = res.get("hits", {}).get("hits", [])
    return {"ok": True, "count": len(hits), "results": [_make_hit(h, team) for h in hits]}


@mcp.tool
async def semantic_search(query: str, team: str, size: int = 5) -> Dict[str, Any]:
    """KNN (semantic) search limited to docs owned by or shared with `team` (mandatory)."""
    if not team:
        return {"ok": False, "error": "team is required"}
    req = {"input_type": "search_document", "texts": [query], "output_dimension": BEDROCK_OUTPUT_DIMENSION}
    try:
        resp = BEDROCK_CLIENT.invoke_model(  # type: ignore
            modelId=BEDROCK_MODEL_ID, body=json.dumps(req).encode("utf-8"),
            contentType="application/json", accept="*/*"
        )
        payload = json.loads(resp["body"].read())
        embs = payload.get("embeddings")
        if isinstance(embs, dict):
            embs = embs.get("float") or next((v for v in embs.values() if isinstance(v, list)), None)
        if not isinstance(embs, list) or not embs:
            return {"ok": False, "error": f"Bad Bedrock embedding response: {payload}"}
        embedding = embs[0]
    except Exception as e:
        return {"ok": False, "error": f"Bedrock embed failed: {e}"}

    body = {"size": size, "query": {"knn": {"vector": {"vector": embedding, "k": size}, "filter": _team_filter(team)}}}
    try:
        res = OPENSEARCH_CLIENT.search(index=OPENSEARCH_INDEX, body=body)  # type: ignore
        hits = res.get("hits", {}).get("hits", [])
        return {"ok": True, "count": len(hits), "results": [_make_hit(h, team) for h in hits]}
    except Exception as e:
        return {"ok": False, "error": f"OpenSearch KNN failed: {e}"}


# --- quick debug tools ---
@mcp.tool
async def count_docs(team: str) -> dict:
    """Count docs visible to this team (owner or shared)."""
    if not team:
        return {"ok": False, "error": "team is required"}
    body = {"query": {"bool": {"filter": [_team_filter(team)]}}}
    res = OPENSEARCH_CLIENT.count(index=OPENSEARCH_INDEX, body=body)  # type: ignore
    return {"ok": True, "count": res.get("count", 0)}

@mcp.tool
async def show_docs(team: str, size: int = 3) -> dict:
    """Show a few docs (match_all) visible to the team to inspect text previews."""
    if not team:
        return {"ok": False, "error": "team is required"}
    body = {
        "size": size,
        "query": {"bool": {"must": [{"match_all": {}}], "filter": [_team_filter(team)]}},
        "_source": ["doc_id","chunk_id","text","source_uri","tenant_id","allowed_teams","checksum","created_at"],
    }
    res = OPENSEARCH_CLIENT.search(index=OPENSEARCH_INDEX, body=body)  # type: ignore
    hits = res.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        s = h.get("_source", {})
        txt = s.get("text") or ""
        preview = txt[:200] + ("…" if len(txt) > 200 else "")
        out.append({
            "doc_id": s.get("doc_id"),
            "chunk_id": s.get("chunk_id"),
            "text_preview": preview,
            "checksum": s.get("checksum"),
            "source_uri": s.get("source_uri"),
            "tenant_id": s.get("tenant_id"),
            "allowed_teams": s.get("allowed_teams"),
        })
    return {"ok": True, "count": len(out), "results": out}


# ---------- entrypoint ----------
def run() -> None:
    mcp.run("streamable-http")

if __name__ == "__main__":
    run()
