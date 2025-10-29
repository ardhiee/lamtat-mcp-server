# lamtat-mcp-server


A tiny FastMCP server exposing two example tools — `process_data` (resource fetch + summary) and `multiply` (numerical product) — configured for stateless HTTP transport.

## Code
```python
from fastmcp import FastMCP, Context

mcp = FastMCP(
    name="lamtat-mcp-server",
    host="0.0.0.0",
    port=6565,
    stateless_http=True,
    json_response=True,
)

@mcp.tool
async def process_data(uri: str, ctx: Context):
    await ctx.info(f"Processing {uri}...")
    data = await ctx.read_resource(uri)
    summary = await ctx.sample(f"Summarize: {data.content[:500]}")
    return summary.text

@mcp.tool
def multiply(a: float, b: float) -> float:
    return a * b

mcp.run("streamable-http")
```

The repository keeps a thin wrapper (`server.py`) so you can ship the server as a module or container and configure host/port via environment variables.

### store_docs tool
Use `store_docs` to archive files in S3 under a team-specific prefix:

```python
@mcp.tool
async def store_docs(team: str, files: list[dict[str, Any]], ctx: Context):
    ...
```

Provide a `files` list with at least one element; for single-file uploads pass a list containing one {"filename", "content_base64"} entry (optionally `content_type`).

Required environment variables:
- `S3_BUCKET_NAME`
- Optional: `AWS_REGION` / `AWS_DEFAULT_REGION` if not already set for the task role

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
lamtat-mcp-server
```

Environment variables:
- `APP_NAME` – display name (default `lamtat-mcp-server`)
- `HOST` – bind address (default `0.0.0.0`)
- `PORT` – listen port (default `6565`)

## Container
```bash
docker build -t lamtat-mcp-server:latest .
docker run --rm -p 6565:6565 lamtat-mcp-server:latest
```

## Deploy with AWS Copilot
1. Ensure the Copilot CLI is installed and AWS credentials are configured.
2. Initialise (skip if already done):
   ```bash
   copilot init --app lamtat-mcp --name lamtat-mcp-server --type "Load Balanced Web Service" --dockerfile Dockerfile
   ```
3. Deploy, e.g. to `test`:
   ```bash
   copilot env init --name test --profile default --region us-east-1
   copilot deploy --name lamtat-mcp-server --env test
   ```

The manifest at `copilot/lamtat-mcp-server/manifest.yml` exposes port 6565 with the default FastMCP routing.

## Copilot scripts
- `scripts/deploy.sh` – bootstrap app/env (if needed) and deploy service
- `scripts/destroy.sh` – remove service, env, and app

For ECS on x86 (Fargate), ensure images are built for `linux/amd64`. Copilot handles this via the manifest platform setting; for manual builds use `docker build --platform linux/amd64 ...`.
