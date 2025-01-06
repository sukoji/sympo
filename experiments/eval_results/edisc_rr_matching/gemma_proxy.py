"""
Gemma vLLM 서버 앞단 프록시.
- 클라이언트 코드(get_llm 등)를 수정하지 않고도, 과도한 max_tokens(65000)를
  vLLM의 max_model_len(32768) 안으로 맞춰주기 위한 얇은 경유 레이어.
- POST /v1/chat/completions, POST /v1/completions 에서 max_tokens를 상한값으로 캡.
- 그 외 엔드포인트(/v1/models 등)는 단순 forward.

기동 예:
  python3 eval_results/edisc_rr_matching/gemma_proxy.py --upstream http://localhost:8081 --port 8091 --cap 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn


def create_app(upstream: str, max_tokens_cap: int) -> FastAPI:
    app = FastAPI()
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    def cap_payload(body: Dict[str, Any]) -> Dict[str, Any]:
        mt = body.get("max_tokens")
        if isinstance(mt, int) and mt > max_tokens_cap:
            body["max_tokens"] = max_tokens_cap
        # max_completion_tokens (OpenAI 신규 필드)도 캡
        mct = body.get("max_completion_tokens")
        if isinstance(mct, int) and mct > max_tokens_cap:
            body["max_completion_tokens"] = max_tokens_cap
        return body

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        raw = await req.body()
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        body = cap_payload(body)
        try:
            r = await client.post(f"{upstream}/v1/chat/completions", json=body)
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": {"message": f"upstream error: {e}"}})
        # 스트리밍 대응이 필요할 수 있으나 symPO는 non-stream → 단순 전달
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    @app.post("/v1/completions")
    async def completions(req: Request):
        raw = await req.body()
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        body = cap_payload(body)
        try:
            r = await client.post(f"{upstream}/v1/completions", json=body)
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": {"message": f"upstream error: {e}"}})
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    @app.get("/v1/models")
    async def models():
        try:
            r = await client.get(f"{upstream}/v1/models")
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": {"message": f"upstream error: {e}"}})
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    @app.get("/health")
    async def health():
        return {"status": "ok", "upstream": upstream, "max_tokens_cap": max_tokens_cap}

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://localhost:8081")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cap", type=int, default=8000,
                    help="max_tokens 상한값 (vLLM max_model_len - 안전 마진 이하)")
    args = ap.parse_args()
    print(f"[gemma_proxy] upstream={args.upstream} port={args.port} max_tokens_cap={args.cap}", flush=True)
    app = create_app(args.upstream, args.cap)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
