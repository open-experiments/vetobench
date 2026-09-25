"""HTTP clients for the OpenShift inference endpoints (OpenAI-compatible, e.g. vLLM)."""

from __future__ import annotations

import httpx

from ..config import Settings, Upstream


class UpstreamError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"upstream returned HTTP {status}: {body[:500]}")
        self.status, self.body = status, body


class UpstreamPool:
    """One pooled AsyncClient per upstream; routes by served model name."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self._transport = transport  # injectable for tests
        self._clients: dict[str, httpx.AsyncClient] = {}

    def client(self, up: Upstream) -> httpx.AsyncClient:
        if up.name not in self._clients:
            self._clients[up.name] = httpx.AsyncClient(
                base_url=up.base_url.rstrip("/") + "/",
                headers={"Authorization": f"Bearer {up.api_key}"},
                timeout=httpx.Timeout(up.timeout_s, connect=30.0),
                verify=up.verify_tls,
                limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
                transport=self._transport,
            )
        return self._clients[up.name]

    async def post(self, model: str, path: str, payload: dict,
                   timeout_s: float | None = None) -> httpx.Response:
        up = self.settings.upstream_for(model)
        kwargs = {"timeout": timeout_s} if timeout_s else {}
        return await self.client(up).post(path.lstrip("/"), json={**payload, "model": model}, **kwargs)

    async def chat(self, model: str, payload: dict, timeout_s: float | None = None) -> dict:
        """ChatBackend implementation used by judges: raises on non-2xx."""
        r = await self.post(model, "chat/completions", payload, timeout_s)
        if r.status_code >= 300:
            raise UpstreamError(r.status_code, r.text)
        return r.json()

    async def completion(self, model: str, payload: dict, timeout_s: float | None = None) -> dict:
        r = await self.post(model, "completions", payload, timeout_s)
        if r.status_code >= 300:
            raise UpstreamError(r.status_code, r.text)
        return r.json()

    async def list_models(self) -> dict[str, list[str] | str]:
        out: dict[str, list[str] | str] = {}
        for up in self.settings.upstreams:
            try:
                r = await self.client(up).get("models")
                r.raise_for_status()
                out[up.name] = [m["id"] for m in r.json().get("data", [])]
            except Exception as e:  # report per upstream; one bad route should not hide the others
                out[up.name] = f"error: {e}"
        return out

    async def aclose(self) -> None:
        for c in self._clients.values():
            await c.aclose()
        self._clients.clear()
