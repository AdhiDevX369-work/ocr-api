import json
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Exception raised for errors in LLM server communication."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

class LLMClient:
    def __init__(
        self,
        base_url: str = settings.llm_server_url,
        api_key: str = settings.llm_api_key,
        default_backend: str = settings.default_backend,
        default_model: str = settings.default_model,
        timeout: float = settings.llm_timeout
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_backend = default_backend
        self.default_model = default_model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _get_headers(self, backend: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        target_backend = backend or self.default_backend
        if target_backend in ("llama-cpp", "llama_cpp", "llama.cpp"):
            headers["X-LLM-Backend"] = "llama_cpp"
        elif target_backend == "ollama":
            headers["X-LLM-Backend"] = "ollama"

        # Always read latest API key from settings or init
        api_key = settings.llm_api_key or self.api_key
        if api_key:
            headers["X-API-Key"] = api_key
        return headers

    def _convert_to_ollama_payload(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        backend: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Converts OpenAI format messages to native Ollama format for /api/chat fallback."""
        ollama_msgs = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                ollama_msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts = []
                images = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if "," in url:
                                b64 = url.split(",", 1)[1]
                            else:
                                b64 = url
                            if b64:
                                images.append(b64)
                turn = {"role": role, "content": " ".join(text_parts)}
                if images:
                    turn["images"] = images
                ollama_msgs.append(turn)
            else:
                ollama_msgs.append({"role": role, "content": str(content)})

        payload = {
            "model": model or self.default_model,
            "messages": ollama_msgs,
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if backend:
            payload["backend"] = "llama_cpp" if backend in ("llama-cpp", "llama_cpp", "llama.cpp") else backend
        return payload

    async def check_health(self) -> Dict[str, Any]:
        client = await self.get_client()
        headers = self._get_headers()
        try:
            resp = await client.get("/health", headers=headers, timeout=5.0)
            if resp.status_code == 200:
                res = resp.json()
                if "status" not in res:
                    res["status"] = "healthy"
                return res
            
            # Fallback for native Ollama server (returns 200 on / or /api/version)
            resp_root = await client.get("/", headers=headers, timeout=5.0)
            if resp_root.status_code == 200:
                return {
                    "status": "healthy",
                    "server": "ollama",
                    "detail": resp_root.text.strip()
                }
            return {"status": "unhealthy", "http_status": resp.status_code, "detail": resp.text}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        backend: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        client = await self.get_client()
        headers = self._get_headers(backend)

        # 1. Try standard OpenAI format /v1/chat/completions first
        openai_payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        if max_tokens:
            openai_payload["max_tokens"] = max_tokens

        try:
            resp = await client.post("/v1/chat/completions", json=openai_payload, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            
            logger.warning(f"/v1/chat/completions returned {resp.status_code}, trying /api/chat fallback...")
        except Exception as e:
            logger.warning(f"/v1/chat/completions request failed: {e}, trying /api/chat fallback...")

        # 2. Fallback to Ollama format /api/chat
        ollama_payload = self._convert_to_ollama_payload(
            messages=messages,
            model=model,
            backend=backend,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )

        try:
            resp = await client.post("/api/chat", json=ollama_payload, headers=headers)
            if resp.status_code == 200:
                res_data = resp.json()
                assistant_content = res_data.get("message", {}).get("content", "")
                return {
                    "model": res_data.get("model", model or self.default_model),
                    "choices": [
                        {"message": {"role": "assistant", "content": assistant_content}}
                    ],
                    "usage": {}
                }
            
            error_text = resp.text
            if "image input is not supported" in error_text or "mmproj" in error_text:
                error_msg = (
                    "Port 8080 llama-server is running a text model without Vision Projector (--mmproj). "
                    "Please restart llama-server on port 8080 with '--mmproj <path_to_mmproj.gguf>'."
                )
            else:
                error_msg = f"LLM Gateway error (HTTP {resp.status_code}): {error_text}"

            logger.error(error_msg)
            raise LLMClientError(error_msg, status_code=resp.status_code)
        except httpx.RequestError as e:
            logger.error(f"Failed to communicate with LLM server at {self.base_url}: {e}")
            raise LLMClientError(f"Could not connect to LLM server on port 8100: {str(e)}", status_code=502)

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        backend: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048
    ) -> AsyncGenerator[str, None]:
        client = await self.get_client()
        headers = self._get_headers(backend)

        # 1. Try streaming via /v1/chat/completions
        openai_payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        if max_tokens:
            openai_payload["max_tokens"] = max_tokens

        v1_success = False
        try:
            async with client.stream("POST", "/v1/chat/completions", json=openai_payload, headers=headers) as resp:
                if resp.status_code == 200:
                    v1_success = True
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                    return
        except Exception as e:
            logger.warning(f"/v1/chat/completions stream failed: {e}, falling back to /api/chat...")

        if not v1_success:
            # 2. Fallback streaming via Ollama /api/chat NDJSON stream
            ollama_payload = self._convert_to_ollama_payload(
                messages=messages,
                model=model,
                backend=backend,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            try:
                async with client.stream("POST", "/api/chat", json=ollama_payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        raise LLMClientError(f"LLM Stream error ({resp.status_code}): {error_body.decode()}", status_code=resp.status_code)

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            msg = chunk.get("message", {})
                            content = msg.get("content", "")
                            if content:
                                yield content
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
            except httpx.RequestError as e:
                logger.error(f"Streaming failed: {e}")
                raise LLMClientError(f"Stream error communicating with port 8100: {str(e)}", status_code=502)

llm_client = LLMClient()
