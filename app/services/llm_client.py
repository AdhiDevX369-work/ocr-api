import json
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Exception raised for errors in direct LLM server communication."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

class LLMClient:
    def __init__(
        self,
        llama_cpp_url: str = settings.llama_cpp_url,
        ollama_url: str = settings.ollama_url,
        default_backend: str = settings.default_backend,
        default_model: str = settings.default_model,
        timeout: float = settings.llm_timeout
    ):
        self.llama_cpp_url = llama_cpp_url.rstrip("/")
        self.ollama_url = ollama_url.rstrip("/")
        self.default_backend = default_backend
        self.default_model = default_model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _get_timeout(self) -> httpx.Timeout:
        """Returns httpx Timeout configuration with 30s connect and 120s read timeout for large model loads."""
        return httpx.Timeout(
            timeout=self.timeout,
            connect=30.0,
            read=self.timeout,
            write=60.0,
            pool=30.0
        )

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._get_timeout(),
                limits=httpx.Limits(max_keepalive_connections=50, max_connections=200, keepalive_expiry=300.0)
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def get_backend_url(self, backend: Optional[str] = None) -> str:
        target = backend or self.default_backend
        if target in ("llama-cpp", "llama_cpp", "llama.cpp", "llamacpp"):
            return settings.llama_cpp_url.rstrip("/")
        elif target in ("llm-server", "llm_server", "llmserver", "gateway", "8100"):
            return settings.llm_server_url.rstrip("/")
        elif target == "ollama":
            return settings.ollama_url.rstrip("/")
        return settings.llm_server_url.rstrip("/")

    async def get_available_models(self, backend: Optional[str] = None) -> List[Dict[str, Any]]:
        """Queries the direct LLM backend for available/loaded models."""
        target_backend = backend or self.default_backend
        backend_url = self.get_backend_url(target_backend)
        client = await self.get_client()

        models_list: List[Dict[str, Any]] = []

        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["x-api-key"] = settings.llm_api_key

        try:
            if target_backend in ("llama-cpp", "llama_cpp", "llama.cpp", "llamacpp"):
                # Query llama-server /v1/models
                resp = await client.get(f"{backend_url}/v1/models", headers=headers, timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    for item in data:
                        models_list.append({
                            "id": item.get("id"),
                            "aliases": item.get("aliases", []),
                            "status": item.get("status", {}).get("value", "unknown"),
                            "input_modalities": item.get("architecture", {}).get("input_modalities", ["text"]),
                            "backend": "llama-cpp"
                        })
            elif target_backend in ("llm-server", "llm_server", "llmserver", "gateway", "8100"):
                # Query gateway llm-server /v1/models
                resp = await client.get(f"{backend_url}/v1/models", headers=headers, timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    for item in data:
                        models_list.append({
                            "id": item.get("id"),
                            "name": item.get("id"),
                            "aliases": [item.get("id")],
                            "status": "loaded",
                            "input_modalities": ["text", "image"],
                            "backend": "llm-server"
                        })
            else:
                # Query Ollama /api/tags
                resp = await client.get(f"{backend_url}/api/tags", headers=headers, timeout=5.0)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    for item in models:
                        models_list.append({
                            "id": item.get("name"),
                            "name": item.get("name"),
                            "aliases": [item.get("name"), item.get("model")],
                            "status": "loaded",
                            "details": item.get("details", {}),
                            "backend": "ollama"
                        })
                else:
                    # Try /v1/models on Ollama
                    resp_v1 = await client.get(f"{backend_url}/v1/models", headers=headers, timeout=5.0)
                    if resp_v1.status_code == 200:
                        data = resp_v1.json().get("data", [])
                        for item in data:
                            models_list.append({
                                "id": item.get("id"),
                                "aliases": [item.get("id")],
                                "status": "loaded",
                                "backend": "ollama"
                            })
        except Exception as e:
            logger.warning(f"Failed to query available models from [{target_backend}] at {backend_url}: {e}")

        return models_list

    async def pick_model(
        self,
        backend: Optional[str] = None,
        requested_model: Optional[str] = None,
        has_images: bool = False
    ) -> str:
        """Inspects available models on target server and selects the best candidate model."""
        target_backend = backend or self.default_backend
        available_models = await self.get_available_models(target_backend)

        if not available_models:
            fallback = requested_model or self.default_model
            logger.info(f"🤖 No models returned by [{target_backend}], using fallback model: '{fallback}'")
            return fallback

        vision_keywords = ["ministral-3", "ministral", "pixtral", "qwen2.5vl", "qwen2.5-vl", "qwen3-vl", "qwen3vl", "vl", "vision", "llava", "moondream", "gemma4", "minicpm-v", "llama3.2-vision", "bakllava"]

        # 1. If has_images=True, verify requested model supports vision or select a vision-capable model
        if has_images:
            # Check if requested model itself is a vision model
            if requested_model:
                req_lower = requested_model.lower()
                is_req_vision = any(vk in req_lower for vk in vision_keywords)
                if is_req_vision:
                    for m in available_models:
                        m_id = m.get("id", "")
                        m_aliases = [a.lower() for a in m.get("aliases", [])]
                        if m_id.lower() == req_lower or req_lower in m_aliases or req_lower in m_id.lower():
                            logger.info(f"🎯 Using requested vision model '{m_id}' on [{target_backend}]")
                            return m_id

            # Select first available vision-capable model
            for m in available_models:
                m_id = m.get("id", "").lower()
                modalities = m.get("input_modalities", [])
                if "image" in modalities or any(vk in m_id for vk in vision_keywords):
                    logger.info(f"👁️ Auto-selected vision-capable model '{m['id']}' on [{target_backend}] (requested model '{requested_model}' is text-only)")
                    return m["id"]

            logger.warning(f"⚠️ An image was provided but no known vision models found on [{target_backend}].")

        # 2. If requested model is specified and available in server tags, use it directly
        if requested_model:
            req_lower = requested_model.lower()
            for m in available_models:
                m_id = m.get("id", "")
                m_aliases = [a.lower() for a in m.get("aliases", [])]
                if m_id.lower() == req_lower or req_lower in m_aliases or req_lower in m_id.lower():
                    logger.info(f"🎯 Using requested model '{m_id}' on [{target_backend}]")
                    return m_id

        # 3. Prioritize 'loaded' model
        for m in available_models:
            if m.get("status") == "loaded":
                logger.info(f"⚡ Picked loaded model '{m['id']}' on [{target_backend}]")
                return m["id"]

        # 4. Fallback to first model in list
        if available_models:
            first_model = available_models[0].get("id", self.default_model)
            logger.info(f"ℹ️ Selected default available model '{first_model}' on [{target_backend}]")
            return first_model
        return self.default_model

    def _convert_to_ollama_payload(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
        stream: bool = False,
        json_mode: bool = False
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
                            b64 = url.strip().split(",")[-1]
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
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": max(temperature, 0.1) if temperature == 0.0 else temperature,
                "repeat_penalty": 1.1,
                "num_gpu": 99,
                "num_thread": 8,
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": max_tokens if max_tokens else 8192
            }
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    async def warmup_model(self, backend: Optional[str] = None, model: Optional[str] = None):
        """Asynchronously warm up / preload the vision model into VRAM."""
        target_backend = backend or self.default_backend
        selected_model = await self.pick_model(backend=target_backend, requested_model=model or self.default_model, has_images=True)
        logger.info(f"🔥 Pre-loading / Warming up vision model '{selected_model}' on [{target_backend}]...")
        try:
            if target_backend == "ollama":
                client = await self.get_client()
                headers = {"Content-Type": "application/json"}
                if settings.llm_api_key:
                    headers["x-api-key"] = settings.llm_api_key
                warmup_payload = {
                    "model": selected_model,
                    "messages": [{"role": "user", "content": "warmup"}],
                    "stream": False,
                    "keep_alive": settings.ollama_keep_alive,
                    "options": {
                        "num_gpu": 99,
                        "num_ctx": settings.ollama_num_ctx,
                        "num_predict": 1
                    }
                }
                resp = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json=warmup_payload,
                    headers=headers,
                    timeout=60.0
                )
                if resp.status_code == 200:
                    logger.info(f"⚡ Vision model '{selected_model}' successfully pre-loaded into GPU VRAM!")
                else:
                    logger.warning(f"⚠️ Model warmup returned HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"⚠️ Model warmup encountered an issue: {e}")

    async def check_health(self) -> Dict[str, Any]:
        """Performs connection & model discovery checks concurrently against llama.cpp & Ollama backends."""
        client = await self.get_client()

        llama_status = {"status": "unreachable", "url": self.llama_cpp_url, "models": []}
        ollama_status = {"status": "unreachable", "url": self.ollama_url, "models": []}

        async def _check_llama():
            nonlocal llama_status
            try:
                resp = await client.get(f"{self.llama_cpp_url}/v1/models", timeout=1.5)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    models = [m.get("id") for m in data if m.get("id")]
                    llama_status = {
                        "status": "healthy",
                        "url": self.llama_cpp_url,
                        "models": models
                    }
            except Exception as e:
                llama_status["error"] = str(e)

        async def _check_ollama():
            nonlocal ollama_status
            try:
                resp = await client.get(f"{self.ollama_url}/api/tags", timeout=1.5)
                if resp.status_code == 200:
                    models = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
                    ollama_status = {
                        "status": "healthy",
                        "url": self.ollama_url,
                        "models": models
                    }
            except Exception as e:
                ollama_status["error"] = str(e)

        await asyncio.gather(_check_llama(), _check_ollama(), return_exceptions=True)

        overall_healthy = (llama_status["status"] == "healthy" or ollama_status["status"] == "healthy")
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "backends": {
                "llama-cpp": llama_status,
                "ollama": ollama_status
            }
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        backend: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
        stream: bool = False,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        target_backend = backend or self.default_backend
        backend_url = self.get_backend_url(target_backend)
        client = await self.get_client()

        has_images = any(
            isinstance(m.get("content"), list) and any(
                isinstance(part, dict) and part.get("type") == "image_url" for part in m.get("content", [])
            ) for m in messages
        ) or any("images" in m and m["images"] for m in messages)

        selected_model = await self.pick_model(
            backend=target_backend,
            requested_model=model,
            has_images=has_images
        )

        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["x-api-key"] = settings.llm_api_key

        # 1. If backend is Ollama, use native Ollama endpoint /api/chat directly
        if target_backend == "ollama":
            ollama_payload = self._convert_to_ollama_payload(
                messages=messages,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                json_mode=json_mode
            )
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    resp = await client.post(f"{backend_url}/api/chat", json=ollama_payload, headers=headers, timeout=self._get_timeout())
                    if resp.status_code == 200:
                        res_data = resp.json()
                        msg_obj = res_data.get("message", {})
                        assistant_content = msg_obj.get("content", "")
                        thinking = msg_obj.get("thinking", "")
                        if not assistant_content and thinking:
                            assistant_content = f"> *Thinking:*\n> {thinking}\n"
                        return {
                            "model": res_data.get("model", selected_model),
                            "choices": [
                                {"message": {"role": "assistant", "content": assistant_content}}
                            ],
                            "usage": {}
                        }

                    if "loading model" in resp.text.lower() and attempt < max_retries - 1:
                        logger.info(f"⏳ Ollama is loading model into VRAM... Waiting 3s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(3.0)
                        continue

                    raise LLMClientError(f"Ollama server returned HTTP {resp.status_code}: {resp.text}", status_code=resp.status_code)
                except httpx.TimeoutException as e:
                    logger.error(f"Timeout communicating with Ollama server at {backend_url}: {e}")
                    raise LLMClientError(f"Request to Ollama server timed out after {self.timeout}s", status_code=504)
                except httpx.RequestError as e:
                    logger.error(f"Failed to communicate with Ollama server at {backend_url}: {e}")
                    raise LLMClientError(f"Could not connect to Ollama server at {backend_url}: {str(e)}", status_code=502)

        # 2. Standard llama-cpp / OpenAI format endpoint /v1/chat/completions
        openai_payload = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        if max_tokens:
            openai_payload["max_tokens"] = max_tokens
        if json_mode:
            openai_payload["response_format"] = {"type": "json_object"}

        try:
            resp = await client.post(f"{backend_url}/v1/chat/completions", json=openai_payload, headers=headers, timeout=self._get_timeout())
            if resp.status_code == 200:
                res_json = resp.json()
                if "model" not in res_json or not res_json.get("model"):
                    res_json["model"] = selected_model
                return res_json
            
            error_text = resp.text
            if "image input is not supported" in error_text or "mmproj" in error_text:
                error_msg = (
                    f"Server at {backend_url} is running a model without Vision Projector (--mmproj). "
                    "Please start llama-server with '--mmproj <path_to_mmproj.gguf>' to enable vision capability."
                )
            else:
                error_msg = f"llama.cpp Server error at {backend_url} (HTTP {resp.status_code}): {error_text}"

            logger.error(error_msg)
            raise LLMClientError(error_msg, status_code=resp.status_code)
        except httpx.TimeoutException as e:
            logger.error(f"Timeout communicating with llama.cpp server at {backend_url}: {e}")
            raise LLMClientError(f"Request to llama.cpp server timed out after {self.timeout}s", status_code=504)
        except httpx.RequestError as e:
            logger.error(f"Failed to communicate with llama.cpp server at {backend_url}: {e}")
            raise LLMClientError(f"Could not connect to llama.cpp server at {backend_url}: {str(e)}", status_code=502)

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        backend: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
        json_mode: bool = False
    ) -> AsyncGenerator[str, None]:
        target_backend = backend or self.default_backend
        backend_url = self.get_backend_url(target_backend)
        client = await self.get_client()

        has_images = any(
            isinstance(m.get("content"), list) and any(
                isinstance(part, dict) and part.get("type") == "image_url" for part in m.get("content", [])
            ) for m in messages
        ) or any("images" in m and m["images"] for m in messages)

        selected_model = await self.pick_model(
            backend=target_backend,
            requested_model=model,
            has_images=has_images
        )

        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["x-api-key"] = settings.llm_api_key

        # 1. Ollama streaming via /api/chat NDJSON stream
        if target_backend == "ollama":
            ollama_payload = self._convert_to_ollama_payload(
                messages=messages,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                json_mode=json_mode
            )

            max_retries = 10
            for attempt in range(max_retries):
                try:
                    async with client.stream("POST", f"{backend_url}/api/chat", json=ollama_payload, headers=headers, timeout=self._get_timeout()) as resp:
                        if resp.status_code != 200:
                            error_body = await resp.aread()
                            error_text = error_body.decode()
                            if "loading model" in error_text.lower() and attempt < max_retries - 1:
                                logger.info(f"⏳ Ollama stream is waiting for model load... (attempt {attempt + 1}/{max_retries})")
                                await asyncio.sleep(3.0)
                                continue
                            raise LLMClientError(f"Ollama Stream error ({resp.status_code}): {error_text}", status_code=resp.status_code)

                        in_thinking = False
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                                msg = chunk.get("message", {})
                                content = msg.get("content", "")
                                thinking = msg.get("thinking", "")
                                if thinking:
                                    if not in_thinking:
                                        in_thinking = True
                                        yield "> 🧠 *Thinking...*\n> "
                                    yield thinking.replace("\n", "\n> ")
                                if content:
                                    if in_thinking:
                                        in_thinking = False
                                        yield "\n\n---\n\n"
                                    yield content
                                if chunk.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
                    return
                except httpx.TimeoutException as e:
                    logger.error(f"Ollama streaming timed out: {e}")
                    raise LLMClientError(f"Stream error communicating with Ollama: timeout after {self.timeout}s", status_code=504)
                except httpx.RequestError as e:
                    logger.error(f"Ollama streaming failed: {e}")
                    raise LLMClientError(f"Stream error communicating with Ollama at {backend_url}: {str(e)}", status_code=502)

        # 2. llama.cpp streaming via /v1/chat/completions SSE stream
        openai_payload = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        if max_tokens:
            openai_payload["max_tokens"] = max_tokens

        try:
            async with client.stream("POST", f"{backend_url}/v1/chat/completions", json=openai_payload, headers=headers, timeout=self._get_timeout()) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    raise LLMClientError(f"llama.cpp Stream error ({resp.status_code}): {error_body.decode()}", status_code=resp.status_code)

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
        except httpx.TimeoutException as e:
            logger.error(f"llama.cpp streaming timed out: {e}")
            raise LLMClientError(f"Stream error communicating with llama.cpp: timeout after {self.timeout}s", status_code=504)
        except httpx.RequestError as e:
            logger.error(f"llama.cpp streaming failed: {e}")
            raise LLMClientError(f"Stream error communicating with llama.cpp at {backend_url}: {str(e)}", status_code=502)

llm_client = LLMClient()
