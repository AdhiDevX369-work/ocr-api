  # LLM Server Dual Gateway

  A standalone, high-performance proxy and unified API gateway for **Ollama** (`:11434`) and **llama.cpp / llama-server** (`:8080`). It supports dynamic runtime backend routing (via HTTP headers, JSON body fields, or model name prefixes), automatic format translation between Ollama and OpenAI schemas, streaming responses, API key authentication, in-memory LRU caching, backend failover, and aggregated health & model listing.

  This gateway acts as the centralized LLM access point for the **Rev-RAG** architecture, enabling FastAPI services (`app/core/ollama_client.py`), Streamlit dashboards, and external scripts to interact with multiple local LLM engines using a single set of API endpoints.

  ---

  ## Key Features

  - **Dual-Backend Support**: Unified proxying for both Ollama (`/api/...` and `/v1/...`) and llama.cpp (`/v1/...`).
  - **Multimodal Support (Text & Image)**: Full support for image inputs alongside text prompts. Accepts Base64 strings (`images: ["<base64>"]`) or OpenAI image parts (`image_url`) with automatic bidirectional schema conversion.
  - **Dynamic Backend Routing**: Override target backend on per-request level using HTTP headers, JSON body parameter, or model name prefixes.
  - **Bi-directional Format Translation**: Automatically translates Ollama requests to OpenAI Chat Completion format for llama.cpp, and vice versa.
  - **LRU Response Caching**: In-memory caching for deterministic responses (`temperature=0`) with configurable max capacity and TTL.
  - **Automatic Backend Failover**: Optional fallback to secondary backend if the primary backend is unreachable or returns an error.
  - **Aggregated Model Listing**: Combined endpoints (`/v1/models` and `/api/tags`) listing models across all active backends with metadata tagging.
  - **API Key Security**: Optional key-based HTTP authorization middleware (`X-API-Key`).
  - **CORS & Proxy Ready**: Built-in CORS middleware and support for ngrok public tunneling or local network bridging.

  ---

  ## Environment Variables

  Configure the gateway using environment variables or by adding them to the root `.env` file (`../.env` relative to `llm_server.py`).

  | Environment Variable | Default | Type | Description |
  |---|---|---|---|
  | `LLM_SERVER_KEYS` or `API_KEYS` | _(empty)_ | String | Comma-separated API keys. If empty, authentication is disabled. |
  | `DEFAULT_BACKEND` | `ollama` | String | Target backend when unspecified (`ollama` or `llama_cpp`). |
  | `OLLAMA_BASE_URL` | `http://localhost:11434` | String | Base URL of the Ollama server. |
  | `LLAMA_CPP_BASE_URL` | `http://localhost:8080` | String | Base URL of the llama.cpp server (`llama-server`). |
  | `ENABLE_BACKEND_FAILOVER` | `false` | Boolean | If `true` (`1`/`yes`), automatically retries failed Ollama requests on llama.cpp. |
  | `OLLAMA_MODEL` | `qwen2.5:7b` | String | Default model used when not specified in request body. |
  | `OLLAMA_KEEP_ALIVE` | `-1` | Integer | Model keep-alive duration in seconds (`-1` = keep loaded indefinitely). |
  | `OLLAMA_NUM_CTX` | `32768` | Integer | Context window size injected into Ollama request options. |
  | `OLLAMA_TIMEOUT` | `300` | Integer | Client request timeout in seconds. |
  | `LLM_CACHE_MAX` | `500` | Integer | Maximum number of items in LRU response cache. |
  | `LLM_CACHE_TTL` | `900` | Integer | Time-To-Live (TTL) for cached responses in seconds (15 minutes). |
  | `LLM_SERVER_LOG_LEVEL` | `INFO` | String | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
  | `LLM_SERVER_ORIGINS` | `*` | String | Comma-separated allowed CORS origins. |

  ---

  ## Dynamic Backend Routing

  The gateway resolves which backend to use (`ollama` or `llama_cpp`) for each incoming request in the following priority order:

  1. **HTTP Header Override**:
    - Header: `X-LLM-Backend: llama_cpp` (accepts `llama_cpp`, `llama-cpp`, `llama.cpp`, `llamacpp`)
    - Header: `X-LLM-Backend: ollama`
  2. **JSON Body Parameter**:
    - `"backend": "llama_cpp"` or `"backend": "ollama"`
  3. **Model Name Prefix**:
    - Model name starting with `llama_cpp/`, `llama-cpp/`, or `llamacpp/` (e.g. `"model": "llama_cpp/qwen3.5-4b"`). The prefix is stripped before sending to the backend.
    - Model name starting with `ollama/` (e.g. `"model": "ollama/qwen2.5:7b"`). The prefix is stripped before sending to the backend.
  4. **Default Fallback**:
    - Uses `DEFAULT_BACKEND` environment variable (defaults to `ollama`).

  ---

## Multimodal Processing Mechanics (Text & Image)

The gateway supports vision-capable models (such as `qwen2-vl`, `llava`, `llama3-vision`, etc.) for multimodal inference combining text prompts with image inputs.

### Supported Input Schemas

1. **Ollama Native Format (`images: ["<base64>"]`)**:
   Pass base64-encoded strings directly in the `images` array of a message object (`/api/chat`) or at the root level (`/api/generate`).
   ```json
   {
     "model": "qwen2-vl",
     "messages": [
       {
         "role": "user",
         "content": "What is in this image?",
         "images": ["/9j/4AAQSkZJRg..."]
       }
     ]
   }
   ```

2. **OpenAI Vision Format (`image_url`)**:
   Pass content as an array of typed parts under `/v1/chat/completions` or `/api/chat`.
   ```json
   {
     "model": "qwen2-vl",
     "messages": [
       {
         "role": "user",
         "content": [
           {"type": "text", "text": "What is in this image?"},
           {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."}}
         ]
       }
     ]
   }
   ```

### Automatic Cross-Backend Translation

- When an Ollama multimodal request is routed to `llama_cpp`, base64 image strings in `images` are automatically converted into OpenAI `image_url` content blocks (`data:image/jpeg;base64,...`).
- When an OpenAI multimodal request is routed to `ollama`, content arrays containing `image_url` objects are automatically parsed and converted into native Ollama text strings and `images` arrays.

  ---

  ## Authentication

  When `LLM_SERVER_KEYS` (or `API_KEYS`) is configured with valid tokens, all requests except public system endpoints (`/` and `/health`) and CORS `OPTIONS` preflight requests must include the `X-API-Key` header.

  ```http
  X-API-Key: sk_your_configured_key_here
  ```

  If no keys are defined in environment variables, authentication is disabled and all endpoints accept unauthenticated calls.

  - **Success**: Request processed normally.
  - **Failure (Invalid/Missing key)**: Returns `401 Unauthorized`:
    ```json
    {"error": "Invalid or missing API key"}
    ```

  ---

  ## Detailed API Endpoints & Parameter Reference

  ### 1. Server Metadata

  #### `GET /`
  Returns basic service metadata and backend configurations.

  - **Authentication**: None (Public)
  - **Query Parameters**: None
  - **Request Body**: None

  **Response `200 OK`**:
  ```json
  {
    "service": "LLM Server Dual Gateway",
    "default_backend": "ollama",
    "ollama_url": "http://localhost:11434",
    "llama_cpp_url": "http://localhost:8080"
  }
  ```

  ---

  ### 2. Unified Health Check

  #### `GET /health`
  Performs connection checks against both Ollama and llama.cpp backends and lists available models on each.

  - **Authentication**: None (Public)
  - **Query Parameters**: None
  - **Request Body**: None

  **Response `200 OK` (At least one backend connected)**:
  ```json
  {
    "status": "healthy",
    "default_backend": "ollama",
    "backends": {
      "ollama": {
        "status": "connected",
        "url": "http://localhost:11434",
        "models": ["qwen2.5:7b", "llama3.1:latest"]
      },
      "llama_cpp": {
        "status": "connected",
        "url": "http://localhost:8080",
        "models": ["llama-cpp-model"]
      }
    }
  }
  ```

  **Status Values**:
  - `"healthy"`: All configured backends are connected.
  - `"degraded"`: One backend is connected, but another backend is offline.
  - `"unhealthy"`: Both backends are disconnected (HTTP Status `503 Service Unavailable`).

  **Response `503 Service Unavailable` (All backends offline)**:
  ```json
  {
    "status": "unhealthy",
    "default_backend": "ollama",
    "backends": {
      "ollama": {
        "status": "disconnected",
        "url": "http://localhost:11434",
        "error": "All connection attempts failed"
      },
      "llama_cpp": {
        "status": "disconnected",
        "url": "http://localhost:8080",
        "error": "Connection refused"
      }
    }
  }
  ```

  ---

  ### 3. Ollama-Compatible Chat Endpoint

  #### `POST /api/chat`
  Performs multi-turn conversational chat completion. Compatible with native Ollama API schema. If targeted at `llama_cpp`, payload is automatically converted to OpenAI format and converted back to Ollama structure.

  - **Authentication**: Requires `X-API-Key` header (if keys set)
  - **Headers**:
    - `Content-Type: application/json`
    - `X-API-Key: <key>` (optional depending on server configuration)
    - `X-LLM-Backend: ollama | llama_cpp` (optional override)

  **Request Body Parameters**:

  | Parameter | Type | Required | Default | Description |
  |---|---|---|---|---|
  | `messages` | Array of Objects | **Yes** | — | Array of message objects: `[{"role": "user"\|"assistant"\|"system", "content": "string", "images": ["base64_1"]}]`. |
  | `messages[].images` | Array of Strings | No | — | Optional base64-encoded image strings or image URLs for multimodal vision models. |
  | `model` | String | No | `OLLAMA_MODEL` env | Model name. Supports prefix (`ollama/name` or `llama_cpp/name`). |
  | `backend` | String | No | — | Backend selector (`ollama` or `llama_cpp`). Overrides default. |
  | `stream` | Boolean | No | `false` | If `true`, streams NDJSON chunks (`application/x-ndjson`). |
  | `keep_alive` | Integer / String | No | `OLLAMA_KEEP_ALIVE` env | Time to keep model loaded in memory (Ollama only). |
  | `options` | Object | No | `{}` | Generation hyperparameter settings. |
  | `options.temperature` | Float | No | `0.0` | Sampling temperature. `0` enables LRU response caching. |
  | `options.top_p` | Float | No | — | Nucleus sampling parameter. |
  | `options.num_ctx` | Integer | No | `OLLAMA_NUM_CTX` env | Context window size (e.g. `32768`). |
  | `options.num_predict` | Integer | No | — | Maximum number of tokens to generate. |

  **Example Request Body (Text Only)**:
  ```json
  {
    "model": "qwen2.5:7b",
    "messages": [
      {"role": "system", "content": "You are an expert software architect."},
      {"role": "user", "content": "What is the difference between REST and gRPC?"}
    ],
    "stream": false,
    "options": {
      "temperature": 0.0,
      "num_ctx": 16384,
      "top_p": 0.9
    }
  }
  ```

  **Example Request Body (Multimodal Text + Image)**:
  ```json
  {
    "model": "qwen2-vl",
    "messages": [
      {
        "role": "user",
        "content": "Describe what is shown in this image.",
        "images": [
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ]
      }
    ]
  }
  ```

  **Response `200 OK` (Non-Streaming)**:
  ```json
  {
    "model": "qwen2.5:7b",
    "created_at": "2026-08-10T09:15:00.000000+00:00",
    "message": {
      "role": "assistant",
      "content": "REST uses standard HTTP methods (GET, POST, etc.) with JSON or XML payloads..."
    },
    "done": true,
    "done_reason": "stop",
    "total_duration": 452000000,
    "cached": true
  }
  ```

  **Response (Streaming `stream=true`)**:
  Content type: `application/x-ndjson`. Returns line-delimited JSON chunks:
  ```json
  {"model":"qwen2.5:7b","created_at":"...","message":{"role":"assistant","content":"REST"},"done":false}
  {"model":"qwen2.5:7b","created_at":"...","message":{"role":"assistant","content":" uses"},"done":false}
  {"model":"qwen2.5:7b","created_at":"...","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}
  ```

  ---

  ### 4. Ollama-Compatible Text Generation Endpoint

  #### `POST /api/generate`
  Single-prompt text completion (without explicit chat turn arrays). Proxies to Ollama or translates to OpenAI completion via system/user message layout for llama.cpp.

  - **Authentication**: Requires `X-API-Key` header (if keys set)
  - **Headers**: `X-API-Key`, `X-LLM-Backend`

  **Request Body Parameters**:

  | Parameter | Type | Required | Default | Description |
  |---|---|---|---|---|
  | `prompt` | String | **Yes** | — | Input prompt string. |
  | `system` | String | No | `""` | Optional system instruction prompt. |
  | `model` | String | No | `OLLAMA_MODEL` env | Model name (supports `ollama/` or `llama_cpp/` prefix). |
  | `backend` | String | No | — | Backend selector (`ollama` or `llama_cpp`). |
  | `stream` | Boolean | No | `false` | If `true`, returns streamed NDJSON. |
  | `keep_alive` | Integer / String | No | `OLLAMA_KEEP_ALIVE` env | Model memory hold time. |
  | `options` | Object | No | `{}` | Generation parameters (`temperature`, `top_p`, `num_ctx`, `num_predict`). |

  **Example Request Body**:
  ```json
  {
    "prompt": "Summarize the benefits of vector indexing in databases.",
    "system": "Be concise and use bullet points.",
    "model": "llama_cpp/qwen3.5-4b",
    "stream": false,
    "options": {
      "temperature": 0.0
    }
  }
  ```

  **Response `200 OK`**:
  ```json
  {
    "model": "qwen3.5-4b",
    "created_at": "2026-08-10T09:15:00.000000+00:00",
    "response": "- Sub-linear search time for nearest neighbor queries\n- Scalable retrieval for RAG pipelines...",
    "done": true,
    "done_reason": "stop",
    "total_duration": 310000000
  }
  ```

  ---

  ### 5. OpenAI-Compatible Chat Completions Endpoint

  #### `POST /v1/chat/completions`
  Standard OpenAI format endpoint. Can target Ollama or llama.cpp backend directly.

  - **Authentication**: Requires `X-API-Key` header (if keys set)
  - **Headers**: `X-API-Key`, `X-LLM-Backend`

  **Request Body Parameters**:

  | Parameter | Type | Required | Default | Description |
  |---|---|---|---|---|
  | `messages` | Array of Objects | **Yes** | — | Standard OpenAI message list: `[{"role": "system"\|"user"\|"assistant", "content": "..."}]`. |
  | `model` | String | No | `OLLAMA_MODEL` env | Model name or prefixed model (`llama_cpp/...` or `ollama/...`). |
  | `backend` | String | No | — | Backend selector (`ollama` or `llama_cpp`). |
  | `temperature` | Float | No | `0.0` | Sampling temperature (`0.0` triggers LRU caching). |
  | `top_p` | Float | No | — | Nucleus sampling probability. |
  | `max_tokens` | Integer | No | — | Maximum tokens in completion response. |
  | `stream` | Boolean | No | `false` | If `true`, returns Server-Sent Events (`text/event-stream`). |

  **Example Request Body**:
  ```json
  {
    "model": "qwen2.5:7b",
    "messages": [
      {"role": "user", "content": "Write a Python function to compute Fibonacci numbers."}
    ],
    "temperature": 0.0,
    "max_tokens": 512,
    "stream": false
  }
  ```

  **Response `200 OK`**:
  ```json
  {
    "id": "chatcmpl-12345",
    "object": "chat.completion",
    "created": 1770628500,
    "model": "qwen2.5:7b",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "def fibonacci(n: int) -> int:\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 18,
      "completion_tokens": 58,
      "total_tokens": 76
    }
  }
  ```

  ---

  ### 6. OpenAI-Compatible Models Endpoint

  #### `GET /v1/models`
  Fetches available models from both Ollama and llama.cpp backends, merging them into an OpenAI model list format with an added `backend` attribute.

  - **Authentication**: Requires `X-API-Key` header (if keys set)
  - **Query Parameters**: None

  **Response `200 OK`**:
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "qwen2.5:7b",
        "object": "model",
        "created": 1770628000,
        "owned_by": "library",
        "backend": "ollama"
      },
      {
        "id": "llama-cpp-model",
        "object": "model",
        "created": 1770628000,
        "owned_by": "llama-cpp",
        "backend": "llama_cpp"
      }
    ]
  }
  ```

  ---

  ### 7. Ollama-Compatible Tags Endpoint

  #### `GET /api/tags`
  Aggregates models from both Ollama (`/api/tags`) and llama.cpp (`/v1/models`) into Ollama tag format.

  - **Authentication**: Requires `X-API-Key` header (if keys set)
  - **Query Parameters**: None

  **Response `200 OK`**:
  ```json
  {
    "models": [
      {
        "name": "qwen2.5:7b",
        "model": "qwen2.5:7b",
        "size": 4661224676,
        "backend": "ollama",
        "details": {
          "format": "gguf",
          "family": "qwen2"
        }
      },
      {
        "name": "llama-cpp-model",
        "model": "llama-cpp-model",
        "backend": "llama_cpp",
        "details": {
          "format": "gguf",
          "family": "llama_cpp"
        }
      }
    ]
  }
  ```

  ---

  ### 8. Cache Management Endpoints

  #### `POST /cache/flush`
  Clears all entries stored in the in-memory response cache.

  - **Authentication**: Requires `X-API-Key` header (if keys set)
  - **Request Body**: None

  **Response `200 OK`**:
  ```json
  {
    "cleared": 14
  }
  ```

  ---

  #### `GET /cache/stats`
  Retrieves cache usage statistics.

  - **Authentication**: Requires `X-API-Key` header (if keys set)
  - **Query Parameters**: None

  **Response `200 OK`**:
  ```json
  {
    "size": 14,
    "max": 500,
    "ttl": 900
  }
  ```

  ---

  ## Error Handling & Status Codes

  | HTTP Status Code | Meaning | Cause / Description |
  |---|---|---|
  | `200 OK` | Success | Request succeeded. |
  | `400 Bad Request` | Invalid Request Body | Body is not valid JSON or not a JSON object. |
  | `401 Unauthorized` | Auth Failed | `X-API-Key` header missing or invalid when auth enabled. |
  | `502 Bad Gateway` | Backend Error | Target backend (Ollama or llama.cpp) returned HTTP error or is unreachable. |
  | `503 Service Unavailable` | All Backends Down | Returned by `/health` when neither Ollama nor llama.cpp is reachable. |

  **Standard Error Response Body**:
  ```json
  {
    "error": "Invalid or missing API key"
  }
  ```

  ---

  ## Memory LRU Cache Mechanism

  To accelerate repeated queries and reduce redundant GPU load, the gateway integrates an in-memory LRU cache (`LLMCache`).

  - **Activation**: Triggers automatically when `temperature` is explicitly set to `0` (or `0.0`).
  - **Cache Key**: MD5 hash of `{"backend": <resolved_backend>, "data": <request_body_data>}`.
  - **Eviction Strategy**: Least Recently Used (LRU) when cached item count reaches `LLM_CACHE_MAX` (default: `500`).
  - **Expiration**: Items older than `LLM_CACHE_TTL` (default: `900` seconds / 15 mins) are invalidated automatically.
  - **Cache Response Flag**: Responses served from cache contain `"cached": true` in non-streaming responses.

  ---

  ## Python Integration Examples

  ### Synchronous Python Client (`httpx`)

  ```python
  import httpx

  class LLMServerClient:
      def __init__(self, base_url: str = "http://localhost:8100", api_key: str = ""):
          self.base_url = base_url.rstrip("/")
          self.headers = {"Content-Type": "application/json"}
          if api_key:
              self.headers["X-API-Key"] = api_key

      def health(self) -> dict:
          return httpx.get(f"{self.base_url}/health", timeout=10).json()

      def chat(
          self,
          messages: list[dict],
          model: str = "",
          backend: str = "",
          temperature: float = 0.0,
      ) -> str:
          headers = dict(self.headers)
          if backend:
              headers["X-LLM-Backend"] = backend

          payload = {"messages": messages, "stream": False}
          if model:
              payload["model"] = model
          if temperature is not None:
              payload["options"] = {"temperature": temperature}

          resp = httpx.post(f"{self.base_url}/api/chat", json=payload, headers=headers, timeout=120)
          resp.raise_for_status()
          return resp.json()["message"]["content"]

      def generate(self, prompt: str, model: str = "", system: str = "") -> str:
          payload = {"prompt": prompt, "stream": False}
          if model:
              payload["model"] = model
          if system:
              payload["system"] = system

          resp = httpx.post(f"{self.base_url}/api/generate", json=payload, headers=self.headers, timeout=120)
          resp.raise_for_status()
          return resp.json()["response"]

      def analyze_image(self, image_path: str, prompt: str, model: str = "qwen2-vl") -> str:
          import base64
          with open(image_path, "rb") as f:
              img_b64 = base64.b64encode(f.read()).decode("utf-8")
          payload = {
              "model": model,
              "messages": [
                  {"role": "user", "content": prompt, "images": [img_b64]}
              ],
              "stream": False
          }
          resp = httpx.post(f"{self.base_url}/api/chat", json=payload, headers=self.headers, timeout=120)
          resp.raise_for_status()
          return resp.json()["message"]["content"]

      def openai_chat(self, messages: list[dict], model: str = "") -> dict:
          payload = {"messages": messages, "stream": False}
          if model:
              payload["model"] = model
          resp = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=self.headers, timeout=120)
          resp.raise_for_status()
          return resp.json()


  # --- Usage Example ---
  if __name__ == "__main__":
      client = LLMServerClient(base_url="http://localhost:8100", api_key="YOUR_KEY")

      # 1. Health check
      print("Health:", client.health())

      # 2. Chat using default backend (Ollama)
      reply = client.chat([{"role": "user", "content": "Explain async Python."}])
      print("Ollama Chat:", reply)

      # 3. Chat overriding backend to llama.cpp via header
      reply_llama = client.chat(
          messages=[{"role": "user", "content": "Explain async Python."}],
          backend="llama_cpp"
      )
      print("llama.cpp Chat:", reply_llama)

      # 4. Single prompt generation
      gen_text = client.generate(prompt="List 3 benefits of RAG.")
      print("Generation:", gen_text)

      # 5. Multimodal Image Analysis
      # vision_reply = client.analyze_image("diagram.png", "Explain this diagram.")
      # print("Vision Analysis:", vision_reply)
  ```

  ---

  ### Asynchronous Python Client (`httpx.AsyncClient`)

  ```python
  import asyncio
  import httpx

  async def main():
      base_url = "http://localhost:8100"
      headers = {"X-API-Key": "YOUR_KEY", "Content-Type": "application/json"}

      async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=60.0) as client:
          # Check available models across both backends
          res = await client.get("/v1/models")
          print("Models:", res.json())

          # Call Chat endpoint
          chat_req = {
              "model": "qwen2.5:7b",
              "messages": [{"role": "user", "content": "Hello, gateway!"}],
              "options": {"temperature": 0.0}
          }
          res = await client.post("/api/chat", json=chat_req)
          print("Chat Response:", res.json()["message"]["content"])

  asyncio.run(main())
  ```

  ---

  ## cURL & Command-Line Usage

  ### Linux / macOS (cURL)

  ```bash
  # 1. Health Check
  curl http://localhost:8100/health

  # 2. Chat Completion (Ollama backend)
  curl -X POST http://localhost:8100/api/chat \
    -H "X-API-Key: YOUR_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [{"role": "user", "content": "Hello!"}],
      "options": {"temperature": 0}
    }'

  # 3. Multimodal (Text + Image) Chat Completion
  IMAGE_B64=$(base64 -w 0 sample.jpg)
  curl -X POST http://localhost:8100/api/chat \
    -H "X-API-Key: YOUR_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen2-vl",
      "messages": [
        {
          "role": "user",
          "content": "What is in this picture?",
          "images": ["'"$IMAGE_B64"'"]
        }
      ]
    }'

  # 4. Chat Completion targeting llama.cpp via Header
  curl -X POST http://localhost:8100/api/chat \
    -H "X-API-Key: YOUR_KEY" \
    -H "X-LLM-Backend: llama_cpp" \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [{"role": "user", "content": "Hello llama.cpp!"}]
    }'

  # 5. OpenAI Chat Completions Endpoint
  curl -X POST http://localhost:8100/v1/chat/completions \
    -H "X-API-Key: YOUR_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen2.5:7b",
      "messages": [{"role": "user", "content": "Write a hello world program in Rust."}]
    }'

  # 6. List all models
  curl -H "X-API-Key: YOUR_KEY" http://localhost:8100/api/tags

  # 7. Flush Cache
  curl -X POST -H "X-API-Key: YOUR_KEY" http://localhost:8100/cache/flush
  ```

  ---

  ### Windows (PowerShell)

  > **Note**: Standard `curl` in PowerShell is an alias to `Invoke-WebRequest`. Use `curl.exe` or `Invoke-RestMethod`.

  ```powershell
  # Using native PowerShell Invoke-RestMethod
  $headers = @{
      "X-API-Key" = "YOUR_KEY"
      "Content-Type" = "application/json"
  }

  # Health Check
  Invoke-RestMethod -Uri "http://localhost:8100/health" -Method Get

  # Chat Request
  $body = @{
      model = "qwen2.5:7b"
      messages = @(
          @{ role = "user"; content = "Summarize database index types." }
      )
      options = @{ temperature = 0.0 }
  } | ConvertTo-Json -Depth 5

  $response = Invoke-RestMethod -Uri "http://localhost:8100/api/chat" -Method Post -Headers $headers -Body $body
  $response.message.content

  # Using curl.exe
  curl.exe -X POST "http://localhost:8100/api/chat" `
    -H "X-API-Key: YOUR_KEY" `
    -H "Content-Type: application/json" `
    -d "{\`"messages\`":[{\`"role\`":\`"user\`",\`"content\`":\`"hello\`"}]}"
  ```

  ---

  ## Server Execution & Deployment Options

  ### Running Standalone Script

  ```bash
  # Run on default host (0.0.0.0) and port (8100)
  python llm-server/llm_server.py

  # Custom host, port, and auto-reload for development
  python llm-server/llm_server.py --host 0.0.0.0 --port 8100 --reload
  ```

  ---

  ### Public Tunneling via ngrok

  Expose the LLM Gateway securely over HTTPS using ngrok:

  ```bash
  ngrok http 8100
  # OR using pre-configured ngrok.yml:
  ngrok start --config ngrok.yml llm-server
  ```

  Public endpoint format:
  `https://<your-subdomain>.ngrok-free.dev`

  ---

  ### WSL Port Forwarding (LAN Access)

  When running inside Windows Subsystem for Linux (WSL), allow external devices on your Local Area Network (LAN) to reach port 8100:

  ```powershell
  # Run in PowerShell as Administrator:
  netsh interface portproxy add v4tov4 listenport=8100 listenaddress=0.0.0.0 connectport=8100 connectaddress=<WSL_IP>

  # Add Windows Firewall rule:
  netsh advfirewall firewall add rule name="LLM Server Gateway 8100" dir=in action=allow protocol=TCP localport=8100
  ```

  - Find WSL IP: `hostname -I` (inside WSL terminal)
  - Find LAN IP: `ipconfig` (inside Windows PowerShell)
