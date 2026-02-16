# claude-max-api-proxy-rs

**Use your Claude Max subscription with any OpenAI or Anthropic-compatible client — no separate API costs.**

A fast Rust proxy that wraps the [Claude Code CLI](https://github.com/anthropics/claude-code) as a subprocess and exposes both **OpenAI** and **Anthropic** HTTP APIs. Any tool that speaks either protocol can now use your Max subscription directly.

## Why

| Approach | Cost | Limitation |
|----------|------|------------|
| Anthropic API | ~$15/M input, ~$75/M output tokens | Pay per use |
| Claude Max | $200/month flat | OAuth blocked for third-party API use |
| **This proxy** | $0 extra (uses Max subscription) | Routes through CLI |

Anthropic blocks OAuth tokens from third-party API clients, but the Claude Code CLI can use them. This proxy bridges that gap.

## How It Works

```
Your App (OpenAI or Anthropic client)
         │
         ▼
    HTTP Request (either format)
         │
         ▼
   claude-max-api-proxy-rs
         │
         ▼
   Claude Code CLI (subprocess)
         │
         ▼
   Anthropic API (via Max OAuth)
         │
         ▼
   Response → Your format → Your App
```

## Features

- **Dual API support** — OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` on the same server
- **Streaming** — Real-time SSE for both protocols, matching their native event formats
- **Model mapping** — Flexible name resolution (`claude-opus-4`, `claude-sonnet-4-20250514`, `opus`, etc.)
- **Session management** — Conversation continuity via persistent session IDs
- **Zero config** — Uses existing Claude CLI auth, no API keys to manage
- **Fast** — Native Rust binary, ~3MB stripped. Starts instantly.
- **Safe** — No shell execution; all subprocess args passed directly

## Prerequisites

1. **Claude Max subscription** — [Subscribe here](https://claude.ai)
2. **Claude Code CLI** installed and authenticated:
   ```bash
   npm install -g @anthropic-ai/claude-code
   claude auth login
   ```
3. **Rust toolchain** (for building from source):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

## Installation

```bash
git clone https://github.com/thhuang/claude-max-api-proxy-rs.git
cd claude-max-api-proxy-rs
cargo build --release
```

The binary is at `target/release/claude-max-api`.

## Usage

```bash
# Default (port 8080)
claude-max-api

# Custom port
claude-max-api 3456

# Custom port + working directory for CLI subprocesses
claude-max-api 8080 --cwd ~/projects
```

The server binds to `127.0.0.1` (localhost only).

### Quick test

```bash
# Health check
curl http://localhost:8080/health

# List models
curl http://localhost:8080/v1/models

# OpenAI format
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Anthropic format
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming (add -N to disable curl buffering)
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with uptime |
| `/v1/models` | GET | OpenAI-compatible model list |
| `/v1/chat/completions` | POST | OpenAI Chat Completions (streaming & non-streaming) |
| `/v1/messages` | POST | Anthropic Messages (streaming & non-streaming) |

## Models

| Model ID | CLI Alias | Context Window | Max Output |
|----------|-----------|----------------|------------|
| `claude-opus-4` | `opus` | 1,000,000 | 128,000 |
| `claude-sonnet-4` | `sonnet` | 200,000 | 64,000 |
| `claude-haiku-4` | `haiku` | 200,000 | 64,000 |

Date-suffixed variants (e.g. `claude-opus-4-20250514`) and `claude-code-cli/` prefixed names are also accepted.

## Client Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="claude-opus-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Python (Anthropic SDK)

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="not-needed"
)

message = client.messages.create(
    model="claude-opus-4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(message.content[0].text)
```

### Continue.dev

```json
{
  "models": [{
    "title": "Claude (Max)",
    "provider": "openai",
    "model": "claude-opus-4",
    "apiBase": "http://localhost:8080/v1",
    "apiKey": "not-needed"
  }]
}
```

## Architecture

```
src/
├── main.rs           # CLI args, startup checks, graceful shutdown
├── server.rs         # Axum router, CORS, middleware
├── routes.rs         # Endpoint handlers (health, models, completions, messages)
├── subprocess.rs     # Claude CLI process lifecycle and NDJSON parsing
├── session.rs        # Session persistence (~/.claude-code-cli-sessions.json)
├── error.rs          # Unified error types → HTTP responses
├── types/
│   ├── openai.rs     # OpenAI request/response types
│   ├── anthropic.rs  # Anthropic request/response types
│   └── claude_cli.rs # CLI NDJSON message types
└── adapter/
    ├── openai_to_cli.rs    # OpenAI request → CLI invocation
    ├── cli_to_openai.rs    # CLI output → OpenAI response
    ├── anthropic_to_cli.rs # Anthropic request → CLI invocation
    └── cli_to_anthropic.rs # CLI output → Anthropic response
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `claude_max_api=info` | Log level filter ([tracing env-filter syntax](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html)) |

## Acknowledgements

This project is inspired by [claude-max-api-proxy](https://docs.openclaw.ai/providers/claude-max-api-proxy), the Node.js proxy documented by the OpenClaw project. That work demonstrated the core idea — wrapping the Claude Code CLI as a local API server to unlock Max subscription access for third-party clients. This Rust rewrite builds on that foundation with native Anthropic API support and a leaner runtime.

## Comparison with Node.js Version

| | Node.js | Rust |
|---|---------|------|
| **API support** | OpenAI only | OpenAI + Anthropic |
| **Binary size** | ~200MB (with node_modules) | ~3MB |
| **Startup** | ~500ms | <10ms |
| **Dependencies** | Express + many transitive deps | Axum + minimal deps |
| **Build** | `npm install && npm run build` | `cargo build --release` |

## License

MIT
