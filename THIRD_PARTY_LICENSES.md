# Third-Party Licenses

AutoMedal depends on the following third-party software. All licenses are
compatible with AutoMedal's distribution model.

## MIT License

### anthropic (Python SDK)
- Source: https://github.com/anthropics/anthropic-sdk-python
- Version: 0.40.0+
- License: MIT © Anthropic, PBC
- Role: Provider client for the bespoke agent kernel — used for direct
  Anthropic API calls and for opencode-go (`base_url=https://opencode.ai/zen/go`).

### openai (Python SDK)
- Source: https://github.com/openai/openai-python
- Version: 1.50.0+
- License: Apache 2.0 © OpenAI
- Role: Provider client for OpenAI direct, Ollama (`/v1`), OpenRouter, Groq,
  and any other OpenAI-shape endpoint.

### rank-bm25
- Source: https://github.com/dorianbrown/rank_bm25
- Version: 0.2.2+
- License: Apache 2.0 © Dorian Brown
- Role: Lexical BM25 ranking for the `recall` cognition tool and for
  motivation-similarity dedupe of queue entries.

### jinja2
- Source: https://github.com/pallets/jinja
- License: BSD-3-Clause © Pallets
- Role: Slot-templated phase prompts in `automedal/agent/prompts/*.md.j2`.

### arxiv (Python client)
- Source: https://github.com/lukasschwab/arxiv.py
- License: MIT © Lukas Schwab
- Role: Researcher-phase paper search tool.

### python-dotenv
- Source: https://github.com/theskumar/python-dotenv
- License: BSD-3-Clause © Saurabh Kumar
- Role: Loads `~/.automedal/.env` at startup for provider credentials.

---

## Apache 2.0

### rich-pixels
- Source: https://github.com/darrenburns/rich-pixels
- License: Apache 2.0 © Darren Burns
- Role: Renders the pixel-art splash logo in the TUI.

---

## Model Providers

Keys are stored at `~/.automedal/.env` (mode 0600), never transmitted outside
the user's machine except to the provider the user authenticated against
(opencode-go/zen, Anthropic, OpenAI, OpenRouter, Groq, or a local Ollama
endpoint).
