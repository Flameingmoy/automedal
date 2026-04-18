# Third-Party Licenses

AutoMedal bundles and/or depends on the following third-party software. All
licenses are compatible with AutoMedal's distribution model.

## MIT License

### deepagents
- Source: https://github.com/langchain-ai/deepagents
- Version: 0.5.3+
- License: MIT © LangChain, Inc.
- Role: Agent runtime (replaces the pi coding agent in AutoMedal v1.1+).

### langchain / langchain-core / langchain-anthropic / langchain-openai
- Source: https://github.com/langchain-ai/langchain
- License: MIT © LangChain, Inc.
- Role: Chat-model factory + tool protocol for the agent runtime.

### langgraph
- Source: https://github.com/langchain-ai/langgraph
- License: MIT © LangChain, Inc.
- Role: Event streaming + stateful graph engine underneath deepagents.

### python-dotenv
- Source: https://github.com/theskumar/python-dotenv
- License: BSD-3-Clause © Saurabh Kumar
- Role: Loads `~/.automedal/.env` at startup for provider credentials.

### @mariozechner/pi-coding-agent (legacy, removed in Phase E)
- Source: https://github.com/badlogic/pi-mono
- License: MIT © Mario Zechner
- Role: Previous agent runtime. Auto-installed into `automedal/_vendor/` on
  first run for backward compatibility via `AUTOMEDAL_AGENT=pi`. Will be
  deleted after the deepagents path has baked in for one full competition.

---

## Apache 2.0

### rich-pixels
- Source: https://github.com/darrenburns/rich-pixels
- License: Apache 2.0 © Darren Burns
- Role: Renders the pixel-art splash logo in the TUI.

---

## Model Providers

The agent itself does not bundle provider SDKs beyond `langchain-anthropic`
and `langchain-openai`. Keys are stored at `~/.automedal/.env` (mode 0600),
never transmitted outside the user's machine except to the provider the user
authenticated against (Anthropic, OpenAI, OpenRouter, Groq, Mistral, Gemini,
opencode-go/zen, or a local Ollama endpoint).
