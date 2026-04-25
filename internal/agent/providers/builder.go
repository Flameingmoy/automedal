package providers

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// Endpoint URLs for the OpenAI-compat providers.
const (
	OpencodeBaseURL   = "https://opencode.ai/zen/go"
	OpenrouterBaseURL = "https://openrouter.ai/api/v1"
	GroqBaseURL       = "https://api.groq.com/openai/v1"
)

// BuildOpts is the optional settings bundle accepted by Build.
type BuildOpts struct {
	Timeout   time.Duration
	MaxTokens int64
}

// Build returns a ChatProvider for `name` bound to `model`. Reads env
// vars for API keys; returns an actionable error if a required key is
// missing.
//
// Port of automedal/agent/providers/__init__.py:build_provider.
func Build(name, model string, opts BuildOpts) (ChatProvider, error) {
	model = normalizeModel(name, model)
	timeout := opts.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	maxTokens := opts.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	switch name {
	case "opencode-go":
		key := os.Getenv("OPENCODE_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("OPENCODE_API_KEY not set (run `automedal setup`)")
		}
		return &AnthropicProvider{
			ModelName: model, APIKey: key, BaseURL: OpencodeBaseURL,
			MaxTokens: maxTokens, Timeout: timeout,
		}, nil

	case "anthropic":
		key := os.Getenv("ANTHROPIC_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY not set")
		}
		return &AnthropicProvider{
			ModelName: model, APIKey: key,
			MaxTokens: maxTokens, Timeout: timeout,
		}, nil

	case "openai":
		key := os.Getenv("OPENAI_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("OPENAI_API_KEY not set")
		}
		return &OpenAIProvider{ModelName: model, APIKey: key, Timeout: timeout}, nil

	case "ollama":
		base := strings.TrimRight(firstNonEmpty(os.Getenv("OLLAMA_BASE_URL"),
			os.Getenv("OLLAMA_HOST"), "http://localhost:11434"), "/")
		if !strings.HasSuffix(base, "/v1") {
			base += "/v1"
		}
		return &OpenAIProvider{ModelName: model, APIKey: "ollama", BaseURL: base, Timeout: timeout}, nil

	case "openrouter":
		key := os.Getenv("OPENROUTER_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("OPENROUTER_API_KEY not set")
		}
		return &OpenAIProvider{ModelName: model, APIKey: key, BaseURL: OpenrouterBaseURL, Timeout: timeout}, nil

	case "groq":
		key := os.Getenv("GROQ_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("GROQ_API_KEY not set")
		}
		return &OpenAIProvider{ModelName: model, APIKey: key, BaseURL: GroqBaseURL, Timeout: timeout}, nil
	}
	return nil, fmt.Errorf("unknown provider: %s", name)
}

// ParseSlug splits "provider/model-id" — Python parity helper.
func ParseSlug(slug string) (provider, model string, err error) {
	if !strings.Contains(slug, "/") {
		return "", "", fmt.Errorf("expected 'provider/model', got %q", slug)
	}
	parts := strings.SplitN(slug, "/", 2)
	return parts[0], parts[1], nil
}

func normalizeModel(provider, model string) string {
	if i := strings.IndexByte(model, '/'); i > 0 && model[:i] == provider {
		return model[i+1:]
	}
	return model
}

func firstNonEmpty(xs ...string) string {
	for _, x := range xs {
		if x != "" {
			return x
		}
	}
	return ""
}
