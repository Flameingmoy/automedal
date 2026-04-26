package scout

import (
	"bytes"

	"gopkg.in/yaml.v3"
)

// encodeYAML emits canonical YAML preserving the insertion order of the
// top-level map keys. yaml.v3 sorts maps; we work around that by
// emitting top-level entries one at a time in the order we want.
func encodeYAML(v map[string]any) (string, error) {
	order := []string{"competition", "task", "dataset", "submission", "objectives", "meta"}
	var buf bytes.Buffer
	enc := yaml.NewEncoder(&buf)
	enc.SetIndent(2)
	for _, k := range order {
		val, ok := v[k]
		if !ok {
			continue
		}
		if err := enc.Encode(map[string]any{k: val}); err != nil {
			return "", err
		}
	}
	for k, val := range v {
		seen := false
		for _, s := range order {
			if s == k {
				seen = true
				break
			}
		}
		if seen {
			continue
		}
		if err := enc.Encode(map[string]any{k: val}); err != nil {
			return "", err
		}
	}
	if err := enc.Close(); err != nil {
		return "", err
	}
	return buf.String(), nil
}
