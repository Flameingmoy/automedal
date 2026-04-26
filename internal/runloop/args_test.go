package runloop

import (
	"reflect"
	"testing"
)

func TestParseRunArgs(t *testing.T) {
	cases := []struct {
		name     string
		in       []string
		wantArgs []string
		wantEnv  map[string]string
	}{
		{
			name:     "no flags",
			in:       []string{"10", "fast"},
			wantArgs: []string{"10", "fast"},
			wantEnv:  map[string]string{},
		},
		{
			name:     "advisor with model",
			in:       []string{"--advisor", "kimi-k2.6", "10"},
			wantArgs: []string{"10"},
			wantEnv: map[string]string{
				"AUTOMEDAL_ADVISOR":       "1",
				"AUTOMEDAL_ADVISOR_MODEL": "kimi-k2.6",
			},
		},
		{
			name:     "advisor without model — digit follows",
			in:       []string{"--advisor", "10"},
			wantArgs: []string{"10"},
			wantEnv: map[string]string{
				"AUTOMEDAL_ADVISOR":       "1",
				"AUTOMEDAL_ADVISOR_MODEL": DefaultAdvisorModel,
			},
		},
		{
			name:     "advisor at end",
			in:       []string{"5", "--advisor"},
			wantArgs: []string{"5"},
			wantEnv: map[string]string{
				"AUTOMEDAL_ADVISOR":       "1",
				"AUTOMEDAL_ADVISOR_MODEL": DefaultAdvisorModel,
			},
		},
		{
			name:     "advisor before another flag",
			in:       []string{"--advisor", "--other"},
			wantArgs: []string{"--other"},
			wantEnv: map[string]string{
				"AUTOMEDAL_ADVISOR":       "1",
				"AUTOMEDAL_ADVISOR_MODEL": DefaultAdvisorModel,
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, env := ParseRunArgs(c.in)
			if !reflect.DeepEqual(got, c.wantArgs) {
				t.Errorf("args: want %v got %v", c.wantArgs, got)
			}
			if !reflect.DeepEqual(env, c.wantEnv) {
				t.Errorf("env: want %v got %v", c.wantEnv, env)
			}
		})
	}
}
