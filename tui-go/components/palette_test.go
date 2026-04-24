package components

import "testing"

func TestIsQuit(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want bool
	}{
		{"q", true},
		{"Q", true},
		{"quit", true},
		{"exit", true},
		{":q", true},
		{":quit", true},
		{":wq", true},
		{"q extra-args-ignored", true},
		{"run 30", false},
		{"", false},
		{"  ", false},
		{"qq", false},
		{"query", false},
	} {
		if got := IsQuit(tc.in); got != tc.want {
			t.Errorf("IsQuit(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestNormalize(t *testing.T) {
	for _, tc := range []struct {
		in, want string
	}{
		{"", ""},
		{"  ", ""},
		{"run", "run"},
		{"run30", "run 30"},
		{"run30 --advisor kimi", "run 30 --advisor kimi"},
		{"doctor", "doctor"},
		{"xyz42", "xyz42"}, // not a known command → unchanged
	} {
		if got := Normalize(tc.in); got != tc.want {
			t.Errorf("Normalize(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestSplitForAdvisor(t *testing.T) {
	isAdv, prefix, _ := SplitForAdvisor("run 10 --advisor k")
	if !isAdv || prefix != "k" {
		t.Fatalf("want isAdv=true prefix=k, got %v %q", isAdv, prefix)
	}
	isAdv, prefix, _ = SplitForAdvisor("run 10 --advisor ")
	// Trailing space → parts[-1] is "", so advIdx is not last unless no space at all.
	// Our Python mirror also treats this as "right after --advisor".
	if !isAdv {
		t.Fatalf("want isAdv=true on bare --advisor, got false")
	}
	_ = prefix

	isAdv, _, _ = SplitForAdvisor("run 10")
	if isAdv {
		t.Fatal("want isAdv=false on run 10")
	}
}

func TestAutocompleteSingleMatchAppendsSpace(t *testing.T) {
	out, fired := Autocomplete("doc")
	if !fired {
		t.Fatal("expected autocompletion to fire for unique prefix")
	}
	if out != "doctor " {
		t.Errorf("want 'doctor ', got %q", out)
	}
}

func TestAutocompleteAmbiguousStaysPut(t *testing.T) {
	// 'r' matches 'run' and 'render' → no single-completion.
	out, _ := Autocomplete("r")
	if out == "run " || out == "render " {
		t.Errorf("ambiguous completion for %q → %q", "r", out)
	}
}

func TestAutocompleteNoMatchIsNoop(t *testing.T) {
	out, fired := Autocomplete("zzz")
	if fired {
		t.Errorf("unexpected fired on non-match")
	}
	if out != "zzz" {
		t.Errorf("want zzz, got %q", out)
	}
}
