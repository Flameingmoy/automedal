package scout

import "fmt"

// stringf is a tiny alias to keep the scoring source readable.
func stringf(format string, a ...any) string { return fmt.Sprintf(format, a...) }
