package components

import "strconv"

func strconvF4(v float64) string {
	return strconv.FormatFloat(v, 'f', 4, 64)
}
