// Pure-Go BM25 cognition — `recall(query, k)` over knowledge.md +
// research_notes.md.
//
// Implementation: BM25Okapi with k1=1.5, b=0.75 (same defaults as the
// Python rank_bm25 lib used in the previous Python harness, so dedupe
// thresholds calibrated against rank_bm25 carry over unchanged).
package tools

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
)

const (
	bm25K1 = 1.5
	bm25B  = 0.75
)

var tokenRX = regexp.MustCompile(`[a-z0-9_]+`)

// Tokenize is the BM25 tokenizer (lowercase + alphanumeric+underscore words).
func Tokenize(text string) []string {
	return tokenRX.FindAllString(strings.ToLower(text), -1)
}

// ── BM25Okapi ────────────────────────────────────────────────────────

// BM25 is a precomputed BM25Okapi index over a slice of token streams.
type BM25 struct {
	K1, B   float64
	N       int
	AvgDL   float64
	DocLen  []float64
	IDF     map[string]float64
	TF      []map[string]int
}

// NewBM25 indexes `corpus` (already tokenized).
func NewBM25(corpus [][]string) *BM25 {
	n := len(corpus)
	idx := &BM25{K1: bm25K1, B: bm25B, N: n, IDF: map[string]float64{}}
	idx.DocLen = make([]float64, n)
	idx.TF = make([]map[string]int, n)
	df := map[string]int{}
	totalLen := 0.0

	for i, doc := range corpus {
		idx.DocLen[i] = float64(len(doc))
		totalLen += float64(len(doc))
		tf := map[string]int{}
		for _, tok := range doc {
			tf[tok]++
		}
		idx.TF[i] = tf
		for tok := range tf {
			df[tok]++
		}
	}
	if n > 0 {
		idx.AvgDL = totalLen / float64(n)
	}
	for tok, d := range df {
		// Standard BM25Okapi IDF (rank_bm25 formula).
		idx.IDF[tok] = math.Log((float64(n)-float64(d)+0.5)/(float64(d)+0.5) + 1.0)
	}
	return idx
}

// Score returns BM25 scores for `query` against every doc, in the same
// order as the corpus.
func (b *BM25) Score(query []string) []float64 {
	out := make([]float64, b.N)
	for i := 0; i < b.N; i++ {
		var s float64
		dl := b.DocLen[i]
		for _, tok := range query {
			tf, ok := b.TF[i][tok]
			if !ok {
				continue
			}
			idf := b.IDF[tok]
			denom := float64(tf) + b.K1*(1.0-b.B+b.B*dl/b.AvgDL)
			s += idf * (float64(tf) * (b.K1 + 1.0)) / denom
		}
		out[i] = s
	}
	return out
}

// BM25ScorePairs scores `query` against each candidate text. Re-exported
// helper for the dedupe path (mirrors rank_bm25 → motivation dedupe).
func BM25ScorePairs(query string, candidates []string) []float64 {
	if len(candidates) == 0 {
		return nil
	}
	corpus := make([][]string, len(candidates))
	for i, c := range candidates {
		corpus[i] = Tokenize(c)
	}
	return NewBM25(corpus).Score(Tokenize(query))
}

// ── markdown chunker ─────────────────────────────────────────────────

// chunkByHeading splits a markdown document on `## ` headings and returns
// (heading, body) pairs. Lines before the first `## ` go under `source`.
func chunkByHeading(text, source string) [][2]string {
	var (
		out         [][2]string
		curHead     = source
		curBody     []string
	)
	push := func() {
		body := strings.TrimSpace(strings.Join(curBody, "\n"))
		if body != "" {
			out = append(out, [2]string{curHead, body})
		}
	}
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(line, "## ") {
			push()
			head := strings.TrimSpace(strings.TrimPrefix(line, "## "))
			if head == "" {
				head = source
			}
			curHead = head
			curBody = []string{line}
			continue
		}
		curBody = append(curBody, line)
	}
	push()
	return out
}

func tailResearchNotes(text string, n int) [][2]string {
	all := chunkByHeading(text, "research_notes.md")
	if n <= 0 || len(all) <= n {
		return all
	}
	return all[len(all)-n:]
}

// ── lazy global index ────────────────────────────────────────────────

type bm25Index struct {
	mu      sync.Mutex
	bm25    *BM25
	labels  []string
	bodies  []string
	mtimes  map[string]int64
}

var globalIndex = &bm25Index{}

func (idx *bm25Index) refresh() {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	root := RepoRoot()
	paths := []string{filepath.Join(root, "knowledge.md"), filepath.Join(root, "research_notes.md")}

	seen := map[string]int64{}
	for _, p := range paths {
		if st, err := os.Stat(p); err == nil {
			seen[p] = st.ModTime().UnixNano()
		}
	}
	if !needRebuild(idx.mtimes, seen) && idx.bm25 != nil {
		return
	}

	var labels, bodies []string
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		base := filepath.Base(p)
		var blocks [][2]string
		if base == "research_notes.md" {
			blocks = tailResearchNotes(string(b), 20)
		} else {
			blocks = chunkByHeading(string(b), base)
		}
		for _, hb := range blocks {
			labels = append(labels, fmt.Sprintf("%s — %s", base, hb[0]))
			bodies = append(bodies, hb[1])
		}
	}
	idx.labels = labels
	idx.bodies = bodies
	idx.mtimes = seen
	if len(bodies) > 0 {
		corpus := make([][]string, len(bodies))
		for i, body := range bodies {
			corpus[i] = Tokenize(body)
		}
		idx.bm25 = NewBM25(corpus)
	} else {
		idx.bm25 = nil
	}
}

func needRebuild(prev, cur map[string]int64) bool {
	if len(prev) != len(cur) {
		return true
	}
	for k, v := range cur {
		if pv, ok := prev[k]; !ok || pv < v {
			return true
		}
	}
	return false
}

func (idx *bm25Index) query(q string, k int) []bm25Hit {
	idx.refresh()
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.bm25 == nil || len(idx.bodies) == 0 {
		return nil
	}
	scores := idx.bm25.Score(Tokenize(q))
	hits := make([]bm25Hit, 0, len(scores))
	for i, s := range scores {
		if s <= 0 {
			continue
		}
		hits = append(hits, bm25Hit{score: s, label: idx.labels[i], body: idx.bodies[i]})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].score > hits[j].score })
	if k > 0 && len(hits) > k {
		hits = hits[:k]
	}
	return hits
}

type bm25Hit struct {
	score float64
	label string
	body  string
}

func runRecall(ctx context.Context, args map[string]any) (ToolResult, error) {
	q := StrArg(args, "query", "")
	if q == "" {
		return MissingArg("query"), nil
	}
	k := IntArg(args, "k", 5)
	if k < 1 {
		k = 1
	}
	if k > 10 {
		k = 10
	}
	hits := globalIndex.query(q, k)
	if len(hits) == 0 {
		return Result("(no chunks matched — knowledge base may be empty)"), nil
	}
	parts := make([]string, 0, len(hits))
	for _, h := range hits {
		body := h.body
		if len(body) > 600 {
			body = body[:597] + "..."
		}
		parts = append(parts, fmt.Sprintf("### %s  (score=%.2f)\n%s", h.label, h.score, body))
	}
	return Result(strings.Join(parts, "\n\n")), nil
}

var Recall = Tool{
	Name: "recall",
	Description: "BM25-search the curated knowledge base (knowledge.md + recent " +
		"research_notes.md) for chunks relevant to `query`. Returns up to " +
		"`k` ranked snippets.",
	Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{"type": "string", "description": "Free-text query"},
			"k":     map[string]any{"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
		},
		"required": []string{"query"},
	},
	Run: runRecall,
}

// CognitionTools is the canonical bundle phases use.
var CognitionTools = []Tool{Recall}
