// arxiv_search tool — researcher-only.
//
// Pulls the arxiv.org Atom API directly via net/http; no SDK. Returns a
// fixed-format text block the Researcher prompt knows how to consume.
// Skips papers older than max_age_days (default 3 years).
package tools

import (
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const arxivAPI = "http://export.arxiv.org/api/query"

type arxivFeed struct {
	XMLName xml.Name     `xml:"feed"`
	Entries []arxivEntry `xml:"entry"`
}

type arxivEntry struct {
	ID        string `xml:"id"`
	Title     string `xml:"title"`
	Summary   string `xml:"summary"`
	Published string `xml:"published"`
}

func parseArxivID(entryID string) string {
	id := entryID
	if i := strings.LastIndex(id, "/"); i >= 0 {
		id = id[i+1:]
	}
	if i := strings.Index(id, "v"); i >= 0 {
		id = id[:i]
	}
	return id
}

func formatPaper(idx int, e arxivEntry, fullAbstract bool) string {
	pub, err := time.Parse(time.RFC3339, e.Published)
	pubDay := e.Published
	ageDays := -1
	if err == nil {
		pubDay = pub.UTC().Format("2006-01-02")
		ageDays = int(time.Since(pub.UTC()).Hours() / 24)
	}
	abstract := strings.TrimSpace(strings.ReplaceAll(e.Summary, "\n", " "))
	if !fullAbstract && len(abstract) > 300 {
		abstract = abstract[:297] + "..."
	}
	return fmt.Sprintf(
		"=== Paper %d ===\nTitle: %s\nArXiv ID: %s\nDate: %s (%dd ago)\nAbstract: %s\n===",
		idx,
		strings.TrimSpace(e.Title),
		parseArxivID(e.ID),
		pubDay,
		ageDays,
		abstract,
	)
}

func arxivFetch(ctx context.Context, q url.Values) ([]arxivEntry, error) {
	cctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(cctx, http.MethodGet, arxivAPI+"?"+q.Encode(), nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("arxiv: HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var feed arxivFeed
	if err := xml.Unmarshal(body, &feed); err != nil {
		return nil, fmt.Errorf("arxiv: bad XML: %w", err)
	}
	return feed.Entries, nil
}

func runArxivSearch(ctx context.Context, args map[string]any) (ToolResult, error) {
	query := StrArg(args, "query", "")
	ids := StrArg(args, "ids", "")
	if query == "" && ids == "" {
		return Error("error: provide either `query` or `ids`"), nil
	}
	maxResults := IntArg(args, "max_results", 5)
	if maxResults < 1 {
		maxResults = 1
	}
	if maxResults > 10 {
		maxResults = 10
	}
	maxAgeDays := IntArg(args, "max_age_days", 1095)
	if maxAgeDays < 30 {
		maxAgeDays = 30
	}

	q := url.Values{}
	if ids != "" {
		idList := []string{}
		for _, s := range strings.Split(ids, ",") {
			if s = strings.TrimSpace(s); s != "" {
				idList = append(idList, s)
			}
		}
		if len(idList) == 0 {
			return Error("error: empty ids list"), nil
		}
		q.Set("id_list", strings.Join(idList, ","))
		q.Set("max_results", fmt.Sprintf("%d", len(idList)))
		entries, err := arxivFetch(ctx, q)
		if err != nil {
			return Error("error: %s", err.Error()), nil
		}
		if len(entries) == 0 {
			return Result(fmt.Sprintf("(no papers found for IDs %v)", idList)), nil
		}
		parts := make([]string, 0, len(entries))
		for i, e := range entries {
			parts = append(parts, formatPaper(i+1, e, true))
		}
		return Result(strings.Join(parts, "\n\n")), nil
	}

	q.Set("search_query", query)
	q.Set("sortBy", "relevance")
	q.Set("max_results", fmt.Sprintf("%d", maxResults*3))
	entries, err := arxivFetch(ctx, q)
	if err != nil {
		return Error("error: %s", err.Error()), nil
	}
	cutoff := time.Now().UTC().Add(-time.Duration(maxAgeDays) * 24 * time.Hour)
	keep := []arxivEntry{}
	for _, e := range entries {
		pub, err := time.Parse(time.RFC3339, e.Published)
		if err != nil {
			continue
		}
		if pub.UTC().Before(cutoff) {
			continue
		}
		keep = append(keep, e)
		if len(keep) >= maxResults {
			break
		}
	}
	if len(keep) == 0 {
		return Result(fmt.Sprintf("(no recent results for query %q)", query)), nil
	}
	parts := make([]string, 0, len(keep))
	for i, e := range keep {
		parts = append(parts, formatPaper(i+1, e, false))
	}
	return Result(strings.Join(parts, "\n\n")), nil
}

var ArxivSearch = Tool{
	Name: "arxiv_search",
	Description: "Search arxiv by `query` (3-6 keywords) or fetch full abstracts by " +
		"comma-separated `ids`. Filters out papers older than 3 years by " +
		"default. Researcher-only.",
	Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query":        map[string]any{"type": "string"},
			"ids":          map[string]any{"type": "string", "description": "Comma-separated arxiv IDs"},
			"max_results":  map[string]any{"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
			"max_age_days": map[string]any{"type": "integer", "default": 1095, "minimum": 30},
		},
		"required": []string{},
	},
	Run: runArxivSearch,
}
