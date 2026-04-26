// Adapter that lets the cmd / phase layer pass agent.EventSink to
// advisor.Consult without creating an import cycle here. The agent
// package would normally import advisor (ConsultFunc, etc.) so we
// can't import agent in advisor; instead we expose a SinkFunc adapter.
package advisor

// SinkFunc is a thin function-typed adapter that satisfies AdvisorEvents.
// Wire it as `advisor.SinkFunc(func(...) { sink.AdvisorConsult(agent.AdvisorConsultArgs{...}) })`.
type SinkFunc func(purpose, model, reason, preview string, inTokens, outTokens int, skipped bool)

// RecordAdvisorConsult satisfies AdvisorEvents.
func (f SinkFunc) RecordAdvisorConsult(purpose, model, reason, preview string, inTokens, outTokens int, skipped bool) {
	if f == nil {
		return
	}
	f(purpose, model, reason, preview, inTokens, outTokens, skipped)
}
