# AutoMedal — top-level build targets.
#
# Phase 1 of the Go control-plane port. Builds two binaries into bin/:
#   automedal      — the CLI (cmd/automedal)
#   automedal-tui  — the user-facing Charmbracelet shell (internal/ui)
#
# Once Phase 4 lands these will collapse into a single static binary.

GO        ?= go
LDFLAGS   ?= -s -w
GOFLAGS   ?=
BIN       := bin

.PHONY: all build test vet tidy clean

all: build

build:
	$(GO) build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o $(BIN)/automedal     ./cmd/automedal
	$(GO) build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o $(BIN)/automedal-tui ./internal/ui

test:
	$(GO) test ./...

vet:
	$(GO) vet ./...

tidy:
	$(GO) mod tidy

clean:
	rm -rf $(BIN)
