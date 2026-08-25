BINARY_NAME=go-infer
VERSION ?= $(shell cat VERSION 2>/dev/null || echo "1.0.0")
LDFLAGS ?= -s -w -X main.Version=$(VERSION)

.PHONY: all build build-prod test clean install release run metallib packages deb rpm

all: metallib build

## Compile Apple Metal Shaders to precompiled metallib
metallib:
	@if command -v xcrun >/dev/null 2>&1; then \
		echo "Compiling Metal shaders to kernels.metallib..."; \
		xcrun -sdk macosx metal -O3 -c pkg/metal/kernels.metal -o pkg/metal/kernels.air 2>/dev/null && \
		xcrun -sdk macosx metallib pkg/metal/kernels.air -o pkg/metal/kernels.metallib 2>/dev/null && \
		rm -f pkg/metal/kernels.air || true; \
	fi

## Build standard local binary
build: metallib
	go build -ldflags="-X main.Version=$(VERSION)" -o $(BINARY_NAME) .

## Build optimized binary with stripped debug symbols
build-prod: metallib
	go build -ldflags="$(LDFLAGS)" -o $(BINARY_NAME) .

## Run all unit and integration tests
test:
	go test -v ./...

## Run Go performance benchmarks
test-bench:
	go test -bench=. ./...

## Build benchmark utility
bench: metallib
	go build -o bench ./cmd/bench

## Build model quantizer utility
quantize:
	go build -o quantize ./cmd/quantize

## Build distribution packages (.deb for Ubuntu/Debian, .rpm for RHEL/CentOS/Fedora)
packages:
	go run ./cmd/package -version $(VERSION)

deb: packages
rpm: packages

## Install binary to $GOPATH/bin
install:
	go install -ldflags="$(LDFLAGS)" .

## Remove compiled binaries
clean:
	rm -rf $(BINARY_NAME) bin/

## Cross-compile binaries for all supported platforms
release: clean
	mkdir -p bin
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -ldflags="$(LDFLAGS)" -o bin/$(BINARY_NAME)-darwin-arm64 .
	CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build -ldflags="$(LDFLAGS)" -o bin/$(BINARY_NAME)-darwin-amd64 .
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="$(LDFLAGS)" -o bin/$(BINARY_NAME)-linux-amd64 .
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -ldflags="$(LDFLAGS)" -o bin/$(BINARY_NAME)-linux-arm64 .
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -ldflags="$(LDFLAGS)" -o bin/$(BINARY_NAME)-windows-amd64.exe .
