# ==============================================================================
# Build Stage: Multi-Arch Pure Go Builder
# ==============================================================================
FROM golang:1.24-alpine AS builder

WORKDIR /src

# Install build tools & CA certificates
RUN apk add --no-cache git make ca-certificates tzdata

# Copy dependency manifests and source files
COPY go.mod ./
COPY . .

# Build stripped, zero-dependency static executable
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w -X main.Version=$(cat VERSION 2>/dev/null || echo '1.0.0')" \
    -trimpath \
    -o /bin/goinfer .

# Build CLI utility binaries
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -trimpath -o /bin/quantize ./cmd/quantize
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -trimpath -o /bin/bench ./cmd/bench

# ==============================================================================
# Final Runtime Stage: Minimal Hardened Alpine Container
# ==============================================================================
FROM alpine:3.21

# Install CA certs and curl for container healthchecks
RUN apk add --no-cache ca-certificates tzdata curl && \
    addgroup -g 10001 -S goinfer && \
    adduser -u 10001 -S goinfer -G goinfer -s /sbin/nologin

# Create model volume and cache directories with proper permissions
RUN mkdir -p /models /app /var/log/goinfer && \
    chown -R goinfer:goinfer /models /app /var/log/goinfer

# Copy static binaries from builder stage
COPY --from=builder /bin/goinfer /usr/local/bin/goinfer
COPY --from=builder /bin/quantize /usr/local/bin/quantize
COPY --from=builder /bin/bench /usr/local/bin/bench

# Symlink alias for go-infer
RUN ln -s /usr/local/bin/goinfer /usr/local/bin/go-infer

USER goinfer
WORKDIR /models

# Expose HTTP API & Web UI port
EXPOSE 8080

# Built-in Healthcheck against /health endpoint
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Default Entrypoint & Command
ENTRYPOINT ["/usr/local/bin/goinfer"]
CMD ["--serve", ":8080", "/models/model.gguf"]
