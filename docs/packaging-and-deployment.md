# Packaging & Enterprise Deployment Guide

This guide covers deployment options for `go-infer`, including native Linux system packages (`.deb` / `.rpm`), systemd service orchestration, Docker containers, Apple Metal GPU acceleration, and iOS mobile compilation.

---

## 1. Native Linux Distribution Packages (`.deb` & `.rpm`)

`go-infer` includes an in-tree, pure Go universal package generator (`cmd/package`) that creates production-ready Debian and Red Hat packages without requiring external tools like `fpm` or `dpkg-deb`.

### Building Packages
```bash
make packages
```
Generated installers are saved to `bin/dist/`:
- `goinfer_0.1_amd64.deb` (Ubuntu / Debian x86_64)
- `goinfer_0.1_arm64.deb` (Ubuntu / Debian ARM64 / AWS Graviton)
- `goinfer-0.1-1.x86_64.rpm` (RHEL / CentOS / Rocky Linux / Fedora)
- `goinfer-0.1-1.aarch64.rpm` (RHEL / Fedora ARM64)

### Installing on Ubuntu / Debian
```bash
sudo dpkg -i bin/dist/goinfer_0.1_amd64.deb
```

### Installing on RHEL / CentOS / Rocky Linux
```bash
sudo rpm -ivh bin/dist/goinfer-0.1-1.x86_64.rpm
```

### Systemd Service Management
The package registers a hardened, sandboxed systemd unit running under a dedicated unprivileged `goinfer` user (`ProtectSystem=full`, `ProtectHome=true`):

```bash
# Configure model file location and listen port
sudo nano /etc/goinfer/goinfer.conf

# Start and enable the service
sudo systemctl enable --now goinfer

# Inspect live status
sudo systemctl status goinfer

# Stream logs
sudo journalctl -u goinfer -f
```

---

## 2. Docker & Container Deployment

### Multi-Stage Container Build
The included [`Dockerfile`](file:///Users/andrewmarcum/git/go-infer/Dockerfile) uses multi-stage compilation to build a scratch/alpine-based unprivileged container:

```bash
docker build -t go-infer:latest .
```

### Running with Local Models Mounted
```bash
docker run -d \
  --name goinfer \
  -p 8080:8080 \
  -v $(pwd)/models:/models \
  go-infer:latest --serve :8080 /models/model.gguf
```

### Docker Compose
Deploy instantly with [`docker-compose.yml`](file:///Users/andrewmarcum/git/go-infer/docker-compose.yml):
```bash
docker compose up -d
```
Access the streaming Web UI dashboard at `http://localhost:8080`.

---

## 3. Apple Metal GPU & Architecture Support

### Automatic Build-Time Hardware Detection
`go-infer` determines compute acceleration automatically during compilation without requiring runtime CLI flags:
- **Apple Silicon macOS (M1/M2/M3/M4):** When compiled on macOS with CGO enabled (the default for `go build` and `make build`), the build process automatically activates [`pkg/metal`](file:///Users/andrewmarcum/git/go-infer/pkg/metal) and [`inference_core.c`](file:///Users/andrewmarcum/git/go-infer/inference_core.c). The engine dynamically binds to Apple's Metal runtime framework with 8-way SIMD dispatch, fused SwiGLU kernels, and 100% unified memory residency.
- **Non-Apple Architectures (Linux x86_64, Linux ARM64, Windows, etc.):** When building on or targeting non-Apple operating systems, Go build tags (`//go:build !darwin || !cgo`) automatically substitute [`pkg/metal/metal_fallback.go`](file:///Users/andrewmarcum/git/go-infer/pkg/metal/metal_fallback.go) and standard ISO C99 CPU routines. The engine automatically runs on the parallel multi-threaded CPU GEMV matrix multiplication engine ([`pkg/math/gemv.go`](file:///Users/andrewmarcum/git/go-infer/pkg/math/gemv.go)).

### iOS Mobile Deployment (iPhone 15 Pro / 16 / Apple A17 & A18)
Because the codebase is built in Go + Metal MSL shaders, it compiles directly for iOS ARM64:

```bash
# Compile standalone iOS binary for testing / sideloading
CGO_ENABLED=1 GOOS=ios GOARCH=arm64 go build -o goinfer-ios .

# Or build an iOS .xcframework library via gomobile
gomobile bind -target=ios -o GoInfer.xcframework ./pkg/engine ./pkg/server ./pkg/guardrails
```

---

## 4. Cross-Compilation Reference

Thanks to the zero-dependency standard library design and the non-CGO fallback ([`c_bridge_nocgo.go`](file:///Users/andrewmarcum/git/go-infer/c_bridge_nocgo.go)), cross-compiling for any target architecture is instant and requires no external toolchains:

```bash
# Build all supported platforms at once
make release
```

Individual platform commands:
```bash
# macOS Apple Silicon (ARM64)
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -o bin/go-infer-darwin-arm64 .

# macOS Intel (x86_64 / AMD64)
CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build -o bin/go-infer-darwin-amd64 .

# Linux (x86_64 / AMD64)
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o bin/go-infer-linux-amd64 .

# Linux (ARM64 / Graviton / Raspberry Pi)
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o bin/go-infer-linux-arm64 .

# Windows (x86_64)
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o bin/go-infer-windows-amd64.exe .
```
