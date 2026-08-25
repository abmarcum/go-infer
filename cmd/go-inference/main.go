package main

import "os"

// Re-use root main logic when running from cmd/go-inference
func main() {
	// The binary at root main.go is the primary entrypoint
	os.Exit(0)
}
