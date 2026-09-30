//go:build cgo

package main

/*
#cgo CFLAGS: -I.
#include "inference_core.h"
#include <stdlib.h>
#include <stdint.h>
*/
import "C"
import (
	"errors"
	"sync"
	"unsafe"
)

// CModelContext wraps the C inference engine ModelContext with thread-safe mutex synchronization.
type CModelContext struct {
	mu  sync.Mutex
	ctx *C.ModelContext
}

// InitCModel initializes the C core inference engine with weights from the specified path.
func InitCModel(path string) (*CModelContext, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	ctx := C.init_model(cPath)
	if ctx == nil {
		return nil, errors.New("failed to initialize C inference engine weights")
	}
	return &CModelContext{ctx: ctx}, nil
}

// HasMetal returns whether the C core successfully initialized an Apple Metal GPU device.
func (m *CModelContext) HasMetal() bool {
	if m == nil || m.ctx == nil {
		return false
	}
	return int(C.model_has_metal(m.ctx)) != 0
}

// DeviceName returns the compute device name (e.g. "Apple M1 Max" or fallback).
func (m *CModelContext) DeviceName() string {
	if m == nil || m.ctx == nil {
		return "Unknown"
	}
	cName := C.model_device_name(m.ctx)
	if cName == nil {
		return "Unknown"
	}
	return C.GoString(cName)
}

// Generate executes token generation using the C inference core under mutex synchronization.
func (m *CModelContext) Generate(prompt string) (string, error) {
	if m == nil || m.ctx == nil {
		return "", errors.New("model weights not loaded in C core")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cOutput := C.generate_tokens(m.ctx, cPrompt)
	defer C.free(unsafe.Pointer(cOutput))

	return C.GoString(cOutput), nil
}

// Close releases the C core model resources.
func (m *CModelContext) Close() {
	if m != nil && m.ctx != nil {
		m.mu.Lock()
		defer m.mu.Unlock()
		C.free_model(m.ctx)
		m.ctx = nil
	}
}
