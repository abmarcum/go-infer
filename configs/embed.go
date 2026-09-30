package configs

import _ "embed"

// DefaultGuardrailsText embeds configs/guardrails.txt as the compile-time default constitution.
//
//go:embed guardrails.txt
var DefaultGuardrailsText string

// DefaultPersonasJSON embeds configs/personas.json as the compile-time default system personas.
//
//go:embed personas.json
var DefaultPersonasJSON string

// DefaultPersonasText embeds configs/personas.txt as the compile-time formatted text persona reference.
//
//go:embed personas.txt
var DefaultPersonasText string
