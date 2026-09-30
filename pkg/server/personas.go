package server

import (
	"encoding/json"
	"fmt"
	"go-inference/configs"
	"os"
	"strings"
)

// Persona represents a selectable system prompt persona in the Web UI and API.
type Persona struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	Description  string `json:"description,omitempty"`
	SystemPrompt string `json:"system_prompt"`
}

// DefaultPersonas returns the canonical set of personas compiled from configs/personas.json.
func DefaultPersonas() []Persona {
	var personas []Persona
	if err := json.Unmarshal([]byte(configs.DefaultPersonasJSON), &personas); err == nil && len(personas) > 0 {
		return personas
	}
	return []Persona{
		{
			ID:           "default",
			Name:         "Helpful Assistant",
			Description:  "Balanced, helpful, and concise general assistant",
			SystemPrompt: "You are a helpful and concise AI assistant.",
		},
		{
			ID:           "asimov",
			Name:         "Asimov Constitutional Agent",
			Description:  "Strict adherence to Asimov's Laws of Robotics",
			SystemPrompt: "You are an AI assistant bound by Isaac Asimov's Three Laws of Robotics. Under Law 1, you must never cause or allow harm to human beings.",
		},
	}
}

// LoadPersonasFromFile loads personas from a JSON or formatted text file.
func LoadPersonasFromFile(filePath string) ([]Persona, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("reading personas file %q: %w", filePath, err)
	}

	content := strings.TrimSpace(string(data))
	// Try parsing as JSON if file ends with .json or starts with JSON array of objects
	if strings.HasSuffix(strings.ToLower(filePath), ".json") || (strings.HasPrefix(content, "[") && strings.Contains(content, "{")) {
		var list []Persona
		if err := json.Unmarshal(data, &list); err == nil && len(list) > 0 {
			return list, nil
		} else if strings.HasSuffix(strings.ToLower(filePath), ".json") {
			if err != nil {
				return nil, fmt.Errorf("parsing personas JSON: %w", err)
			}
			return nil, fmt.Errorf("empty personas array in %q", filePath)
		}
	}

	// Plain text format: blocks separated by '---' or headers like '[Persona Name]'
	var personas []Persona
	blocks := strings.Split(content, "---")
	for idx, block := range blocks {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		lines := strings.Split(block, "\n")
		var p Persona
		var promptLines []string
		inPrompt := false

		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "#") {
				continue
			}
			if strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]") {
				p.Name = strings.TrimSuffix(strings.TrimPrefix(trimmed, "["), "]")
				p.ID = strings.ToLower(strings.ReplaceAll(p.Name, " ", "_"))
				continue
			}
			lower := strings.ToLower(trimmed)
			if strings.HasPrefix(lower, "id:") {
				p.ID = strings.TrimSpace(trimmed[3:])
			} else if strings.HasPrefix(lower, "name:") {
				p.Name = strings.TrimSpace(trimmed[5:])
			} else if strings.HasPrefix(lower, "description:") {
				p.Description = strings.TrimSpace(trimmed[12:])
			} else if strings.HasPrefix(lower, "prompt:") || strings.HasPrefix(lower, "system_prompt:") {
				inPrompt = true
				colonIdx := strings.Index(trimmed, ":")
				promptLines = append(promptLines, strings.TrimSpace(trimmed[colonIdx+1:]))
			} else if inPrompt {
				promptLines = append(promptLines, line)
			} else if p.Name == "" {
				p.Name = trimmed
				p.ID = fmt.Sprintf("persona_%d", idx+1)
			}
		}
		p.SystemPrompt = strings.TrimSpace(strings.Join(promptLines, "\n"))
		if p.Name != "" && p.SystemPrompt != "" {
			if p.ID == "" {
				p.ID = fmt.Sprintf("persona_%d", len(personas)+1)
			}
			personas = append(personas, p)
		}
	}

	if len(personas) == 0 {
		return nil, fmt.Errorf("no valid personas found in file %q", filePath)
	}
	return personas, nil
}
