package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/tiktoken-go/tokenizer"
)

type claudeResult struct {
	Usage struct {
		InputTokens int `json:"input_tokens"`
	} `json:"usage"`
}

type row struct {
	Case        string
	Tiktoken    int
	Sub2API     int
	ClaudeCode  string
	ClaudeError string
}

func main() {
	cases := []string{
		"hello world",
		"hello  world",
		"hello   world",
		"supercalifragilistic",
		"We know what we are, but know not what we may be.",
	}

	codec, err := tokenizer.Get(tokenizer.O200kBase)
	if err != nil {
		fail("load o200k_base tokenizer", err)
	}

	claudeAvailable := claudeInInteractiveShell()
	rows := make([]row, 0, len(cases))
	for _, text := range cases {
		tokCount, err := codec.Count(text)
		if err != nil {
			fail("count tiktoken", err)
		}

		// Mirrors the current sub2api OAuth fallback behavior for these plain
		// single-user-text cases:
		// 3 item overhead + token("user") + 1 content-part overhead + token(text).
		roleCount, err := codec.Count("user")
		if err != nil {
			fail("count role token", err)
		}

		r := row{
			Case:     text,
			Tiktoken: tokCount,
			Sub2API:  4 + roleCount + tokCount,
		}

		if claudeAvailable {
			val, err := runClaudeCount(text)
			if err != nil {
				r.ClaudeCode = "ERR"
				r.ClaudeError = err.Error()
			} else {
				r.ClaudeCode = fmt.Sprintf("%d", val)
			}
		} else {
			r.ClaudeCode = "N/A"
			r.ClaudeError = "claude not available from interactive zsh"
		}

		rows = append(rows, r)
	}

	fmt.Println("| Case | tiktoken | sub2api interface | claude code |")
	fmt.Println("|---|---:|---:|---:|")
	for _, r := range rows {
		fmt.Printf("| `%s` | %d | %d | %s |\n", escapeTable(r.Case), r.Tiktoken, r.Sub2API, r.ClaudeCode)
	}

	hasErr := false
	for _, r := range rows {
		if r.ClaudeError != "" {
			if !hasErr {
				fmt.Println()
				fmt.Println("Claude details:")
				hasErr = true
			}
			fmt.Printf("- `%s`: %s\n", r.Case, r.ClaudeError)
		}
	}
}

func runClaudeCount(text string) (int, error) {
	shellCmd := "cd /tmp && claude -p --no-session-persistence --output-format json " + shellQuote(text)
	cmd := exec.Command("zsh", "-ic", shellCmd)
	cmd.Env = os.Environ()
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err != nil {
		return 0, fmt.Errorf("run claude: %w: stdout=%s stderr=%s", err, strings.TrimSpace(stdout.String()), strings.TrimSpace(stderr.String()))
	}

	var result claudeResult
	raw := strings.TrimSpace(stdout.String())
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		return 0, fmt.Errorf("parse claude json: %w: stdout=%s stderr=%s", err, raw, strings.TrimSpace(stderr.String()))
	}
	if result.Usage.InputTokens <= 0 {
		return 0, fmt.Errorf("claude usage.input_tokens missing: stdout=%s stderr=%s", raw, strings.TrimSpace(stderr.String()))
	}
	return result.Usage.InputTokens, nil
}

func claudeInInteractiveShell() bool {
	cmd := exec.Command("zsh", "-ic", "command -v claude >/dev/null 2>&1")
	cmd.Env = os.Environ()
	return cmd.Run() == nil
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'"'"'`) + "'"
}

func escapeTable(s string) string {
	return strings.ReplaceAll(s, "|", `\|`)
}

func fail(step string, err error) {
	fmt.Fprintf(os.Stderr, "%s: %v\n", step, err)
	os.Exit(1)
}
