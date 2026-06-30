package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
)

type docCase struct {
	Name        string
	ExpectedDoc int
	Body        string
	Params      func() anthropic.MessageCountTokensParams
}

type compareRow struct {
	Name           string
	ExpectedDoc    int
	AnthropicAPI   string
	Sub2API        string
	AnthropicError string
	Sub2APIError   string
}

type noopHTTPUpstream struct {
	resp *http.Response
}

func (n *noopHTTPUpstream) Do(req *http.Request, proxyURL string, accountID int64, accountConcurrency int) (*http.Response, error) {
	return cloneHTTPResponse(n.resp), nil
}

func (n *noopHTTPUpstream) DoWithTLS(req *http.Request, proxyURL string, accountID int64, accountConcurrency int, profile *tlsfingerprint.Profile) (*http.Response, error) {
	return cloneHTTPResponse(n.resp), nil
}

func main() {
	cases := []docCase{
		{
			Name:        "basic messages",
			ExpectedDoc: 14,
			Body: `{
  "model": "claude-opus-4-8",
  "system": "You are a scientist",
  "messages": [
    {"role": "user", "content": "Hello, Claude"}
  ]
}`,
			Params: func() anthropic.MessageCountTokensParams {
				return anthropic.MessageCountTokensParams{
					Model: anthropic.ModelClaudeOpus4_8,
					System: anthropic.MessageCountTokensParamsSystemUnion{
						OfString: anthropic.String("You are a scientist"),
					},
					Messages: []anthropic.MessageParam{
						{
							Role: anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{
								{OfText: &anthropic.TextBlockParam{Text: "Hello, Claude"}},
							},
						},
					},
				}
			},
		},
		{
			Name:        "messages with tools",
			ExpectedDoc: 403,
			Body: `{
  "model": "claude-opus-4-8",
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather in a given location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          }
        },
        "required": ["location"]
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "What's the weather like in San Francisco?"}
  ]
}`,
			Params: func() anthropic.MessageCountTokensParams {
				return anthropic.MessageCountTokensParams{
					Model: anthropic.ModelClaudeOpus4_8,
					Tools: []anthropic.MessageCountTokensToolUnionParam{
						{
							OfTool: &anthropic.ToolParam{
								Name:        "get_weather",
								Description: anthropic.String("Get the current weather in a given location"),
								InputSchema: anthropic.ToolInputSchemaParam{
									Properties: map[string]any{
										"location": map[string]any{
											"type":        "string",
											"description": "The city and state, e.g. San Francisco, CA",
										},
									},
									Required: []string{"location"},
								},
							},
						},
					},
					Messages: []anthropic.MessageParam{
						{
							Role: anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{
								{OfText: &anthropic.TextBlockParam{Text: "What's the weather like in San Francisco?"}},
							},
						},
					},
				}
			},
		},
		{
			Name:        "messages with extended thinking",
			ExpectedDoc: 88,
			Body: `{
  "model": "claude-sonnet-4-6",
  "thinking": {"type": "enabled", "budget_tokens": 16000},
  "messages": [
    {
      "role": "user",
      "content": "Are there an infinite number of prime numbers such that n mod 4 == 3?"
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "thinking",
          "thinking": "This is a nice number theory question. Let's think about it step by step...",
          "signature": "EuYBCkQYAiJAgCs1le6/Pol5Z4/JMomVOouGrWdhYNsH3ukzUECbB6iWrSQtsQuRHJID6lWV..."
        },
        {
          "type": "text",
          "text": "Yes, there are infinitely many prime numbers p such that p mod 4 = 3..."
        }
      ]
    },
    {"role": "user", "content": "Can you write a formal proof?"}
  ]
}`,
			Params: func() anthropic.MessageCountTokensParams {
				return anthropic.MessageCountTokensParams{
					Model: anthropic.Model("claude-sonnet-4-6"),
					Thinking: anthropic.ThinkingConfigParamUnion{
						OfEnabled: &anthropic.ThinkingConfigEnabledParam{
							BudgetTokens: 16000,
						},
					},
					Messages: []anthropic.MessageParam{
						{
							Role: anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{
								{OfText: &anthropic.TextBlockParam{Text: "Are there an infinite number of prime numbers such that n mod 4 == 3?"}},
							},
						},
						{
							Role: anthropic.MessageParamRoleAssistant,
							Content: []anthropic.ContentBlockParamUnion{
								{OfThinking: &anthropic.ThinkingBlockParam{
									Thinking:  "This is a nice number theory question. Let's think about it step by step...",
									Signature: "EuYBCkQYAiJAgCs1le6/Pol5Z4/JMomVOouGrWdhYNsH3ukzUECbB6iWrSQtsQuRHJID6lWV...",
								}},
								{OfText: &anthropic.TextBlockParam{Text: "Yes, there are infinitely many prime numbers p such that p mod 4 = 3..."}},
							},
						},
						{
							Role: anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{
								{OfText: &anthropic.TextBlockParam{Text: "Can you write a formal proof?"}},
							},
						},
					},
				}
			},
		},
	}

	rows := make([]compareRow, 0, len(cases))
	for _, tc := range cases {
		row := compareRow{
			Name:        tc.Name,
			ExpectedDoc: tc.ExpectedDoc,
		}

		if actual, err := callAnthropicCountTokens(tc.Params()); err != nil {
			row.AnthropicAPI = "N/A"
			row.AnthropicError = err.Error()
		} else {
			row.AnthropicAPI = fmt.Sprintf("%d", actual)
		}

		if actual, err := callSub2APICountTokens(tc.Body); err != nil {
			row.Sub2API = "ERR"
			row.Sub2APIError = err.Error()
		} else {
			row.Sub2API = fmt.Sprintf("%d", actual)
		}

		rows = append(rows, row)
	}

	fmt.Println("| Case | Anthropic doc expected | Anthropic SDK/API actual | sub2api interface |")
	fmt.Println("|---|---:|---:|---:|")
	for _, row := range rows {
		fmt.Printf("| `%s` | %d | %s | %s |\n", row.Name, row.ExpectedDoc, row.AnthropicAPI, row.Sub2API)
	}

	printDetails("Anthropic API details", rows, func(r compareRow) string { return r.AnthropicError })
	printDetails("sub2api details", rows, func(r compareRow) string { return r.Sub2APIError })

	fmt.Println()
	fmt.Println("Notes:")
	fmt.Println("- These rows come directly from Anthropic's token counting docs.")
	fmt.Println("- The Anthropic actual column uses the official anthropic-sdk-go client.")
	fmt.Println("- The sub2api column sends the same raw JSON payload into the current OpenAI OAuth fallback path.")
}

func callAnthropicCountTokens(params anthropic.MessageCountTokensParams) (int, error) {
	if !hasAnthropicCredentials() {
		return 0, fmt.Errorf("no Anthropic SDK credentials found (checked ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / profiles / federation)")
	}

	client := anthropic.NewClient()
	resp, err := client.Messages.CountTokens(context.Background(), params)
	if err != nil {
		return 0, fmt.Errorf("sdk count_tokens: %w", err)
	}
	return int(resp.InputTokens), nil
}

func hasAnthropicCredentials() bool {
	if strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")) != "" {
		return true
	}
	if strings.TrimSpace(os.Getenv("ANTHROPIC_AUTH_TOKEN")) != "" {
		return true
	}
	if strings.TrimSpace(os.Getenv("ANTHROPIC_PROFILE")) != "" {
		return true
	}
	if strings.TrimSpace(os.Getenv("ANTHROPIC_FEDERATION_RULE_ID")) != "" {
		return true
	}
	return false
}

func callSub2APICountTokens(body string) (int, error) {
	gin.SetMode(gin.TestMode)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(body))
	c.Request.Header.Set("content-type", "application/json")
	c.Request.Header.Set("user-agent", "Claude-Code/1.0")

	upstream := &noopHTTPUpstream{
		resp: &http.Response{
			StatusCode: http.StatusUnauthorized,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"error":{"type":"invalid_request_error","message":"unauthorized"}}`)),
		},
	}

	svc := service.NewOpenAIGatewayService(
		nil, // accountRepo
		nil, // usageLogRepo
		nil, // usageBillingRepo
		nil, // userRepo
		nil, // userSubRepo
		nil, // userGroupRateRepo
		nil, // cache
		&config.Config{}, // cfg
		nil, // schedulerSnapshot
		nil, // concurrencyService
		nil, // billingService
		nil, // rateLimitService
		nil, // billingCacheService
		upstream,
		nil, // deferredService
		nil, // openAITokenProvider
		nil, // grokTokenProvider
		nil, // resolver
		nil, // channelService
		nil, // balanceNotifyService
		nil, // settingService
		nil, // userPlatformQuotaRepo
	)

	account := &service.Account{
		ID:          202,
		Name:        "openai-oauth",
		Platform:    service.PlatformOpenAI,
		Type:        service.AccountTypeOAuth,
		Concurrency: 1,
		Credentials: map[string]any{
			"access_token": "oauth-token",
		},
		Status:      service.StatusActive,
		Schedulable: true,
	}

	if err := svc.ForwardCountTokensAsAnthropic(context.Background(), c, account, []byte(body), "gpt-5"); err != nil {
		return 0, err
	}

	var parsed struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &parsed); err != nil {
		return 0, fmt.Errorf("parse response: %w body=%s", err, strings.TrimSpace(rec.Body.String()))
	}
	return parsed.InputTokens, nil
}

func cloneHTTPResponse(resp *http.Response) *http.Response {
	if resp == nil {
		return nil
	}

	clone := *resp
	if resp.Body == nil {
		return &clone
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body = io.NopCloser(bytes.NewReader(body))
	clone.Body = io.NopCloser(bytes.NewReader(body))
	return &clone
}

func printDetails(title string, rows []compareRow, pick func(compareRow) string) {
	printed := false
	for _, row := range rows {
		if msg := pick(row); msg != "" {
			if !printed {
				fmt.Println()
				fmt.Println(title + ":")
				printed = true
			}
			fmt.Printf("- `%s`: %s\n", row.Name, msg)
		}
	}
}
