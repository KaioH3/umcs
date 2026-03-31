package infer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

type GroqClient struct {
	apiKey    string
	baseURL   string
	client    *http.Client
	mu        sync.Mutex
	lastReq   time.Time
	rateLimit time.Duration
	cache     map[string]*GroqEtymology
}

type GroqEtymology struct {
	Root      string `json:"root"`
	Lang      string `json:"lang"`
	Origin    string `json:"origin"`
	MeaningEN string `json:"meaning_en"`
}

type GroqRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
	MaxTokens   int       `json:"max_tokens"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type GroqResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func NewGroqClient(apiKey string) *GroqClient {
	return &GroqClient{
		apiKey:    apiKey,
		baseURL:   "https://api.groq.com/openai/v1/chat/completions",
		client:    &http.Client{Timeout: 30 * time.Second},
		rateLimit: 2 * time.Second,
		cache:     make(map[string]*GroqEtymology),
	}
}

func (c *GroqClient) InferEtymology(word, lang string) (*GroqEtymology, error) {
	if c.apiKey == "" {
		return nil, fmt.Errorf("GROQ_API_KEY not set")
	}

	wordLower := strings.ToLower(word)
	if cached, ok := c.cache[wordLower]; ok {
		return cached, nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if time.Since(c.lastReq) < c.rateLimit {
		time.Sleep(c.rateLimit - time.Since(c.lastReq))
	}
	c.lastReq = time.Now()

	prompt := fmt.Sprintf(`You are a linguistic etymology expert. Given a word, identify its etymological root.

Respond ONLY with valid JSON array. No other text.

Examples:
Input: "beauty" (EN)
Output: [{"root":"bel","lang":"EN","origin":"LATIN","meaning_en":"beautiful"}]

Input: "bonito" (ES)  
Output: [{"root":"bon","lang":"ES","origin":"LATIN","meaning_en":"good"}]

Input: "lindo" (PT)
Output: [{"root":"bel","lang":"PT","origin":"LATIN","meaning_en":"beautiful"}]

Now analyze:
Input: "%s" (%s)
Output:`, word, lang)

	req := GroqRequest{
		Model: "llama-3.1-8b-instant",
		Messages: []Message{
			{Role: "user", Content: prompt},
		},
		Temperature: 0.1,
		MaxTokens:   256,
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequest("POST", c.baseURL, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Groq API error: %s", string(body))
	}

	var groqResp GroqResponse
	if err := json.NewDecoder(resp.Body).Decode(&groqResp); err != nil {
		return nil, err
	}

	if len(groqResp.Choices) == 0 {
		return nil, fmt.Errorf("no response from Groq")
	}

	content := groqResp.Choices[0].Message.Content
	content = strings.TrimSpace(content)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	var etymologies []GroqEtymology
	if err := json.Unmarshal([]byte(content), &etymologies); err != nil {
		return nil, fmt.Errorf("parse JSON: %v", err)
	}

	if len(etymologies) == 0 {
		return nil, fmt.Errorf("no etymology found")
	}

	result := &etymologies[0]
	c.cache[wordLower] = result

	return result, nil
}

func GetAPIKey() string {
	return os.Getenv("GROQ_API_KEY")
}
