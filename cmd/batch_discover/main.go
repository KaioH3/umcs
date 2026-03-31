package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: batch_discover <wordlist.txt> <langs>")
		fmt.Println("  wordlist.txt: one word per line")
		fmt.Println("  langs: comma-separated (EN,PT,ES,IT,DE,FR)")
		os.Exit(1)
	}

	wordListPath := os.Args[1]
	langs := os.Args[2]
	batchSize := 20
	concurrency := 3

	file, err := os.Open(wordListPath)
	if err != nil {
		log.Fatalf("open wordlist: %v", err)
	}
	defer file.Close()

	var words []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		word := strings.TrimSpace(scanner.Text())
		if word != "" && !strings.HasPrefix(word, "#") {
			words = append(words, word)
		}
	}

	fmt.Printf("Loaded %d words, discovering in batches of %d...\n", len(words), batchSize)

	var wg sync.WaitGroup
	sem := make(chan struct{}, concurrency)
	var mu sync.Mutex
	processed := 0

	for i := 0; i < len(words); i += batchSize {
		end := i + batchSize
		if end > len(words) {
			end = len(words)
		}
		batch := words[i:end]

		wg.Add(1)
		go func(batch []string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			args := []string{"discover-remote", "--lang", langs, "--append"}
			args = append(args, batch...)

			cmd := exec.Command("./umcs", args...)
			output, err := cmd.CombinedOutput()
			mu.Lock()
			processed += len(batch)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Batch error: %v\n%s\n", err, string(output))
			} else {
				fmt.Printf("Processed %d/%d words\n", processed, len(words))
			}
			mu.Unlock()

			time.Sleep(2 * time.Second)
		}(batch)
	}

	wg.Wait()
	fmt.Println("Done!")
}
