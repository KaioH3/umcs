package main

import (
	"fmt"
	"os"
	"os/exec"
)

func main() {
	run("clear")

	fmt.Print(`
╭──────────────────────────────────────────────────────────────╮
│  UMCS - Universal Morpheme Coordinate System                │
│  5.1M Words | 44 Languages | Binary Semantic Encoding     │
╰──────────────────────────────────────────────────────────────╯
`)

	fmt.Println()

	demo("PORTUGUESE SENTIMENT", []string{
		"este filme é maravilhoso",
		"isso é horrível e me apavora",
		"feliz aniversário pra você",
	})

	fmt.Println()
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Println("                       ENGLISH SENTIMENT")
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Println()

	run(`./lexsent analyze "i love you more than words can say"`)
	fmt.Println()
	run(`./lexsent analyze "terrible horrible disgusting day"`)
	fmt.Println()
	run(`./lexsent analyze "this is absolutely wonderful and amazing"`)

	fmt.Println()
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Println("                         LEXICON STATS")
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Println()
	run("./lexsent stats")

	fmt.Println()
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Println("                       COGNATE LOOKUPS")
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Println()

	fmt.Println("  amor (love):")
	run("./lexsent lookup amor | head -15")
	fmt.Println()

	fmt.Println("  triste (sad):")
	run("./lexsent lookup triste | head -15")
	fmt.Println()

	fmt.Println("  feliz (happy):")
	run("./lexsent lookup feliz | head -15")
}

func demo(title string, texts []string) {
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Printf("                       %s\n", title)
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Println()

	for _, text := range texts {
		fmt.Printf("  %q\n", text)
		run(`./lexsent analyze "` + text + `"`)
		fmt.Println()
	}
}

func run(cmd string) {
	c := exec.Command("sh", "-c", cmd)
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	c.Run()
}
