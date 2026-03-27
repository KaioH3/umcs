// Package capi exposes UMCS as a C shared library (libumcs.so).
//
// Build:
//
//	go build -buildmode=c-shared -o libumcs.so github.com/kak/umcs/pkg/capi
//
// The generated umcs.h declares all exported symbols. Link with -lumcs.
package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"sync"
	"unsafe"

	"github.com/kak/umcs/pkg/analyze"
	"github.com/kak/umcs/pkg/classify"
	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/morpheme"
)

// ── Static string interning ────────────────────────────────────────────────────
//
// C.CString allocates a heap buffer per call. For functions declared as
// returning "static" strings in umcs.h (caller must NOT free), we intern
// C strings by value so each unique string is allocated exactly once.
// This prevents unbounded memory growth when called in tight loops.
var staticCStrings sync.Map // string → *C.char

// staticStr returns an interned *C.char for s. The pointer is valid for the
// lifetime of the process. Do NOT pass the result to C.free().
func staticStr(s string) *C.char {
	if v, ok := staticCStrings.Load(s); ok {
		return v.(*C.char)
	}
	cs := C.CString(s)
	// Store with LoadOrStore to avoid double-allocation under race.
	actual, loaded := staticCStrings.LoadOrStore(s, cs)
	if loaded {
		C.free(unsafe.Pointer(cs)) // lost the race — free the duplicate
		return actual.(*C.char)
	}
	return cs
}

// ── global state (protected by mu) ────────────────────────────────────────────

var (
	mu  sync.RWMutex
	lex *lexdb.Lexicon
	clf *classify.Classifier
)

const version = "2.0.0"

// ── umcs_load ─────────────────────────────────────────────────────────────────

//export umcs_load
func umcs_load(path *C.char) C.int32_t {
	p := C.GoString(path)
	l, err := lexdb.Load(p)
	if err != nil {
		return -1
	}
	mu.Lock()
	lex = l
	mu.Unlock()
	return 0
}

// ── umcs_load_model ───────────────────────────────────────────────────────────

//export umcs_load_model
func umcs_load_model(path *C.char) C.int32_t {
	p := C.GoString(path)
	c, err := classify.Load(p)
	if err != nil {
		return -1
	}
	mu.Lock()
	clf = c
	mu.Unlock()
	return 0
}

// ── umcs_lookup ───────────────────────────────────────────────────────────────

//export umcs_lookup
func umcs_lookup(word, lang *C.char) C.uint64_t {
	w, l := C.GoString(word), C.GoString(lang)
	mu.RLock()
	l2 := lex
	mu.RUnlock()
	if l2 == nil {
		return 0
	}
	wr := l2.LookupWordInLang(w, l)
	if wr == nil {
		return 0
	}
	tok := morpheme.Pack64(wr.WordID, wr.Sentiment, wr.Flags)
	return C.uint64_t(uint64(tok))
}

// ── umcs_classify_word ────────────────────────────────────────────────────────

//export umcs_classify_word
func umcs_classify_word(word, lang *C.char) *C.char {
	w, l := C.GoString(word), C.GoString(lang)
	mu.RLock()
	l2, c := lex, clf
	mu.RUnlock()
	if l2 == nil || c == nil {
		return staticStr("UNKNOWN")
	}
	f, ok := classify.ExtractFromLexicon(l2, w, l)
	if !ok {
		return staticStr("UNKNOWN")
	}
	class, _ := c.Predict(f)
	return staticStr(class)
}

// ── umcs_analyze ──────────────────────────────────────────────────────────────

//export umcs_analyze
func umcs_analyze(text *C.char) *C.char {
	t := C.GoString(text)
	mu.RLock()
	l2 := lex
	mu.RUnlock()
	if l2 == nil {
		return staticStr("{\"error\":\"lexicon not loaded\"}")
	}
	result := analyze.Analyze(l2, t)
	data, err := json.Marshal(result)
	if err != nil {
		return staticStr("{\"error\":\"json marshal failed\"}")
	}
	// umcs_analyze returns a caller-owned string (must be freed with umcs_free).
	// This is the only function that allocates a new C string per call.
	return C.CString(string(data))
}

// ── umcs_ipa ──────────────────────────────────────────────────────────────────

//export umcs_ipa
func umcs_ipa(word, lang *C.char) *C.char {
	w, l := C.GoString(word), C.GoString(lang)
	mu.RLock()
	l2 := lex
	mu.RUnlock()
	if l2 == nil {
		return staticStr("")
	}
	wr := l2.LookupWordInLang(w, l)
	if wr == nil {
		return staticStr("")
	}
	return staticStr(l2.WordPron(wr))
}

// ── umcs_antonym ──────────────────────────────────────────────────────────────

//export umcs_antonym
func umcs_antonym(word, lang *C.char) *C.char {
	w, l := C.GoString(word), C.GoString(lang)
	mu.RLock()
	l2 := lex
	mu.RUnlock()
	if l2 == nil {
		return staticStr("")
	}
	wr := l2.LookupWordInLang(w, l)
	if wr == nil {
		return staticStr("")
	}
	root := l2.LookupRoot(wr.RootID)
	if root == nil {
		return staticStr("")
	}
	ant := l2.Antonym(root)
	if ant == nil {
		return staticStr("")
	}
	return staticStr(l2.RootStr(ant))
}

// ── umcs_version ──────────────────────────────────────────────────────────────

//export umcs_version
func umcs_version() *C.char {
	return staticStr(version)
}

// ── umcs_free ─────────────────────────────────────────────────────────────────

//export umcs_free
func umcs_free(ptr *C.char) {
	C.free(unsafe.Pointer(ptr))
}

// main is required for c-shared build mode.
func main() {}
