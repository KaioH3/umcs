# UMCS — Universal Morpheme Coordinate System

> One number carries the entire semantic identity of a word — across every human language.

[![Go](https://img.shields.io/badge/Go-1.24-blue?logo=go)](https://go.dev)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-18_packages-brightgreen)]()
[![Languages](https://img.shields.io/badge/languages-70-orange)]()
[![Words](https://img.shields.io/badge/words-4.37M-purple)]()
[![Datasets](https://img.shields.io/badge/datasets-23-yellow)](DATASETS.md)

---

## What is UMCS?

UMCS assigns every word in every language a **deterministic coordinate** based on its etymological root. Words that share the same ancestor share the same coordinate prefix — regardless of language, script, or writing system.

```
"terrible" (EN) → word_id = 40961   (root_id=10, variant=1)
"terrível"  (PT) → word_id = 40962   (root_id=10, variant=2)
"terrible"  (FR) → word_id = 40963   (root_id=10, variant=3)
"terribile" (IT) → word_id = 40964   (root_id=10, variant=4)
"terrible"  (ES) → word_id = 40965   (root_id=10, variant=5)
```

Same root (`terr`, Latin *terror*) = same embedding slot in any model. No translation step. No alignment. One bit shift and you know two words are cognates.

---

## Why UMCS?

| Feature | UMCS | VADER | spaCy | Regex | DistilBERT |
|---------|------|-------|-------|-------|------------|
| Cross-lingual single lookup | **O(1)** | - | - | - | - |
| Scope-aware sentiment (negation, intensifiers) | yes | yes | yes | - | yes |
| Etymology chains to proto-language | yes | - | - | - | - |
| Emotion decomposition (Plutchik) | yes | - | - | - | - |
| Sentiment drift detection | yes | - | - | - | - |
| Embeddable binary | yes | - | - | yes | - |
| HuggingFace vocab export | yes | - | - | - | - |
| No Python / No GPU | yes | - | - | yes | - |
| 11 semantic dimensions per word | yes | - | partial | - | - |
| C FFI (libumcs.so) | yes | - | - | - | - |
| 70 languages, 23 datasets | yes | - | partial | - | yes |
| LLM grounding endpoint | yes | - | - | - | - |
| Root-indexed embeddings | yes | - | - | - | - |
| Prefix search | yes | - | - | - | - |

---

## How it works

```
Input text
  │
  ▼ normalize() — lowercase + diacritic strip
  │  café→cafe  straße→strasse  terrível→terrivel
  │
  ▼ lexdb.LookupWord() — O(1) hash map
  │  WordRecord{ word_id, root_id, Sentiment uint32, Flags uint32 }
  │
  ▼ morpheme.Pack64() — pack into Token64 (uint64)
  │  root_id(20b) | variant(12b) | pos(3b) | concrete(1b) |
  │  scope(4b) | role(4b) | intensity(4b) | ontological(4b) |
  │  register(4b) | polarity(2b) | arousal(2b) | dominance(2b) | aoa(2b)
  │
  ▼ analyze.Analyze() — scope resolution
  │  negation window=3 tokens, intensifier 2×, downtoner 0.5×
  │
  ▼ Result{ TotalScore, Verdict, Tokens[], Emotions[], Drift[] }
```

---

## Installation

```bash
git clone https://github.com/KaioH3/umcs.git
cd umcs
go build -o lexsent ./cmd/lexsent

# Build the binary lexicon from CSV data
./lexsent build --roots data/roots.csv --words data/words.csv --out lexicon.umcs
```

**Requirements:** Go 1.24+. No external dependencies besides the standard library.

---

## Quick Start

### CLI

```bash
# Look up a word (diacritics stripped automatically)
./lexsent lookup terrível
./lexsent lookup café

# Language-specific lookup (disambiguates homographs)
./lexsent lookup mais --lang PT   # PT: mais = more
./lexsent lookup mais --lang FR   # FR: mais = but

# All cognates across languages (same root)
./lexsent cognates terrible

# Etymology chain: root → parent → proto-language
./lexsent etymo philosophy

# Scope-aware sentiment analysis
./lexsent analyze "this product is not terrible at all"
./lexsent analyze "muito bom, recomendo"

# Lexicon stats + most productive roots
./lexsent stats --productive

# Start REST API server
./lexsent serve --port 8080
```

### REST API

```bash
# Health check
curl localhost:8080/health

# Lookup with diacritic normalization
curl "localhost:8080/lookup?word=caf%C3%A9"

# Sentiment analysis with negation scope
curl -X POST localhost:8080/analyze -d "this product is not terrible at all"

# Emotion decomposition (Plutchik wheel: 8 emotions)
curl "localhost:8080/emotion?text=I+am+furious+and+terrified"

# Sentiment drift detection (trajectory patterns)
curl -X POST localhost:8080/drift -d "started great, then terrible, awful ending"

# Cross-lingual sentiment consensus
curl "localhost:8080/crosslingual?word=terrible"

# Batch analysis (up to 100 texts)
curl -X POST localhost:8080/analyze/batch \
  -H 'Content-Type: application/json' \
  -d '[{"text":"terrible service"},{"text":"serviço terrível"},{"text":"servicio terrible"}]'

# Etymology chain
curl "localhost:8080/etymo?word=terrible"

# Cognate family grouped by language
curl "localhost:8080/cognates?word=negative"

# Decode a sentiment bitmask
curl "localhost:8080/sentiment/decode?s=0x60130140"

# HuggingFace-compatible vocab export
curl localhost:8080/vocab > umcs_vocab.json

# Prefix search
curl "localhost:8080/search?q=terr&limit=5&lang=EN"

# Phonological analysis (IPA, syllables, stress)
curl "localhost:8080/phonology?word=terrible"

# Root-indexed embeddings for LLM integration
curl "localhost:8080/embeddings?limit=10"

# Ground LLM output against sentiment truth
curl -X POST localhost:8080/ground \
  -d '{"text":"this product is amazing","expected_sentiment":"POSITIVE"}'

# Root enumeration with pagination
curl "localhost:8080/roots?limit=10&offset=0&productive=true"
```

---

## The Token64 — one number, everything

A single `uint64` that encodes the **complete semantic identity** of a word. No lookup table needed.

```
 63      44 43      32 31 29 28 27  24 23  20 19  16 15  12 11   8 7 6 5 4 3 2 1 0
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ root_id (20) │ variant (12) │pos│C│scope(4)│role(4)│int(4)│onto(4)│reg(4)│P│A│D│Q│
 └────────────────────────────────────────────────────────────────────────────────┘
   └── cognate ID ──┘            └─── semantic payload (32 bits) ───────────────┘

 P = polarity (2b)   A = arousal (2b)   D = dominance (2b)   Q = AoA (2b)
 C = concrete (1b)   pos = part-of-speech (3b)
```

```go
tok := morpheme.Pack64(word.WordID, word.Sentiment, word.Flags)
// "terrible" → 0x0002800013014020
//   root=10 var=1 pos=ADJ polarity=NEGATIVE intensity=STRONG
//   role=EVALUATION arousal=HIGH dominance=LOW aoa=MID concrete=true
```

---

## Semantic Dimensions (32 bits, zero wasted)

### Sentiment bitmask (`uint32`)

| Bits | Dimension | Values |
|------|-----------|--------|
| 31..29 | **POS** | NOUN / VERB / ADJ / ADV / PARTICLE / PREP / CONJ |
| 28 | **Concreteness** | 1=concrete (chair), 0=abstract (freedom) |
| 27..24 | **Scope flags** | INTENSIFIER / DOWNTONER / NEGATION / AFFIRMATION |
| 23..20 | **Semantic role** | EVALUATION / EMOTION / COGNITION / VOLITION / CAUSATION / TEMPORAL / QUANTIFIER / CONNECTOR |
| 19..16 | **Intensity** | NONE / WEAK / MODERATE / STRONG / EXTREME |
| 15..8 | **Domain** | GENERAL / FINANCIAL / MEDICAL / LEGAL / TECHNICAL / SOCIAL / POLITICAL / ACADEMIC |
| 7..6 | **Polarity** | NEUTRAL / POSITIVE / NEGATIVE / AMBIGUOUS |
| 5..4 | **Arousal** | NONE / LOW / MED / HIGH (psycholinguistic activation) |
| 3..2 | **Dominance** | NONE / LOW / MED / HIGH (power/control — VAD model) |
| 1..0 | **AoA** | EARLY / MID / LATE / TECHNICAL (age of acquisition) |

### Flags bitmask (`uint32`)

| Bits | Dimension | Values |
|------|-----------|--------|
| 7..0 | **Lexical flags** | Proper / Archaic / Colloquial / Domain / FalseFriend / Loanword / Allomorph / Onomatopoeia |
| 11..8 | **Register** | NEUTRAL / FORMAL / INFORMAL / SLANG / VULGAR / ARCHAIC / POETIC / TECHNICAL / SCIENTIFIC |
| 15..12 | **Ontological** | PERSON / PLACE / ARTIFACT / NATURAL / EVENT / STATE / PROPERTY / QUANTITY / RELATION |
| 19..16 | **Polysemy** | Count of distinct senses (0–15) |
| 20 | **Cultural-specific** | 1 = no equivalent in most languages (saudade, schadenfreude) |
| 31..28 | **Syllable count** | 0=unknown, 1–15 |
| 27..26 | **Stress pattern** | unknown / final / penultimate / antepenultimate |
| 25..23 | **Valency** | NA / intransitive / transitive / ditransitive / copular / modal |
| 22 | **Irony-capable** | 1 = participates in ironic inversion |
| 21 | **Neologism** | 1 = coined post-1990 |

---

## Languages (24 core + 50 via datasets)

### Core languages (manually annotated)

| Code | Language | Script | Family |
|------|----------|--------|--------|
| PT | Portuguese | Latin | Romance |
| EN | English | Latin | Germanic |
| ES | Spanish | Latin | Romance |
| IT | Italian | Latin | Romance |
| DE | German | Latin | Germanic |
| FR | French | Latin | Romance |
| NL | Dutch | Latin | Germanic |
| AR | Arabic | Arabic | Semitic |
| ZH | Mandarin | CJK | Sino-Tibetan |
| JA | Japanese | CJK+Kana | Japonic |
| RU | Russian | Cyrillic | Slavic |
| KO | Korean | Hangul | Koreanic |
| TG | Tupi-Guaraní | Latin | Tupian |
| HI | Hindi | Devanagari | Indo-Aryan |
| BN | Bengali | Bengali | Indo-Aryan |
| ID | Indonesian | Latin | Austronesian |
| TR | Turkish | Latin | Turkic |
| FA | Persian | Arabic | Iranian |
| SW | Swahili | Latin | Bantu |
| UK | Ukrainian | Cyrillic | Slavic |
| PL | Polish | Latin | Slavic |
| SA | Sanskrit | Devanagari | Indo-Aryan (classical) |
| TA | Tamil | Tamil | Dravidian |
| HE | Hebrew | Hebrew | Semitic |

### Extended via CogNet / EtymWn / ML-Senticon (50+ additional)

Bulgarian, Catalan, Croatian, Czech, Danish, Estonian, Finnish, Galician, Georgian, Greek, Hungarian, Icelandic, Irish, Latvian, Lithuanian, Macedonian, Maltese, Norwegian, Romanian, Serbian, Slovak, Slovenian, Albanian, Afrikaans, Swedish, Thai, Vietnamese, Malay, Basque, Welsh, Armenian, Amharic, Tagalog, Bosnian, and more.

---

## Data Pipeline

### Curated data (hand-annotated)

| File | Records | Description |
|------|---------|-------------|
| `data/roots.csv` | 365 | Root families with etymology, meanings, semantic relations |
| `data/words.csv` | 2,442 | Words across 24 languages, fully annotated |

### Imported datasets (656K+ entries)

The `cmd/import-new` pipeline imports from 20+ external datasets, merges, deduplicates, enriches with IPA pronunciations, and appends to `data/imported_words.csv`.

#### Phase 1 — Sentiment-bearing datasets

| Dataset | Language | Records | What it provides |
|---------|----------|---------|-----------------|
| **OpLexicon v3.0** | PT | ~32K | POS + polarity (Brazilian Portuguese) |
| **SentiLex-PT02** | PT | ~7K | Sentiment lemmas (European Portuguese) |
| **MPQA** | EN | ~6.9K | Subjectivity clues (strong/weak, prior polarity) |
| **ML-Senticon** | EN/ES/CA/EU/GL | ~5K/lang | XML sentiment lexicons with continuous polarity |
| **SO-CAL** | EN/ES | ~6K | Semantic orientation [-5,+5] by POS |
| **NRC VAD Lexicon** | EN | ~54K | Valence, arousal, dominance (continuous) |
| **NRC Emotion Lexicon** | 100+ langs | ~14K | 8 Plutchik emotions per word |
| **AFINN-165** | EN | ~3.3K | Integer valence [-5,+5] |
| **VADER** | EN | ~7.5K | Mean sentiment with SD |
| **Bing Liu** | EN | ~6.8K | Binary polarity (positive/negative lists) |
| **SentiWordNet 3.0** | EN | ~117K | WordNet synset sentiment scores |
| **Warriner VAD** | EN | ~13.9K | VAD norms with standard deviations |
| **81-language sentiment** | 81 langs | varies | Multilingual polarity annotations |
| **Empath** | EN | ~194 cats | 194 semantic categories |

#### Phase 2 — Morphological / phonological datasets

| Dataset | Language | Records | What it provides |
|---------|----------|---------|-----------------|
| **Lexique383** | FR | ~143K | French phonology, syllables, frequency |
| **UniMorph** | EN/PT/ES/DE/FR | ~100K+ | Morphological inflection tables |
| **MorphoLex** | EN | ~31K | Morphological families with POS |
| **Brysbaert Concreteness** | EN | ~40K | Concreteness [1–5] + SUBTLEX frequency |
| **IPA-dict** | 10 langs | ~500K | IPA pronunciations |
| **CMUDict** | EN | ~134K | ARPABET → IPA pronunciations |

#### Phase 3 — Cross-lingual / etymological datasets

| Dataset | Languages | Records | What it provides |
|---------|-----------|---------|-----------------|
| **CogNet v2.0** | 338 langs | ~3.1M | Cognate pairs with concept IDs |
| **EtymWn** | 50+ langs | ~6M | Etymological relations (inheritance, borrowing, derivation) |

### Import pipeline

```bash
# Build the import tool
go build -o import-new ./cmd/import-new

# Run (reads from data/external/, appends to data/imported_words.csv)
UMCS_ROOT=. ./import-new
```

The pipeline:
1. Reads each dataset with format-specific parsers
2. Normalizes words (lowercase, diacritic strip, single-word filter)
3. Maps polarity/intensity to UMCS scale
4. Enriches with IPA from IPA-dict + CMUDict (fallback)
5. Merges and deduplicates by `(norm, lang)`
6. Excludes entries already in the CSV
7. Appends new entries with sequential `word_id`

---

## Architecture

```
umcs/
├── cmd/
│   ├── lexsent/          # Main CLI (12 commands)
│   └── import-new/       # Dataset import pipeline
├── pkg/
│   ├── lexdb/            # Binary .umcs format (reader, writer, header)
│   ├── morpheme/         # word_id encoding (root+variant) + Token64
│   ├── sentiment/        # 32-bit bitmask packing (11 dimensions)
│   ├── tokenizer/        # Morpheme-aware tokenization
│   ├── analyze/          # Sentiment analysis + emotion + drift
│   ├── classify/         # Logistic regression (48D features, Adam optimizer)
│   ├── api/              # REST API server (16 endpoints)
│   ├── ingest/           # 20+ dataset importers
│   ├── infer/            # Morphological inference rules
│   ├── discover/         # Wiktionary BFS discovery + XML dump parser
│   ├── propagate/        # Cross-lingual sentiment propagation
│   ├── seed/             # CSV loader (roots + words)
│   ├── phon/             # Phonological features (syllables, stress, IPA)
│   ├── ga/               # Genetic algorithm (tournament + elitism)
│   ├── rl/               # REINFORCE policy gradient (online learning)
│   ├── autoqa/           # Automated QA for text generation
│   └── capi/             # C FFI exports (libumcs.so)
├── data/
│   ├── roots.csv         # 365 root families
│   ├── words.csv         # 2,442 annotated words
│   ├── imported_words.csv # 656K+ imported entries
│   ├── staged.csv        # Pending human review
│   └── external/         # Raw dataset files
├── models/               # Trained classifier weights
└── lexicon.umcs          # Compiled binary (~200 KB)
```

### Package responsibilities

| Package | Purpose | Key types |
|---------|---------|-----------|
| `lexdb` | Binary format I/O, O(1) lookup | `Lexicon`, `WordRecord`, `RootRecord` |
| `morpheme` | Word ID bit packing | `Token64`, `MakeWordID()`, `RootOf()` |
| `sentiment` | Semantic dimension packing | `Pack()`, `Decode()`, `Weight()` |
| `tokenizer` | Text → token sequence | `MorphToken`, `Tokenize()` |
| `analyze` | Scope-aware sentiment scoring | `Result`, `EmotionProfile`, `DriftSummary` |
| `classify` | ML classification (48 features) | `Classifier`, `Train()`, `Predict()` |
| `ingest` | Dataset-specific parsers | `Entry`, `ImportOpLexicon()`, `Merge()` |
| `infer` | Suffix/prefix → POS/concreteness | `FillMissing()` |
| `discover` | Wiktionary crawler + staging | `Pipeline`, `Config` |
| `propagate` | Cognate sentiment transfer | `Propagate()` |
| `ga` | Genetic algorithm optimization | `Population`, `Evolve()` |
| `rl` | REINFORCE feedback loop | `Agent`, `Act()`, `Observe()`, `Learn()` |
| `autoqa` | Semantic validation for CI/CD | `CheckBatch()`, `OutputSpec` |
| `capi` | C shared library interface | `umcs_load()`, `umcs_analyze()` |

---

## ML Pipeline

### Classifier (48-dimensional feature extraction)

The classifier extracts 48 features per token, organized into 8 groups:

| Group | Features | Source |
|-------|----------|--------|
| **Sentiment** (0–5) | polarity, intensity, arousal, dominance, AoA, concreteness | VAD model (Warriner) |
| **Phonology** (6–10) | irony flag, neologism, syllables, stress, valency | IPA analysis |
| **Relations** (11–13) | has antonym/hypernym/synonym | Root-level links |
| **Context** (14–16) | negation/intensifier/downtoner in window | Scope resolution |
| **Semantic role** (17–27) | One-hot encoding of 11 roles | Annotation |
| **Lexical** (28–36) | POS, register, ontological, lang, frequency, polysemy, cognate count | Lexicon |
| **Phonaesthetics** (40–43) | C/V ratio, open vowels, nasals, sibilants | IPA parsing |
| **Morphology** (44–45) | Etymology depth, family size | Root chains |

Training uses Adam optimizer. Feature weights are tunable via genetic algorithm.

### Genetic Algorithm

Tournament selection (k=3) + elitism (top 8) + Gaussian mutation. Optimizes 48-dimensional weight vector against macro-averaged F1 on validation set.

### REINFORCE (online learning)

Policy gradient with baseline variance reduction (Williams 1992). Learns from user corrections in real-time:

```
Agent.Act(features) → prediction
User corrects → Agent.Observe(feedback)
Agent.Learn() → gradient update scaled by advantage
```

---

## Analysis Features

### Sentiment Analysis

Scope-aware with three modifiers:
- **Negation:** inverts polarity within 3-token window
- **Intensifier:** 2× multiplier on next sentiment token
- **Downtoner:** 0.5× multiplier on next sentiment token
- **Double negation:** cancels out (affirmation)

```bash
./lexsent analyze "not terrible"     # → POSITIVE (negation inverts NEGATIVE)
./lexsent analyze "very terrible"    # → NEGATIVE, EXTREME (intensifier)
./lexsent analyze "somewhat bad"     # → NEGATIVE, WEAK (downtoner)
```

### Emotion Decomposition (Plutchik)

Maps VAD dimensions to 8 primary emotions:

| Emotion | Valence | Arousal | Dominance |
|---------|---------|---------|-----------|
| **Joy** | high | high | high |
| **Trust** | positive | low | high |
| **Fear** | negative | high | low |
| **Anger** | negative | high | high |
| **Sadness** | negative | low | low |
| **Surprise** | any | high | any |
| **Disgust** | strong neg | high | any |
| **Serenity** | positive | low | any |

```bash
curl "localhost:8080/emotion?text=I+am+furious+and+terrified"
# → { "dominant": "ANGER", "emotions": {"anger": 0.85, "fear": 0.72, ...} }
```

### Sentiment Drift Detection

Analyzes sentiment trajectory across a text and classifies patterns:

| Pattern | Description |
|---------|-------------|
| STABLE | Consistent sentiment throughout |
| ASCENDING | Negative → Positive |
| DESCENDING | Positive → Negative |
| V-SHAPE | Positive → Negative → Positive |
| INV-V | Negative → Positive → Negative |
| VOLATILE | Rapid sentiment swings |

```bash
curl -X POST localhost:8080/drift -d "Great start, then everything went wrong, disaster"
# → { "pattern": "DESCENDING", "volatility": 0.73, "points": [...] }
```

### Cross-Lingual Consensus

Computes sentiment agreement across all languages that share a root:

```bash
curl "localhost:8080/crosslingual?word=terrible"
# → { "polarity": "NEGATIVE", "confidence": 0.95, "languages": 5 }
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `build` | Compile roots.csv + words.csv → binary .umcs |
| `lookup <word>` | Word lookup with all metadata and cognates |
| `cognates <word>` | Full morphological family across languages |
| `etymo <word>` | Etymology chain to proto-language |
| `analyze <text>` | Scope-aware sentiment analysis |
| `tokenize <text>` | Morpheme tokenization with semantic encoding |
| `stats [--productive]` | Lexicon statistics and root productivity |
| `serve [--port N]` | Launch REST API server |
| `discover` | Automated word discovery from Wiktionary |
| `import` | Batch import from Wiktionary XML dump |
| `train` | Train logistic regression classifier |
| `predict` | Run classifier predictions |
| `feedback` | Submit corrections for REINFORCE learning |
| `evolve` | Genetic algorithm weight optimization |
| `export-c` | Build C shared library (libumcs.so) |
| `stage-review` | Review staged low-confidence discoveries |

---

## API Reference (20 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status, version, checksum, counts |
| GET | `/stats` | Full lexicon statistics |
| GET | `/lookup?word=X&lang=Y` | Single word lookup |
| POST | `/lookup/batch` | Batch lookup (up to 100) |
| GET | `/cognates?word=X` | Cognate family by language |
| GET | `/etymo?word=X` | Etymology chain |
| GET/POST | `/analyze?text=...` | Sentiment analysis |
| POST | `/analyze/batch` | Batch sentiment (up to 100) |
| GET/POST | `/emotion?text=...` | Plutchik emotion decomposition |
| GET/POST | `/drift?text=...` | Sentiment trajectory + pattern |
| GET | `/crosslingual?word=X` | Cross-lingual consensus |
| GET/POST | `/tokenize?text=...` | Morpheme tokenization |
| GET | `/sentiment/decode?s=0xHEX` | Decode bitmask |
| GET | `/vocab` | HuggingFace vocab export |
| GET | `/roots?limit=N&offset=N` | Root enumeration (paginated) |
| GET | `/root/{id}[/words]` | Root metadata / word family |
| GET | `/search?q=X&limit=N&lang=Y` | Prefix search across lexicon |
| GET | `/phonology?word=X` | IPA, syllables, stress, valency |
| GET | `/embeddings?limit=N` | Root-indexed 9D semantic vectors |
| POST | `/ground` | LLM sentiment grounding + conflict detection |
| GET/POST | `/sarcasm?text=...` | Multi-language sarcasm detection |
| GET/POST | `/hate?text=...` | Hate speech detection (8 categories) |
| GET/POST | `/bias?text=...` | Political bias detection (LEFT/RIGHT/CENTER) |
| POST | `/compress` | Dataset compression (RLE, LZ77, Delta, BWT) |

### Export Formats

| Format | Endpoint | Size (typical) |
|--------|----------|----------------|
| JSON | `/vocab` | 177MB |
| JSONL | `/vocab?format=jsonl` | streaming |
| Gob | `/vocab?format=gob` | 60MB |
| Msgpack | `/vocab?format=msgpack` | 106MB |

### LLM Integration Endpoints

#### `/embeddings` — Semantic vectors for LLM grounding

Returns root-indexed 9-dimensional vectors (polarity, intensity, arousal, dominance, AoA, concreteness, POS, role, syllables). Since cognates share root_id, one vector covers all languages.

```bash
curl "localhost:8080/embeddings?limit=5"
```

#### `/ground` — Validate LLM-generated text

Checks if generated text matches expected sentiment, reports conflicting tokens, and provides actionable recommendations (PASS/WARN/FAIL).

```bash
curl -X POST localhost:8080/ground \
  -d '{"text":"I absolutely love this terrible disaster","expected_sentiment":"POSITIVE"}'
# → {"matches": false, "conflicts": [{"word":"terrible","polarity":"NEGATIVE"}], ...}
```

#### `/sarcasm` — Multi-language sarcasm detection

Detects sarcasm using linguistic patterns: laughing (haha, kkk), elongation (loooove), quotation marks, sarcastic questions, contrast. Supports 15+ languages.

```bash
curl -X POST localhost:8080/sarcasm \
  -d '{"text": "Oh wonderful, another meeting that could have been an email"}'
# → {"is_sarcastic": true, "confidence": 0.85, "patterns": ["contrast", "rhetorical_question"]}
```

#### `/hate` — Hate speech detection

Detects 8 categories: RACISM, SEXISM, HOMOPHOBIA, RELIGIOUS_HATRED, VIOLENCE, ABLEISM, GORDOPHOBIA, CAPACITISM

```bash
curl -X POST localhost:8080/hate \
  -d '{"text": "you should all be ashamed"}'
# → {"is_hate": true, "categories": ["VIOLENCE"], "confidence": 0.78}
```

#### `/bias` — Political bias detection

Detects LEFT, RIGHT, or CENTER political leaning.

```bash
curl -X POST localhost:8080/bias \
  -d '{"text": "we need to fight for workers rights and social justice"}'
# → {"bias": "LEFT", "confidence": 0.82}
```

#### `/compress` — Dataset compression

Compress text data using RLE, LZ77, Delta, or BWT algorithms. Useful for creating compact training datasets.

```bash
curl -X POST localhost:8080/compress \
  -d '{"text": "AAAABBBCCDAA", "algorithm": "rle"}'
# → {"original_size": 12, "compressed": "4A3B2C2D2A", "compressed_size": 10, "ratio": 0.83}

curl -X POST localhost:8080/compress \
  -d '{"data": [10,15,18,25,30], "algorithm": "delta"}'
# → {"original_size": 5, "compressed": [10,5,3,7,5], "compressed_size": 5, "ratio": 1.0}
```

---

## Real-World Use Cases

### Cross-lingual content moderation (no regex)

```go
wr := lex.LookupWord(word)
if wr != nil && morpheme.RootOf(wr.WordID) >= 82 && morpheme.RootOf(wr.WordID) <= 85 {
    blockContent() // vulgar root family — covers ALL 24 languages at once
}
```

### Multilingual sentiment analysis

```go
for _, review := range customerReviews {
    result := analyze.Analyze(lex, review.Text)
    // Works identically for "terrible" (EN), "terrível" (PT), "terrible" (ES)
    db.Save(Review{Score: result.TotalScore, Lang: review.Lang})
}
```

### LLM training data

```bash
curl localhost:8080/vocab > umcs_vocab.json    # HuggingFace format
# Cognates share root_id → model learns shared representations for free
```

### Customer support escalation detection

```bash
curl -X POST localhost:8080/drift \
  -d "Thank you for your patience. This is unacceptable. I want a refund now."
# → { "pattern": "DESCENDING", "volatility": 0.8 }
# Trigger escalation when pattern = DESCENDING + high volatility
```

### Readability scoring (AoA tiers)

```go
tokens := tokenizer.Tokenize(lex, text)
earlyAcq := 0
for _, t := range tokens {
    if sentiment.AOA(t.Sentiment) == sentiment.AOAEarly { earlyAcq++ }
}
// > 80% EARLY → child-accessible; < 30% → academic/technical
```

### Phonetic analysis

```go
wr := lex.LookupWord("terrível")
syllables := phon.Syllables(wr.Flags)    // 3
stress := phon.StressPattern(wr.Flags)   // penultimate
ipa := lex.Pronunciation(wr)             // /te.ˈʁi.vew/
```

### C / Python / Ruby integration

```bash
go build -buildmode=c-shared -o libumcs.so ./pkg/capi
```

```c
#include "libumcs.h"
umcs_load("lexicon.umcs");
UmcsAnalysisResult r;
umcs_analyze("terrible service", &r);
printf("score: %f, verdict: %s\n", r.score, r.verdict);
```

---

## Data Model

### word_id encoding

```
bits 31..12 = root_id  (20 bits → up to 1,048,575 root families)
bits 11..0  = variant  (12 bits → up to 4,095 variants per root)

word_id = (root_id << 12) | variant

root "negat" → root_id=1
  "negative"  (EN, v1) → word_id = (1<<12)|1 = 4097
  "negativo"  (PT, v2) → word_id = (1<<12)|2 = 4098
  "negación"  (ES, v4) → word_id = (1<<12)|4 = 4100

morpheme.RootOf(4098) == morpheme.RootOf(4097) → true (same family)
```

### .umcs binary format

```
Offset  Size     Field
0       64       Header (magic=0x4C534442, version, counts, offsets, FNV-1a checksum)
64      N×44     Root table (sorted by root_id)
H       M×36     Word table (sorted by word_id)
W       variable String heap (null-terminated UTF-8)
```

All integers: little-endian uint32. Backward-compatible v1→v2.

---

## Morphological Inference

The `pkg/infer` package auto-fills missing annotations from suffix/prefix patterns:

```
"-ção" / "-tion" / "-keit" / "-té"    → NOUN + ABSTRACT
"-mente" / "-ly" / "-ment"            → ADV
"-oso" / "-ful" / "-lich"             → ADJ
"-ar" / "-er" / "-are"                → VERB
"ex-"                                 → STRONG intensity
"sub-"                                → WEAK intensity
"ir-" / "un-" / "in-"                → NEGATION marker
```

Applied during `lexsent build` via `infer.FillMissing()` — fills empty fields without overwriting explicit annotations. Annotate hundreds manually, infer thousands from patterns.

---

## Tests

```bash
go test ./... -race          # 18 packages, all passing
go test ./pkg/classify -bench .  # Benchmarks vs VADER/DistilBERT/RoBERTa
```

Test types:
- **Unit tests** — all packages
- **Stress tests** — large text analysis, tokenization throughput
- **Property-based tests** — classifier invariants
- **Race condition tests** — concurrency safety
- **Benchmark comparisons** — vs VADER, DistilBERT, RoBERTa

---

## Lexicon Stats

| Metric | Value |
|--------|-------|
| Curated root families | 365 |
| Curated words | 2,442 |
| Total words (with imports) | **4,370,202** |
| Total roots (with synthetic) | **76,622** |
| Languages supported | **70** |
| Entries with IPA | 933,740 |
| External datasets | 23 |
| Theoretical capacity | ~1M roots × 4K variants = **4B word_ids** |
| Extended binary size | ~216 MB |
| Semantic dimensions | 11 per word |
| API endpoints | **20** |
| CLI commands | 16 |
| Feature dimensions (classifier) | 48 |

See [DATASETS.md](DATASETS.md) for dataset details and [BUILDING.md](BUILDING.md) for reproduction steps.

---

## Roadmap

- [ ] **Token128** — extend to 128-bit for lossless encoding including domain flags
- [ ] **ML-weighted propagation** — confidence-scored cross-lingual sentiment transfer
- [ ] **Morphological parser** — decompose `impossível` → `im-` (negation) + `poss` (root) + `-ível` (suffix)
- [ ] **Translation bridge** — `/translate?word=terrible&from=EN&to=PT` via root_id
- [ ] **Readability API** — `/readability?text=...` using AoA tiers
- [ ] **Valency annotation** — syntactic argument structure (TRANSITIVE/INTRANSITIVE/DITRANSITIVE)
- [ ] **Gender annotation** — grammatical gender per word (MASC/FEM/NEUTER/COMMON)
- [ ] **Embedding export** — root_id-indexed embeddings (SentenceTransformers compatible)
- [ ] **Ontology depth** — multi-level hierarchy (BIOLOGICAL > ANIMAL > MAMMAL)
- [ ] **False friends database** — cross-lingual divergence tracking via FalseFriend flag
- [ ] **WebAssembly build** — run UMCS in the browser

---

## License

MIT — see [LICENSE](LICENSE).
