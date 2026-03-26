# UMCS — Universal Morpheme Coordinate System

> One number carries the entire semantic identity of a word — across every human language.

[![Go](https://img.shields.io/badge/Go-1.24-blue?logo=go)](https://go.dev)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What is UMCS?

UMCS assigns every word in every language a **deterministic coordinate** based on its etymological root. Words that share the same ancestor share the same coordinate prefix — regardless of language, script, or script family.

```
"terrible" (EN) → word_id = 40961   (root_id=10, variant=1)
"terrível"  (PT) → word_id = 40962   (root_id=10, variant=2)
"terrible"  (FR) → word_id = 40963   (root_id=10, variant=3)
"terribile" (IT) → word_id = 40964   (root_id=10, variant=4)
"terrible"  (ES) → word_id = 40965   (root_id=10, variant=5)
```

Same root (`terr`, Latin *terror*) = same embedding slot in any model. Cognates across PT/EN/ES/FR/IT/DE/NL/AR/ZH/JA/RU/KO/TG/HI/SA and more are linked by this ID automatically.

---

## Why not VADER / spaCy / regex lists?

| Feature | UMCS | VADER | spaCy | Regex |
|---------|------|-------|-------|-------|
| Cross-lingual single lookup | ✓ | ✗ | ✗ | ✗ |
| O(1) lookup, no model load | ✓ | ✗ | ✗ | ✓ |
| Scope-aware sentiment (negation, intensifiers) | ✓ | ✓ | ✓ | ✗ |
| Etymology chains | ✓ | ✗ | ✗ | ✗ |
| Embeddable binary (<5 MB) | ✓ | ✗ | ✗ | ✓ |
| HuggingFace vocab export | ✓ | ✗ | ✗ | ✗ |
| No Python runtime | ✓ | ✗ | ✗ | ✓ |
| Part-of-speech, arousal, dominance, AoA | ✓ | ✗ | partial | ✗ |
| VAD psychological model (Valence-Arousal-Dominance) | ✓ | ✗ | ✗ | ✗ |
| Tupi-Guaraní, Sanskrit, Arabic, Hindi | ✓ | ✗ | partial | ✗ |

---

## How it works

```
Input text
  │
  ▼ Normalize() — lowercase + diacritic strip
  │  café→cafe  straße→strasse  terrível→terrivel  Tupã→tupa
  │
  ▼ lexdb.LookupWord() — O(1) hash map
  │  WordRecord{ word_id, root_id, Sentiment uint32, Flags uint32 }
  │
  ▼ morpheme.Pack64() — pack into Token64
  │  uint64: root_id(20b) | variant(12b) | pos(3b) | concrete(1b) |
  │          scope(4b) | role(4b) | intensity(4b) | ontological(4b) |
  │          register(4b) | polarity(2b) | arousal(2b) | dominance(2b) | aoa(2b)
  │
  ▼ analyze.Analyze() — scope resolution
     negation window=3, intensifier 2×, downtoner 0.5×
     → Result{ TotalScore, Verdict, Tokens[] }
```

---

## The Token64 — one number, everything

The crown jewel of UMCS: a single `uint64` that encodes the **complete semantic identity** of a word. A language model that receives this token can decode every dimension without any lookup table.

```
 63      44 43      32 31 29 28 27  24 23  20 19  16 15  12 11   8 7 6 5 4 3 2 1 0
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ root_id (20) │ variant (12) │pos│C│scope(4)│role(4)│int(4)│onto(4)│reg(4)│P│A│D│Q│
 └────────────────────────────────────────────────────────────────────────────────┘
   └── cognate ID ──┘            └─── semantic payload (32 bits) ───────────────┘

 P = polarity (2b)   A = arousal (2b)   D = dominance (2b)   Q = AoA (2b)
 C = concrete (1b)   pos = part-of-speech (3b)
```

Example for `"terrible"` (EN, NEGATIVE STRONG EVALUATION, ADJ, HIGH arousal, LOW dominance):
```go
tok := morpheme.Pack64(word.WordID, word.Sentiment, word.Flags)
// → 0x0002800013014020
//   root=10 var=1 pos=ADJ polarity=NEGATIVE intensity=STRONG
//   role=EVALUATION arousal=HIGH dominance=LOW aoa=MID concrete=true
```

---

## Semantic dimensions packed per word

### In `sentiment` uint32 (complete — no bits wasted):

| Bits | Dimension | Values |
|------|-----------|--------|
| 31..29 | **POS** | NOUN/VERB/ADJ/ADV/PARTICLE/PREP/CONJ |
| 28 | **Concreteness** | 1=concrete (chair), 0=abstract (freedom) |
| 27..24 | **Scope flags** | INTENSIFIER / DOWNTONER / NEGATION / AFFIRMATION |
| 23..20 | **Semantic role** | EVALUATION/EMOTION/COGNITION/VOLITION/CAUSATION/TEMPORAL/QUANTIFIER/CONNECTOR/NEGATION_MARKER/INTENSIFIER/DOWNTONER |
| 19..16 | **Intensity** | NONE / WEAK / MODERATE / STRONG / EXTREME |
| 15..8 | **Domain** | GENERAL/FINANCIAL/MEDICAL/LEGAL/TECHNICAL/SOCIAL/POLITICAL/ACADEMIC |
| 7..6 | **Polarity** | NEUTRAL / POSITIVE / NEGATIVE / AMBIGUOUS |
| 5..4 | **Arousal** | NONE/LOW/MED/HIGH — psycholinguistic activation axis |
| 3..2 | **Dominance** | NONE/LOW/MED/HIGH — power/control axis (VAD model) |
| 1..0 | **AoA** | EARLY/MID/LATE/TECHNICAL — age of acquisition |

### In `flags` uint32:

| Bits | Dimension | Values |
|------|-----------|--------|
| 7..0 | **Lexical flags** | Proper/Archaic/Colloquial/Domain/FalseFriend/Loanword/Allomorph/Onomatopoeia |
| 11..8 | **Register** | NEUTRAL/FORMAL/INFORMAL/SLANG/VULGAR/ARCHAIC/POETIC/TECHNICAL/SCIENTIFIC/CHILD/REGIONAL |
| 15..12 | **Ontological** | NONE/PERSON/PLACE/ARTIFACT/NATURAL/EVENT/STATE/PROPERTY/QUANTITY/RELATION/TEMPORAL/BIOLOGICAL/SOCIAL/ABSTRACT |
| 19..16 | **Polysemy** | Count of distinct senses (0=unknown, max 15) |
| 20 | **Cultural-specific** | 1 = no equivalent in most languages (saudade, schadenfreude) |

---

## Languages supported (24)

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

The binary format reserves 32 language slots — 8 remain free for future additions.

---

## Lexicon stats (current)

| Metric | Value |
|--------|-------|
| Root families | 121 |
| Word entries | 1,396 |
| Languages | 24 |
| Theoretical capacity | ~1M roots × 4K variants = **4 billion word_ids** |
| Binary size (lexicon.umcs) | ~0.2 MB |

---

## Installation

```bash
# Build from source (requires Go 1.24+)
git clone https://github.com/KaioH3/umcs.git
cd umcs
go build -o lexsent ./cmd/lexsent

# Build the binary lexicon from CSV data
./lexsent build --roots data/roots.csv --words data/words.csv --out lexicon.umcs
```

---

## Quick Start

### CLI

```bash
# Look up a word (diacritics stripped automatically)
./lexsent lookup terrível
./lexsent lookup café

# Language-specific lookup (disambiguates homographs)
./lexsent lookup mais --lang PT   # PT: mais=more
./lexsent lookup mais --lang FR   # FR: mais=but

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
# Health check (version, checksum, stats)
curl localhost:8080/health

# Lookup with diacritic normalization
curl "localhost:8080/lookup?word=caf%C3%A9"
curl "localhost:8080/lookup?word=mais&lang=PT"

# Sentiment analysis with negation scope
curl -X POST localhost:8080/analyze \
  -d "this product is not terrible at all"

# Cross-lingual batch analysis
curl -X POST localhost:8080/analyze/batch \
  -H 'Content-Type: application/json' \
  -d '[{"text":"terrible service"},{"text":"serviço terrível"},{"text":"servicio terrible"}]'

# Etymology chain
curl "localhost:8080/etymo?word=terrible"

# Cognate family grouped by language
curl "localhost:8080/cognates?word=negative"

# Decode a sentiment bitmask
curl "localhost:8080/sentiment/decode?s=0x60130140"

# Export HuggingFace-compatible vocab (word → word_id map)
curl localhost:8080/vocab > umcs_vocab.json

# Batch content moderation
curl -X POST localhost:8080/lookup/batch \
  -H 'Content-Type: application/json' \
  -d '[{"word":"hate","lang":"EN"},{"word":"ódio","lang":"PT"},{"word":"haine","lang":"FR"}]'

# Root list with pagination
curl "localhost:8080/roots?limit=10&offset=0&productive=true"
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `build` | Compile roots.csv + words.csv → binary .umcs |
| `lookup <word>` | Word lookup with cognates and etymology |
| `cognates <word>` | Full morphological family across languages |
| `etymo <word>` | Etymology chain to proto-language |
| `analyze <text>` | Scope-aware sentiment analysis |
| `tokenize <text>` | Morpheme tokenization with semantic encoding |
| `stats` | Lexicon statistics |
| `serve` | Launch REST API server |
| `discover` | Automated word discovery from Wiktionary (live API) |
| `import` | Batch import from Wiktionary XML dump |

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status, version, checksum, root/word counts |
| GET | `/stats` | Full lexicon statistics |
| GET | `/lookup?word=X&lang=EN` | Single word lookup |
| POST | `/lookup/batch` | Batch lookup (up to 100 words) |
| GET | `/cognates?word=X` | Cognate family by language |
| GET | `/etymo?word=X` | Etymology chain |
| GET/POST | `/analyze` | Sentiment analysis |
| POST | `/analyze/batch` | Batch sentiment (up to 100 texts) |
| GET/POST | `/tokenize` | Morpheme tokenization |
| GET | `/vocab` | HuggingFace vocab JSON |
| GET | `/roots` | Root enumeration (paginated) |
| GET | `/root/{id}` | Root metadata |
| GET | `/root/{id}/words` | Root's word family by language |
| GET | `/sentiment/decode?s=0xHEX` | Decode sentiment bitmask |

---

## Real Use Cases

### Content moderation (cross-lingual, no regex)

```go
// Works for EN, PT, ES, FR, IT simultaneously — same root_id regardless of language
wr := lex.LookupWord(word)
if wr != nil && morpheme.RootOf(wr.WordID) >= 82 && morpheme.RootOf(wr.WordID) <= 85 {
    blockContent() // vulgar root family — universal filter
}
```

No per-language regex lists. No translation step. One check covers all languages.

### Multilingual sentiment analysis

```go
// "terrible service" (EN), "serviço terrível" (PT), "servicio terrible" (ES)
// → all return root_id=10 (terr), same sentiment: NEGATIVE STRONG
for _, review := range customerReviews {
    result := analyze.Analyze(lex, review.Text)
    db.Save(Review{Score: result.TotalScore, Lang: review.Lang})
}
```

### LLM training data preparation

```bash
# Export word→word_id map (HuggingFace tokenizer format)
curl localhost:8080/vocab > umcs_vocab.json

# Generate Token64 values for training corpus
lexsent tokenize "the product quality was not terrible" --format token64
# → 0x0002800013014020 0x... 0x... (one uint64 per token, fully self-describing)
```

Cognates across languages share the same root_id — the model learns shared representations for free without any alignment step.

### Etymology research

```bash
# Trace "philosophy" back to Proto-Indo-European
curl "localhost:8080/etymo?word=philosophy"
# → phil (Greek, "love") → PIE root *bʰiléh₂

# Find all words sharing the "phil" root across languages
curl "localhost:8080/cognates?word=philosophy"
# → filosofia (PT/ES/IT), philosophie (FR/DE), philosopher (EN), ...
```

### Readability scoring (using AoA tiers)

```go
tokens := lex.Tokenize(text)
early := 0
for _, t := range tokens {
    if sentiment.AOA(t.Sentiment) == sentiment.AOAEarly {
        early++
    }
}
readability := float64(early) / float64(len(tokens))
// > 0.8 → child-accessible text; < 0.3 → academic/technical text
```

---

## Data Model

### word_id encoding

```
bits 31..12 = root_id  (20 bits → up to 1,048,575 root families)
bits 11..0  = variant  (12 bits → up to 4,095 variants per root)

word_id = (root_id << 12) | variant

Examples:
  root "negat" → root_id=1
  "negative"  (EN, v1) → word_id = (1<<12)|1 = 4097
  "negativo"  (PT, v2) → word_id = (1<<12)|2 = 4098
  "negación"  (ES, v4) → word_id = (1<<12)|4 = 4100

  Morpheme.RootOf(4098) == Morpheme.RootOf(4097) → true (same family)
```

### .umcs binary format

```
Offset  Size  Field
0       64    Header (magic=0x4C534442, version, counts, offsets, checksum)
64      N×32  Root table (sorted by root_id)
H       M×32  Word table (sorted by word_id)
W       var   String heap (null-terminated UTF-8)
```

All integers are little-endian uint32. The checksum (FNV-1a) covers all data after the header.

---

## The `pkg/infer` package — morphological inference

Words that share morphological patterns share semantic properties. The `infer` package auto-fills missing annotation fields from suffix patterns:

```
"-ção" / "-tion" / "-keit" / "-té"  → NOUN + ABSTRACT
"-mente" / "-ly" / "-ment"          → ADV
"-oso" / "-ful" / "-lich"           → ADJ
"-ar" / "-er" / "-are"              → VERB
```

Applied during `lexsent build`, `infer.FillMissing()` fills in dimensions that the annotator left blank without overwriting explicit annotations. This is how UMCS scales: annotate a few hundred words manually, let morphological rules infer dimensions for thousands more.

---

## Expanding the lexicon

### CSV format

**roots.csv**:
```
root_id,root_str,origin,meaning_en,notes,parent_root_id
10,terr,LATIN,fear or dread,from terror,0
32,terrib,LATIN,terrible or causing terror,derived from terrere,10
```

**words.csv** (new extended format):
```
word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,
freq_rank,flags,pos,arousal,dominance,aoa,concreteness,register,ontological,polysemy
40961,10,1,terrible,EN,terrible,NEGATIVE,STRONG,EVALUATION,GENERAL,
800,0,ADJ,HIGH,LOW,MID,CONCRETE,NEUTRAL,PROPERTY,2
```

New columns (all optional — existing rows default to 0/NONE):
- `pos`: NOUN/VERB/ADJ/ADV/PARTICLE/PREP/CONJ
- `arousal`: NONE/LOW/MED/HIGH
- `dominance`: NONE/LOW/MED/HIGH
- `aoa`: EARLY/MID/LATE/TECHNICAL
- `concreteness`: CONCRETE or empty (abstract)
- `register`: NEUTRAL/FORMAL/INFORMAL/SLANG/VULGAR/ARCHAIC/POETIC/TECHNICAL/SCIENTIFIC/CHILD/REGIONAL
- `ontological`: NONE/PERSON/PLACE/ARTIFACT/NATURAL/EVENT/STATE/PROPERTY/QUANTITY/RELATION/TEMPORAL/BIOLOGICAL/SOCIAL/ABSTRACT
- `polysemy`: integer count of distinct senses

### Automated discovery

```bash
# Discover new words from Wiktionary (live API)
./lexsent discover --lang PT,EN,ES --depth 2 --limit 500

# Import from Wiktionary XML dump (offline)
./lexsent import --dump dumps/enwiktionary.xml.bz2 --lang EN --limit 10000
```

---

## Roadmap

- [ ] **Token128**: extend to 128-bit for lossless encoding of all dimensions including domain
- [ ] **Genetic algorithm**: GA-based optimization of sentiment analysis parameters (negation window size, intensifier multiplier) against labeled corpora
- [ ] **Semi-supervised propagation**: extend `pkg/propagate` with ML-weighted confidence scores
- [ ] **Morphological parser**: decompose "impossível" → im-(negation) + poss(root) + -ível(suffix)
- [ ] **Cross-lingual bridge**: `/translate?word=terrible&from=EN&to=PT` via root_id lookup
- [ ] **Readability API**: `/readability?text=...` using AoA tiers
- [ ] **Valency annotation**: syntactic argument structure for verbs (TRANSITIVE/INTRANSITIVE/DITRANSITIVE)
- [ ] **Gender annotation**: grammatical gender per word (MASC/FEM/NEUTER/COMMON)
- [ ] **Embedding export**: export root_id-indexed embeddings compatible with SentenceTransformers
- [ ] **More languages**: completing annotation for TG/HI/BN/ID/TR/FA/SW/UK/PL/SA/TA/HE
- [ ] **Ontology depth**: multi-level ontological hierarchy (BIOLOGICAL > ANIMAL > MAMMAL > PRIMATE)
- [ ] **False friends database**: cross-lingual false cognate detection via FalseFriend flag

---

## License

MIT — see [LICENSE](LICENSE).
