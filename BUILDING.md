# Building UMCS from Source

Step-by-step guide to reproduce the complete UMCS lexicon from external datasets.

## Prerequisites

- Go 1.24+
- ~500MB disk for datasets in `data/external/`
- ~350MB disk for generated `lexicon.umcs`

## Quick Build (curated words only)

```bash
go build -o lexsent ./cmd/lexsent
./lexsent build --roots data/roots.csv --words data/words.csv --out lexicon.umcs
```

**Expected output:** ~134KB binary with 365 roots, 2,442 words, 24 languages.

## Full Build (with 4.3M+ imported words)

### Step 1: Download External Datasets

See [DATASETS.md](DATASETS.md) for URLs and licenses. Place files in `data/external/`:

```
data/external/
  NRC-VAD-Lexicon.txt
  NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
  AFINN-165.txt
  vader_lexicon.txt
  opinion-lexicon-English/
  SentiWordNet_3.0.0.txt
  Warriner_VAD.csv
  sentiment-81langs/
  empath_categories.csv
  OpLexicon-v3.0/lexico_v3.0.txt
  SentiLex-PT02/SentiLex-lem-PT02.txt
  subjclueslen1-HLTEMNLP05.tff
  ML-Senticon/
  SO-CAL/Resources/dictionaries/
  Lexique383.tsv
  unimorph-eng.tsv, unimorph-por.tsv, unimorph-spa.tsv, unimorph-deu.tsv, unimorph-fra.tsv
  morpholex-en.tsv
  brysbaert-concreteness.tsv
  ipa-dict/
  cmudict-0.7b.txt
  CogNet/CogNet-v2.0.tsv
  etymwn/etymwn.tsv
```

### Step 2: Run Import Pipeline

```bash
go build -o import-new ./cmd/import-new
UMCS_ROOT=$PWD ./import-new
```

**Expected output:**
- `data/imported_words.csv` (~320MB)
- ~4.3M entries from 23 datasets
- ~933K entries with IPA pronunciation
- 68 unique languages
- Source column tracking provenance per entry

### Step 3: Build Extended Lexicon

```bash
go build -o lexsent ./cmd/lexsent
./lexsent build \
  --roots data/roots.csv \
  --words data/words.csv \
  --imported data/imported_words.csv \
  --out lexicon.umcs
```

**Expected output:**
```
Loading imported words from data/imported_words.csv...
  Imported: ~76K synthetic roots, ~4.3M words
  Excluded: ~1.5K duplicates already in curated words
Building lexicon (~76K roots, ~4.3M words)...
Built lexicon.umcs
  Roots:  ~76,622
  Words:  ~4,370,202
  Langs:  PT EN ES IT DE FR NL AR ZH JA RU KO ... +32 more
  Size:   ~216 MB
```

### Step 4: Verify

```bash
# Run all tests
go test ./... -race -count=1

# Test lookups (curated + imported words)
./lexsent lookup terrible        # curated (EN)
./lexsent lookup Mobiltelefon    # imported via CogNet (DE)
./lexsent lookup philosophie     # imported via EtymWn (FR)
./lexsent lookup saudade         # curated (PT)

# Start API and test new endpoints
./lexsent serve --port 8080 &
curl "localhost:8080/health"
curl "localhost:8080/search?q=terr&limit=5"
curl "localhost:8080/phonology?word=terrible"
curl "localhost:8080/embeddings?limit=10"
curl -X POST localhost:8080/ground -d '{"text":"this is terrible","expected_sentiment":"NEGATIVE"}'
```

## Build Variants

### C Shared Library

```bash
go build -buildmode=c-shared -o libumcs.so ./pkg/capi
# Produces: libumcs.so + libumcs.h
```

### Without Imported Words

If you don't have the external datasets, the standard build works with just
the curated data (365 roots, 2,442 words):

```bash
./lexsent build  # uses defaults: data/roots.csv + data/words.csv
```

## Verification Checksums

After a full build you should see approximately:

| Metric | Expected |
|--------|----------|
| imported_words.csv lines | ~4,370,000 |
| imported_words.csv size | ~320 MB |
| lexicon.umcs size | ~216 MB |
| Total roots | ~76,600 |
| Total words | ~4,370,000 |
| Languages | 64 (with bitmask) + more |
| Tests passing | 18 packages |
