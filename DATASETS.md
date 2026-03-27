# UMCS Datasets Reference

This document lists every external dataset integrated into the UMCS import pipeline,
with source URLs, licenses, formats, and what each provides.

All datasets are stored in `data/external/` and processed by `cmd/import-new/main.go`.

---

## Phase 1 — Sentiment Lexicons

### NRC VAD Lexicon v2.1
- **Source:** https://saifmohammad.com/WebPages/nrc-vad.html
- **License:** Non-commercial research use
- **Format:** TSV (`word`, `valence`, `arousal`, `dominance`)
- **Records:** ~20,007 English words
- **Provides:** Continuous VAD scores [0,1] mapped to UMCS arousal/dominance tiers
- **File:** `data/external/NRC-VAD-Lexicon.txt`

### NRC Emotion Lexicon (EmoLex)
- **Source:** https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
- **License:** Non-commercial research use
- **Format:** TSV (`word`, `emotion`, `association`)
- **Records:** ~14,182 words x 100+ languages, 8 Plutchik emotions
- **Provides:** Binary emotion associations per word
- **File:** `data/external/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`

### AFINN-165
- **Source:** https://github.com/fnielsen/afinn
- **License:** ODbL (Open Database License)
- **Format:** TSV (`word`, `valence`)
- **Records:** ~3,382 English words
- **Provides:** Integer valence [-5, +5]
- **File:** `data/external/AFINN-165.txt`

### VADER Sentiment Lexicon
- **Source:** https://github.com/cjhutto/vaderSentiment
- **License:** MIT
- **Format:** TSV (`word`, `mean_sentiment`, `sd`, `ratings`)
- **Records:** ~7,504 entries
- **Provides:** Mean sentiment [-4, +4] with standard deviation
- **File:** `data/external/vader_lexicon.txt`

### Bing Liu Opinion Lexicon
- **Source:** https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
- **License:** Research use
- **Format:** Two text files (positive-words.txt, negative-words.txt)
- **Records:** ~6,789 words (2,006 positive + 4,783 negative)
- **Provides:** Binary polarity
- **File:** `data/external/opinion-lexicon-English/`

### SentiWordNet 3.0
- **Source:** https://github.com/aesuli/SentiWordNet
- **License:** CC BY-SA 4.0
- **Format:** TSV (`POS`, `ID`, `PosScore`, `NegScore`, `SynsetTerms`, `Gloss`)
- **Records:** ~117,659 synsets
- **Provides:** Positive/negative scores per WordNet synset
- **File:** `data/external/SentiWordNet_3.0.0.txt`

### Warriner VAD Norms
- **Source:** https://link.springer.com/article/10.3758/s13428-012-0314-x
- **License:** Research use (Creative Commons)
- **Format:** CSV with VAD means and standard deviations
- **Records:** ~13,915 English words
- **Provides:** Valence, arousal, dominance means + SDs
- **File:** `data/external/Warriner_VAD.csv`

### 81-Language Sentiment Database
- **Source:** https://github.com/Ukraine-Sentiment/sentiment-81langs
- **License:** Research use
- **Format:** CSV per language (`word`, `polarity`)
- **Records:** Variable per language
- **Provides:** Binary/ternary polarity in 81 languages
- **File:** `data/external/sentiment-81langs/`

### Empath
- **Source:** https://github.com/Ejhfast/empath-client
- **License:** MIT
- **Format:** CSV (`category`, `word`)
- **Records:** 194 semantic categories
- **Provides:** Semantic category membership
- **File:** `data/external/empath_categories.csv`

### OpLexicon v3.0
- **Source:** https://github.com/opinando/OpLexicon
- **License:** Creative Commons
- **Format:** TSV (`word`, `POS`, `polarity`)
- **Records:** ~32,191 Brazilian Portuguese words
- **Provides:** POS tags + ternary polarity
- **File:** `data/external/OpLexicon-v3.0/lexico_v3.0.txt`

### SentiLex-PT02
- **Source:** https://b2share.eudat.eu/records/93ab120efdaa4662baec6adee8e7585f
- **License:** Research use
- **Format:** Custom (`lemma.POS=polarity;targets`)
- **Records:** ~7,014 Portuguese lemmas
- **Provides:** Polarity per lemma with target annotations
- **File:** `data/external/SentiLex-PT02/SentiLex-lem-PT02.txt`

### MPQA Subjectivity Lexicon
- **Source:** https://mpqa.cs.pitt.edu/lexicons/subj_lexicon/
- **License:** Research use (GNU GPL)
- **Format:** Custom fields (`type`, `len`, `word1`, `pos1`, `stemmed1`, `priorpolarity`)
- **Records:** ~6,886 English entries
- **Provides:** Subjectivity type (strong/weak) + prior polarity
- **File:** `data/external/subjclueslen1-HLTEMNLP05.tff`

### ML-Senticon
- **Source:** https://github.com/lmartinez-ulr/ML-SentiCon
- **License:** Research use
- **Format:** XML (`synset`, `word`, `polarity_score`)
- **Records:** ~5,000 per language (EN, ES, CA, EU, GL)
- **Provides:** Continuous polarity [-1, +1] with confidence
- **File:** `data/external/ML-Senticon/`

### SO-CAL (Semantic Orientation Calculator)
- **Source:** https://github.com/sfu-discourse-lab/SO-CAL
- **License:** Research use
- **Format:** TSV dictionaries by POS (adj, adv, noun, verb, intensifiers)
- **Records:** ~6,000 EN entries + Spanish equivalents
- **Provides:** Semantic orientation [-5, +5] with POS classification
- **File:** `data/external/SO-CAL/Resources/dictionaries/`

---

## Phase 2 — Morphological / Phonological Datasets

### Lexique383
- **Source:** http://www.lexique.org/
- **License:** CC BY-SA 4.0
- **Format:** TSV (40+ columns: ortho, phon, lemme, cgram, freqlivres, ...)
- **Records:** ~142,694 French entries
- **Provides:** Phonology, syllable count, frequency, lemma, grammatical category
- **File:** `data/external/Lexique383.tsv`

### UniMorph
- **Source:** https://unimorph.github.io/
- **License:** CC BY-SA 4.0
- **Format:** TSV per language (`lemma`, `form`, `features`)
- **Records:** ~100K+ across 5 languages (EN, PT, ES, DE, FR)
- **Provides:** Morphological inflection tables with feature tags
- **Files:** `data/external/unimorph-eng.tsv`, `unimorph-por.tsv`, `unimorph-spa.tsv`, `unimorph-deu.tsv`, `unimorph-fra.tsv`

### MorphoLex
- **Source:** https://github.com/hugomailhot/MorphoLex-en
- **License:** CC BY 4.0
- **Format:** TSV (`Word`, `POS`, `MorphoLexSegm`, `MorphemeCount`, ...)
- **Records:** ~31,000 English entries
- **Provides:** Morphological segmentation, family size, POS
- **File:** `data/external/morpholex-en.tsv`

### Brysbaert Concreteness
- **Source:** https://link.springer.com/article/10.3758/s13428-013-0403-5
- **License:** Research use (supplementary materials)
- **Format:** TSV (`Word`, `Conc.M`, `Conc.SD`, `SUBTLEX`, `Total`)
- **Records:** ~39,954 English words
- **Provides:** Concreteness mean [1–5], standard deviation, SUBTLEX frequency
- **File:** `data/external/brysbaert-concreteness.tsv`

### IPA-dict
- **Source:** https://github.com/open-dict-data/ipa-dict
- **License:** MIT
- **Format:** TSV per language (`word`, `IPA`)
- **Records:** ~500K+ across 10 languages
- **Provides:** IPA pronunciation transcriptions
- **File:** `data/external/ipa-dict/`

### CMU Pronouncing Dictionary
- **Source:** http://www.speech.cs.cmu.edu/cgi-bin/cmudict
- **License:** BSD-like
- **Format:** Custom (`WORD  AH0 B AW1 T`)
- **Records:** ~134,373 English entries (ARPABET notation)
- **Provides:** Phonemic transcription, converted to IPA during import
- **File:** `data/external/cmudict-0.7b.txt`

---

## Phase 3 — Cross-Lingual / Etymological Datasets

### CogNet v2.0
- **Source:** https://github.com/GT-SALT/CogNet
- **License:** CC BY-NC-SA 4.0
- **Format:** TSV (`lang1`, `word1`, `lang2`, `word2`, `concept_id`)
- **Records:** ~3.1M cognate pairs across 338 languages
- **Provides:** Cross-lingual cognate relationships with concept alignment
- **File:** `data/external/CogNet/CogNet-v2.0.tsv`

### Etymological WordNet (EtymWn)
- **Source:** http://www1.icsi.berkeley.edu/~demelo/etymwn/
- **License:** CC BY-SA 3.0
- **Format:** TSV (`lang1:word1`, `relation`, `lang2:word2`)
- **Records:** ~6M etymological relations
- **Provides:** Etymology (inheritance, borrowing, derivation) across 50+ languages
- **Relations:** `rel:etymology`, `rel:etymological_origin_of`, `rel:has_derived_form`, `rel:is_derived_from`
- **File:** `data/external/etymwn/etymwn.tsv`

---

## Download Instructions

```bash
# Create data/external directory
mkdir -p data/external

# Most datasets must be downloaded manually due to license agreements.
# Place each file in the path listed above.

# After downloading all datasets:
go build -o import-new ./cmd/import-new
UMCS_ROOT=$PWD ./import-new

# This generates data/imported_words.csv with all entries merged and deduplicated.
# Expected output: ~4.3M entries, 68 languages, 933K IPA pronunciations.
```

## Column Mapping

| External field | UMCS column | Transform |
|---------------|-------------|-----------|
| valence [0,1] | polarity | >0.6 → POSITIVE, <0.4 → NEGATIVE |
| arousal [0,1] | arousal | >0.66 → HIGH, >0.33 → MED, else LOW |
| dominance [0,1] | dominance | >0.66 → HIGH, >0.33 → MED, else LOW |
| polarity [-5,+5] | polarity+intensity | sign → polarity, abs → intensity tier |
| POS tag | pos | Mapped to NOUN/VERB/ADJ/ADV |
| IPA string | pron | Stored as-is in string heap |
| concreteness [1,5] | concreteness | >4.0 → concrete (bit 28) |
| morpheme count | polysemy | Direct mapping |
