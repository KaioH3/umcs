#!/usr/bin/env python3
"""
UMCS Benchmark vs VADER and TextBlob
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import json
import subprocess
import sys

vader = SentimentIntensityAnalyzer()

test_cases = [
    # English
    ("this is wonderful", "POSITIVE"),
    ("this is terrible", "NEGATIVE"),
    ("i love you", "POSITIVE"),
    ("i hate you", "NEGATIVE"),
    ("this is amazing", "POSITIVE"),
    ("this is awful", "NEGATIVE"),
    ("not terrible", "POSITIVE"),  # Negation test
    ("very wonderful", "POSITIVE"),  # Intensifier test
    ("this is okay", "NEUTRAL"),
    ("terrible horrible disgusting", "NEGATIVE"),
    
    # Portuguese (VADER/TextBlob won't work well)
    ("este filme é maravilhoso", "POSITIVE"),
    ("isso é horrível", "NEGATIVE"),
    ("eu te amo", "POSITIVE"),
    ("feliz aniversário", "POSITIVE"),
]

def umcs_analyze(text):
    try:
        result = subprocess.run(
            ["./lexsent", "analyze", text],
            capture_output=True, text=True, timeout=5, cwd="/home/kak/Área de trabalho/Projetos/umcs"
        )
        output = result.stdout
        
        for line in output.split("\n"):
            if "Verdict:" in line:
                if "POSITIVE" in line:
                    return "POSITIVE"
                elif "NEGATIVE" in line:
                    return "NEGATIVE"
                else:
                    return "NEUTRAL"
        return "NEUTRAL"
    except:
        return "ERROR"

def vader_predict(text):
    score = vader.polarity_scores(text)['compound']
    if score >= 0.05:
        return "POSITIVE"
    elif score <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def textblob_predict(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0.05:
        return "POSITIVE"
    elif blob.sentiment.polarity < -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

print("=" * 70)
print("UMCS BENCHMARK REPORT")
print("=" * 70)
print()

umcs_correct = 0
vader_correct = 0
textblob_correct = 0
total = 0

results = []

for text, expected in test_cases:
    umcs_result = umcs_analyze(text)
    vader_result = vader_predict(text)
    textblob_result = textblob_predict(text)
    
    umcs_ok = 1 if umcs_result == expected else 0
    vader_ok = 1 if vader_result == expected else 0
    tb_ok = 1 if textblob_result == expected else 0
    
    umcs_correct += umcs_ok
    vader_correct += vader_ok
    textblob_correct += tb_ok
    total += 1
    
    status_umcs = "✓" if umcs_ok else "✗"
    status_vader = "✓" if vader_ok else "✗"
    status_tb = "✓" if tb_ok else "✗"
    
    print(f"Text: \"{text}\"")
    print(f"  Expected: {expected}")
    print(f"  UMCS: {umcs_result} {status_umcs}")
    print(f"  VADER: {vader_result} {status_vader}")
    print(f"  TextBlob: {textblob_result} {status_tb}")
    print()

print("=" * 70)
print("ACCURACY RESULTS")
print("=" * 70)
print(f"UMCS:      {umcs_correct}/{total} = {100*umcs_correct/total:.1f}%")
print(f"VADER:     {vader_correct}/{total} = {100*vader_correct/total:.1f}%")
print(f"TextBlob:  {textblob_correct}/{total} = {100*textblob_correct/total:.1f}%")
print()

print("=" * 70)
print("LANGUAGE COVERAGE")
print("=" * 70)
print("UMCS: 25+ languages (PT, EN, ES, IT, DE, FR, NL, AR, ZH, JA, RU, KO...)")
print("VADER: 1 language (English only)")
print("TextBlob: 1 language (English only)")
print()

print("=" * 70)
print("LEXICON SIZE")
print("=" * 70)
print("UMCS: 5,103,253 words (binary)")
print("VADER: ~7,500 words")
print("TextBlob: ~12,000 words")