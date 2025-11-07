import os, re, nltk
from nltk.corpus import stopwords as nltk_stopwords
from collections import Counter

# import topics and fetch_page from your Part 1 module 
from part01 import topics, fetch_page

# ensure stopwords are available (no-op if already downloaded)
nltk.download("stopwords", quiet=True)

### Text Processing and Cleaning
word_pattern = re.compile(r"[a-zA-Z']+")

def normalize_whitespace(text):
    """collapse repeated spaces/newlines into single spaces (tidier tokens)"""
    return re.sub(r"\s+", " ", text).strip()

def strip_brackets(text):
    """remove citation markers like [12] common in wikipedia text"""
    return re.sub(r"\[\d+\]", "", text)

def drop_sections(text, section_heads=("references", "external links", "see also", "notes")):
    """
    cut the article before trailing non-content sections (quick + crude)
    looks for headings by keyword in a lowercase copy of the text
    """
    lower = text.lower()
    cut = len(text)
    for h in section_heads:
        i = lower.find("\n" + h.lower())
        if i != -1:
            cut = min(cut, i)
    return text[:cut]

def tokenize(text):
    """lowercase + extract alphabetic tokens (keeps apostrophes)"""
    return word_pattern.findall(text.lower())



### Part 2.2 Stopword Removal
def load_stopwords():
    """
    Load NLTK English stopwords and return them as a list.
    We convert to lowercase to match our tokenized text.
    """
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    nltk.download("stopwords", quiet=True)  # safe to call repeatedly
    return [w.lower() for w in nltk_stopwords.words("english")]

def remove_stopwords(tokens, stopwords_list):
    """
    Filter out tokens that appear in the stopword set.
    """
    cleaned = []
    for t in tokens:
        if t not in stopwords_list:   # list-based membership check
            cleaned.append(t)
    return cleaned



### Pipeline step for part 2
def clean_and_tokenize():
    """
    Fetch, clean, combine, tokenize, and remove stopwords.
    Returns a list of cleaned tokens for further analysis (Part 2.3).
    """
    pages = []
    for title in topics:
        raw = fetch_page(title)
        text = drop_sections(raw)         # remove trailing reference sections
        text = strip_brackets(text)       # remove [12]-style citations
        text = normalize_whitespace(text) # collapse extra spaces/newlines
        pages.append(text)

    # Combine all pages into one text corpus
    combined_text = "\n".join(pages)

    # Tokenize text into lowercase alphabetic words
    tokens = tokenize(combined_text)

    # Load stopwords list 
    stopwords_list = load_stopwords()
    # Remove stopwords using list-based membership check
    cleaned_tokens = remove_stopwords(tokens, stopwords_list)

    # Sanity check prints (useful for debugging/report screenshots)
    print("Total tokens before stopword removal:", len(tokens))
    print("Total tokens after stopword removal: ", len(cleaned_tokens))
    print("Sample cleaned tokens:", cleaned_tokens[:30])

    return cleaned_tokens




