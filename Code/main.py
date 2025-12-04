import os, re, math, nltk, openai
from collections import Counter
from nltk.corpus import stopwords as nltk_stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Code.fetch_data import topics, fetch_page
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

### Text Processing and Cleaning
word_pattern = re.compile(r"[a-zA-Z']+")

def normalize_whitespace(text):
    """collapse repeated spaces/newlines into single spaces"""
    return re.sub(r"\s+", " ", text).strip()

def strip_brackets(text):
    """remove citation markers like [12] common in wikipedia text"""
    return re.sub(r"\[\d+\]", "", text)

def drop_sections(text, section_heads=("references", "external links", "see also", "notes")):
    """
    cut the article before trailing non-content sections 
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
    """lowercase + extract alphabetic tokens"""
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
    stopwords = set(stopwords_list)   # convert once for speed!
    
    cleaned = []
    for t in tokens:
        if t not in stopwords:
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

    # Combine all pages into one text corpus - Consulted ChatGPT to incorporate text mining knowledge from my QTM course
    combined_text = "\n".join(pages)

    # Tokenize text into lowercase alphabetic words
    tokens = tokenize(combined_text)

    # Load stopwords list 
    stopwords_list = load_stopwords()
    # Remove stopwords using list-based membership check - consulted by ChatGPT
    cleaned_tokens = remove_stopwords(tokens, stopwords_list)

    # Sanity check 
    print("Total tokens before stopword removal:", len(tokens))
    print("Total tokens after stopword removal: ", len(cleaned_tokens))
    print("Sample cleaned tokens:", cleaned_tokens[:30])

    return cleaned_tokens

### Part 2.3 Word Frequency Analysis
def count_words(tokens):
    """Build a frequency dictionary (word -> count)."""
    hist = {}
    for t in tokens:
        hist[t] = hist.get(t, 0) + 1
    return hist

def most_common(hist):
    """
    Convert dict to list of (freq, word) pairs, sorted by frequency desc.
    """
    items = []
    for word, freq in hist.items():
        items.append((freq, word))
    items.sort(key=lambda x: (-x[0], x[1]))
    return items

def print_most_common(hist, num=20):
    """Print top `num` most frequent words."""
    t = most_common(hist)
    for freq, word in t[:num]:
        print(f"{word}\t{freq}")

def run_frequency_test():
    cleaned_tokens = clean_and_tokenize()
    hist = count_words(cleaned_tokens)

    print("\nTop 20 Most Frequent Words (after cleaning + stopwords removed):\n")
    print_most_common(hist, num=20)

## TFIDF test - consulted by ChatGPT and my learning from QTM RStudio code
def document_tokens(separate=True):
    """
    Fetch and clean pages.
    Args:
        separate (bool): If True, return list of token lists (one per document).
                         If False, return a single combined list of tokens.
    Returns:
        list: Either list-of-lists (separate=True) or one flat list (separate=False).
    """
    docs = []
    for title in topics:
        raw = fetch_page(title)
        text = drop_sections(raw)
        text = strip_brackets(text)
        text = normalize_whitespace(text)
        tokens = tokenize(text)
        tokens = remove_stopwords(tokens, load_stopwords())
        docs.append(tokens)

    if separate:
        return docs
    else:
        # flatten into one combined list
        return [t for doc in docs for t in doc]

def compute_tf(doc_tokens):
    """
    Compute term frequency (TF) for each document.
    Returns a list of dicts: one frequency dictionary per document.
    """
    tf_list = []
    for tokens in doc_tokens:
        total = len(tokens)
        freq = Counter(tokens)
        tf = {}
        for word, count in freq.items():
            tf[word] = count / total
        tf_list.append(tf)
    return tf_list


def compute_idf(doc_tokens):
    """
    Compute inverse document frequency across documents.
    """
    num_docs = len(doc_tokens)
    idf = {}
    # count in how many documents each term appears
    doc_word_sets = [set(tokens) for tokens in doc_tokens]
    all_words = set().union(*doc_word_sets)

    for word in all_words:
        docs_containing = sum(1 for doc in doc_word_sets if word in doc)
        idf[word] = math.log(num_docs / docs_containing)
    return idf

def compute_tfidf(tf_list, idf):
    """
    Compute TF-IDF scores:
    For each document, multiply its TF by the IDF.
    Returns a list of dicts (one TF-IDF map per document).
    """
    tfidf_list = []
    for tf in tf_list:
        doc_tfidf = {}
        for word, tf_value in tf.items():
            doc_tfidf[word] = tf_value * idf[word]
        tfidf_list.append(doc_tfidf)
    return tfidf_list

def print_top_tfidf(tfidf_list, num=10):
    """
    Print top TF-IDF words for each document.
    """
    for title, tfidf in zip(topics, tfidf_list):
        print(f"\nTop TF-IDF terms for: {title}\n")
        sorted_words = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:num]:
            print(f"{word:15}  {score:.4f}")

def run_tfidf():
    docs = document_tokens()
    tf_list = compute_tf(docs)
    idf = compute_idf(docs)
    tfidf_list = compute_tfidf(tf_list, idf)
    print_top_tfidf(tfidf_list, num=10)

### Part 2.4 Computing Summary Statistics 

def top_k_words(tokens, k=20): 
    """Return the top-k most frequent words in one list of tokens."""
    hist = {}
    for t in tokens:
        hist[t] = hist.get(t, 0) + 1
    items = sorted(hist.items(), key=lambda x: (-x[1], x[0]))
    return items[:k]

def avg_word_length(tokens):
    """Average length of words in one document."""
    if not tokens:
        return 0
    total_chars = sum(len(t) for t in tokens)
    return total_chars / len(tokens)

def type_token_ratio(tokens):
    """Vocabulary richness: unique words / total words."""
    if not tokens:
        return 0
    return len(set(tokens)) / len(tokens)

def words_unique_to_document(all_docs):
    """
    Words that appear in one document ONLY.
    Returns list: one set of unique words per document.
    """
    doc_sets = [set(d) for d in all_docs]
    unique_words = []
    for i, s in enumerate(doc_sets):
        others = set().union(*[doc_sets[j] for j in range(len(doc_sets)) if j != i])
        unique_words.append(s - others)
    return unique_words

def run_summary_statistics():  # - consulted by ChatGPT
    docs = document_tokens()

    print("\n===== PART 4: SUMMARY STATISTICS =====\n")

    # 1) Top 20 frequent words per document
    for title, tokens in zip(topics, docs):
        print(f"\nTop 20 Words in: {title}\n")
        for word, count in top_k_words(tokens, k=20):
            print(f"{word:15} {count}")
    
    # 2) Vocabulary richness
    print("\nVocabulary Richness (Type-Token Ratio):\n")
    for title, tokens in zip(topics, docs):
        print(f"{title:30}  TTR = {type_token_ratio(tokens):.3f}")

    # 3) Average word length
    print("\nAverage Word Length:\n")
    for title, tokens in zip(topics, docs):
        print(f"{title:30}  Avg Word Length = {avg_word_length(tokens):.2f}")

    # 4) Words that appear in ONLY one document
    print("\nUnique distinguishing words (sample of 10 each):\n")
    unique_sets = words_unique_to_document(docs)
    for title, uw in zip(topics, unique_sets):
        sample = list(uw)[:10]    # just show a few
        print(f"{title:30}  {sample}")

### Part 2.5 Visualizations
# A simple ASCII bar chart
def ascii_bar_chart(freq_pairs, width=40):
    """
    Print a horizontal ASCII bar chart for (word, count) pairs.
    width controls scale of the longest bar.
    """
    if not freq_pairs:
        print("(no data)")
        return
    max_count = freq_pairs[0][1]
    for word, count in freq_pairs:
        bar = "#" * int((count / max_count) * width)
        print(f"{word:15} | {bar} {count}")

def visualize_top_words_per_document(k=10):
    """
    For each document (each Wikipedia page), compute its top-k words
    and display an ASCII bar chart.
    """
    docs = document_tokens()  # reuse function from Part 4

    print("\n===== PART 2.5: VISUALIZATION =====\n")
    for title, tokens in zip(topics, docs):
        print(f"\nTop {k} words in: {title}\n")
        top_words = top_k_words(tokens, k=k)
        ascii_bar_chart(top_words, width=40)

### Advanced Visualizations with Matplotlib and WordCloud - consulted by ChatGPT and R code
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def top_k_from_hist(hist, k=20):
    """Return top-k (word, count) pairs from a histogram dict."""
    items = list(hist.items())
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:k]

def print_ascii_bar(pairs, width=50, title=None):
    """ASCII bar chart for quick console screenshots."""
    if title:
        print("\n" + title + "\n" + "-" * len(title))
    if not pairs:
        print("(no data)")
        return
    maxc = pairs[0][1]
    for w, c in pairs:
        bar_len = int((c / maxc) * width) if maxc else 0
        bar = "#" * bar_len
        print(f"{w:>20} | {bar} {c}")

# matplotlib bar plot
def plot_bar_topk(pairs, title="Top 20 Words", outpath="data/visuals/top20_bar.png"):
    import matplotlib.pyplot as plt
    ensure_dir(outpath)
    words = [w for w, _ in pairs]
    counts = [c for _, c in pairs]

    plt.figure(figsize=(10, 6))
    # horizontal bar so labels are readable
    y = range(len(words))
    plt.barh(y, counts)
    plt.yticks(y, words)
    plt.gca().invert_yaxis()  # largest on top
    plt.title(title)
    plt.xlabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[saved] bar chart -> {outpath}")

# wordcloud from frequency dict
def plot_wordcloud_from_hist(hist, outpath="data/visuals/wordcloud.png"):
    from wordcloud import WordCloud
    ensure_dir(outpath)
    # basic, legible defaults; no custom colors to keep it simple
    wc = WordCloud(width=1200, height=700, background_color="white")
    wc.generate_from_frequencies(hist)
    wc.to_file(outpath)
    print(f"[saved] word cloud -> {outpath}")

def run_visualizations():
    """
    Uses your existing pipeline:
      - cleans + tokenizes
      - builds frequency histogram
      - prints ASCII bar chart
      - saves matplotlib bar chart
      - saves word cloud
    """
    cleaned_tokens = clean_and_tokenize()
    hist = count_words(cleaned_tokens)
    top20 = top_k_from_hist(hist, k=20)

    # 1) ASCII in terminal
    print_ascii_bar(top20, width=50, title="Top 20 words (ASCII)")

    # 2) Matplotlib bar plot
    plot_bar_topk(top20, title="Top 20 Words After Cleaning + Stopwords", outpath="data/visuals/top20_bar.png")

    # 3) Word cloud
    plot_wordcloud_from_hist(hist, outpath="data/visuals/wordcloud.png")

# ========= OPTIONAL TECHNIQUE: SENTIMENT ANALYSIS (VADER) =========
def sentiment_scores(text): # - consulted by ChatGPT
    """
    Compute overall compound sentiment score (-1 to +1).
    """
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)["compound"]


def run_sentiment_analysis(): # - consulted by ChatGPT
    print("\n===== OPTIONAL TECHNIQUE: SENTIMENT ANALYSIS =====\n")

    for title in topics:
        raw = fetch_page(title)
        text = drop_sections(raw)
        text = strip_brackets(text)
        text = normalize_whitespace(text)

        score = sentiment_scores(text)

        print(f"{title:35}  Sentiment Score = {score:.4f}")

def summarize_with_openai(text, model="gpt-4o-mini"):
    """
    Use OpenAI API to summarize text.
    
    Args:
        text (str): Input text to summarize.
        model (str): OpenAI model name (default: gpt-4o-mini).
    Returns:
        str: Summary text from the OpenAI API.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for text analysis."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=300
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[OpenAI summary failed: {e}]"

def run_openai_summary():
    """
    Run OpenAI summarization on each document.
    """
    docs = document_tokens(separate=True)  # keep each page separate
    for title, tokens in zip(topics, docs):
        # limit length so API call is manageable
        text = " ".join(tokens[:1000])
        summary = summarize_with_openai(text)
        print(f"\nOpenAI Summary for {title}:\n{summary}\n")

def main():
    """
    entry point for Part 2:
      1) cleaning + stopwords + frequency (required)
      2) TF-IDF across documents (nice, still Part 2)
      3) summary statistics (required)
      4) at least one visualization (required)
      5) optional technique (sentiment) — best effort
    """
    print("\n===== PART 2: CLEANING + FREQUENCY =====")
    run_frequency_test()

    print("\n===== PART 2: TF-IDF (per document) =====")
    try:
        run_tfidf()
    except Exception as e:
        print("[tfidf skipped]", e)

    print("\n===== PART 4: SUMMARY STATISTICS =====")
    try:
        run_summary_statistics()
    except Exception as e:
        print("[summary stats skipped]", e)

    print("\n===== PART 2.5: VISUALIZATION =====")
    try:
        run_visualizations()
    except Exception as e:
        print("[visualization skipped]", e)

    print("\n===== OPTIONAL: SENTIMENT ANALYSIS (VADER) =====")
    try:
        run_sentiment_analysis()
    except Exception as e:
        print("[sentiment skipped]", e)
    
    print("\n===== OPTIONAL: SENTIMENT ANALYSIS (VADER) =====")
    try:
        run_sentiment_analysis()
    except Exception as e:
        print("[sentiment skipped]", e)

    print("\n===== OPTIONAL: OPENAI SUMMARIZATION =====")
    try:
        run_openai_summary()
    except Exception as e:
        print("[openai summary skipped]", e)


if __name__ == "__main__":
    main()



