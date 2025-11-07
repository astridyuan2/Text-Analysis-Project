# import necessary libraries
from mediawiki import MediaWiki
import os, json, re

## Example usage of MediaWiki to fetch a page
# wikipedia = MediaWiki()
# babson = wikipedia.page("Babson College")
# print(babson.title)
# print(babson.content)

# Define topics and paths of the text analysis project
topics = [
    "Penélope Cruz",
    "Pedro Almodóvar",
    "Spanish cinema",
    "Female representation in film"
]

downloads_dict = "data/downloads"   # where JSON pages will be stored


### File helpers

# Safe python datda to Json file, polished by Chatgpt
def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)     # create necessary directories
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Read Json file and with the python objects, then store in json again, polished by Chatgpt
def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

# Converts any string into a safe filename or URL slug, polished by Chatgpt
def slug(name):
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")  #replaces any non-alphanumeric characters with underscores




### Fetch data from wikipedia
def fetch_page(title):
    """
    Fetch Wikipedia page text, but store once in downloads_dict
    so we do not repeatedly re-download.
    """
    filepath = f"{downloads_dict}/{slug(title)}.json"

    if os.path.exists(filepath):
        data = load_json(filepath)
        return data.get("content", "")

    wiki = MediaWiki()
    page = wiki.page(title)
    save_json({"title": page.title, "content": page.content}, filepath)
    return page.content


def main():
    """
    part 1: just download + cache
    no cleaning or counting here.
    """
    for t in topics:
        text = fetch_page(t)
        print(f"[saved] {t} ({len(text)} characters)")

if __name__ == "__main__":
    main()

