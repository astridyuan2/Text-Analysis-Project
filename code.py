import urllib.request

def main():
    url = 'https://www.gutenberg.org/cache/epub/730/pg730.txt'
    try:
        with urllib.request.urlopen(url) as f:
            text = f.read().decode('utf-8')
            print(text) # for testing
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()