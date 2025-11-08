# Text-Analysis-Project

Please read the [instructions](instructions.md).


## 1. Project Overview (~1 paragraph)
For this project, I collected text from four Wikipedia pages related to climate change: “Climate change,” “Global warming controversy,” “Fossil fuel industry,” and “Environmental movement.” I used Python to fetch and cache the text, clean and tokenize it, remove stopwords, compute word frequencies, and calculate TF-IDF scores to identify which words are most unique to each article. The goal was to understand how different parts of the climate change discussion are framed across different types of discourse. I also noticed several connections to my Quantitative Machine Learning course, particularly with text mining in R (tokenization, frequency tables, and TF-IDF). This project gave me the chance to implement similar analysis techniques in Python, which felt more intuitive and straightforward to control step-by-step. It was interesting to see how two different tools can solve the same problem in slightly different ways.


## 2. Implementation (~1-2 paragraphs)
My project is organized into two main Python files.  
**part01.py** fetches and harvests the articles from Wikipedia using the `mediawiki` library and saves them as local JSON files to avoid downloading the same data repeatedly. This step was important because it made testing faster and avoided unnecessary API calls. 
**part02.py** handles all text processing. I used regular expressions to tokenize the text and remove citation brackets. I used NLTK’s English stopword list to remove common filler words so I don't need to download any. Word frequencies were stored in Python dictionaries (histograms). For TF-IDF, I implemented the calculation manually using dictionary counts, total token length per document, and inverse document frequency across documents. This followed the algorithm we studied in class.

![Visual 1](ChatGPT%20Help/Visual%201.png)
![Visual 2](ChatGPT%20Help/Visual%202.png)
![Visual 3](ChatGPT%20Help/Visual%203.png)
![Sentiment Score](ChatGPT%20Help/Sentiment%20score.png)

One design decision I made was to store all the Wikipedia page titles in a single `topics` list rather than hard-coding them into multiple parts of the code or a specific Url imported. This allows the project to be reused easily with different text sources. If I want to analyze new topics, I only need to update the list, and the rest of the pipeline (fetching, cleaning, tokenizing, and analysis) will still work without any code changes. This design choice made the system more flexible and easier to maintain. I used AI to help with polish my data to make it more precise and the visualization and sentiment socre part of the project. I already understood the purpose of word clouds and bar charts from doing similar text mining work in RStudio in my Quantitative Machine Learning course, but I did not know how to generate those visualizations in Python. I provided ChatGPT with examples of the plots I had previously created in R, and asked how to produce the same type of graphs using Python. The AI mainly helped me translate the steps I already understood into Python code, rather than deciding the analysis for me.



## 3. Results (~1-3 paragraphs + figures/examples)
![Top 10 Words](data/outputs/Top%2010%20words.png)
![Top 20 ASCII](data/outputs/Top%2020%20(ASCII).png)
After cleaning and tokenizing the four articles, I ran word frequency analysis to identify the most common terms across the combined corpus. Unsurprisingly, the words “climate,” “change,” “warming,” and “global” appeared most frequently, reflecting how central these concepts are to the broader climate discourse. However, the frequency charts also revealed thematic differences among the pages. For example, the Fossil fuel industry page emphasized terms like “fossil,” “fuels,” “oil,” and “coal,” while the Environmental movement page highlighted “pollution,” “conservation,” and “nuclear.” These differences align well with the distinct focus and goals of each page.


![tfidf](data/outputs/tfidf.png)
To move beyond raw frequency and identify the unique thematic focus of each article, I calculated TF-IDF scores. This approach proved highly effective:

    The Fossil Fuel Industry page was characterized by technical and economic language, with high TF-IDF scores for terms like “petroleum,” “inflation,” “fuel,” and “burning.”
    In contrast, the Global Warming Controversy article was distinguished by its focus on scientific discourse, featuring elevated scores for words such as “debates,” “controversies,” “discrepancies,” and “apparent.”
    The Environmental Movement page, as expected, emphasized terms like “smoke,” “conservation,” and “tree.”
    This analysis confirms that while these articles share a common topical domain, their specific narratives are framed through distinct and specialized vocabularies. The TF-IDF metric successfully illuminated these unique thematic emphases.


![wordcloud](data/outputs/wordcloud.png)
![Top 20 Bar](data/outputs/top20_bar.png)
I also generated a word cloud and bar chart of the top 20 words across the combined text. The visualizations clearly showed the dominance of climate-related terminology, but also highlighted the influence of policy and industry terms such as “carbon,” “emissions,” “energy,” and “countries.” These visual patterns helped summarize the dataset in a way that is easier to interpret compared to raw frequency tables.


![Sentiment Analysis](data/outputs/Sentiment%20Analysis.png)
Finally, as an optional extension, I performed sentiment analysis on each article. Interestingly, three of the articles — Climate change, Global warming controversy, and Environmental movement — had strongly positive sentiment scores, while the Fossil fuel industry article had a negative score. This suggests that the language framing fossil fuels tends to be more negatively associated (e.g., pollution, damage, emissions), while pages discussing climate action and environmental advocacy carry more constructive or urgent tones. Although sentiment analysis is very context-sensitive, the results align with typical public and academic attitudes toward these topics.


## 4. Reflection (~1-2 paragraphs)
Overall, the project went well once I clearly separated the data collection (Part 1) from the text analysis steps (Part 2 and 3). The biggest challenge was actually selecting a meaningful topic. My original idea was to analyze Wikipedia pages about Penélope Cruz and Spanish cinema, but the language patterns there were not very distinctive or conceptually interesting. Switching to climate change–related pages made the analysis much stronger because the topics were more content-rich, and the differences across articles became clearer in both frequency analysis and TF-IDF scoring.

I did encounter several technical issues during the workflow. At one point, my laptop froze while running code, the VS Code terminal output was blank, and I was unsure whether I should ask for an extension. Eventually, after restarting and re-configuring the environment, the code began running properly again. This reminded me how important it is to save intermediate outputs and design the pipeline so that each step can be re-run independently without re-fetching or re-downloading data. Saving the Wikipedia pages locally was especially useful in preventing repeated API delays.

From a learning perspective, the biggest takeaway was realizing how similar text-processing workflows are across tools: the same concepts of tokenization, stopword removal, word frequencies, and TF-IDF that I used in R for Quantitative Machine Learning directly transferred into Python. AI tools also helped me translate ideas I already understood conceptually into Python implementations, especially when generating visualizations like the bar chart and word cloud. Going forward, I feel more confident about switching between programming environments and applying the same analytical logic with different libraries and languages.
