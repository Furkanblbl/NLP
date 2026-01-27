from transformers import pipeline

summarizer = pipeline("summarization")
text = """
Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on enabling computers
to understand, interpret, and generate human language. It combines linguistics, computer science, and
machine learning to build systems that can work with text and speech in practical ways.

Modern NLP often relies on neural networks and transformer-based models that learn patterns from large
collections of data. These models can perform tasks such as translation, summarization, question answering,
sentiment analysis, and information extraction. Instead of manually writing rules for every language case,
NLP systems learn statistical relationships between words, sentences, and meanings.

In real-world applications, NLP helps power chatbots, search engines, recommendation systems, and tools
that automatically organize or analyze documents. As data grows and models improve, NLP continues to make
human-computer interaction more natural and more useful in everyday products.
"""

 # Generate summary
summary = summarizer(
     text,
     max_length=90,
     min_length=30,
    do_sample=True
 )

print("Summary:", summary[0]['summary_text'])