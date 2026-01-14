from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd

def analyze_sentiment(headlines):
    """
    Uses FinBERT to classify financial headlines as Positive, Negative, or Neutral.
    """
    # 1. Load the FinBERT model (specifically trained for finance)
    model_name = "ProsusAI/finbert"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    
    # 2. Create a pipeline for sentiment analysis
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    # 3. Process the headlines
    results = nlp(headlines)
    return results

if __name__ == "__main__":
    # Test headlines - This is what you'd get from a news feed
    test_news = [
        "Apple reports record-breaking quarterly revenue and iPhone sales",
        "Stock market crashes as inflation fears grip investors",
        "Company maintains steady growth despite supply chain challenges"
    ]
    
    print("--- Analyzing Financial Sentiment ---")
    sentiment_results = analyze_sentiment(test_news)
    
    # Display the results
    for text, res in zip(test_news, sentiment_results):
        print(f"\nHeadline: {text}")
        print(f"Sentiment: {res['label']} (Confidence: {res['score']:.2f})")