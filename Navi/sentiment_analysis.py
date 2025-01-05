from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
from transformers import pipeline

def sa_vader(data, column):
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_scores(text):
        if not isinstance(text, str):
            return None
        return analyzer.polarity_scores(text)

    data['sentiment_scores'] = data[column].apply(get_sentiment_scores)
    data['compound_score'] = data['sentiment_scores'].apply(lambda x: x['compound'] if x else None)
    data['positive_score'] = data['sentiment_scores'].apply(lambda x: x['pos'] if x else None)
    data['neutral_score'] = data['sentiment_scores'].apply(lambda x: x['neu'] if x else None)
    data['negative_score'] = data['sentiment_scores'].apply(lambda x: x['neg'] if x else None)

    return data

def sa_flair(data, column):
    flair_classifier = TextClassifier.load("sentiment")

    def get_flair_sentiment(text):
        if not isinstance(text, str):
            return None
        sentence = Sentence(text)
        flair_classifier.predict(sentence)
        label = sentence.labels[0].value
        score = sentence.labels[0].score
        return {"label": label, "score": score}

    data["flair_sentiment"] = data[column].apply(get_flair_sentiment)
    return data

def sa_distilbert(data, column):
    distilbert_classifier = pipeline("sentiment-analysis")
    def get_distilbert_sentiment(text):
        if not isinstance(text, str):
            return None
        result = distilbert_classifier(text)[0]
        return {"label": result["label"], "score": result["score"]}

    data["distilbert_sentiment"] = data[column].apply(get_distilbert_sentiment)
    return data