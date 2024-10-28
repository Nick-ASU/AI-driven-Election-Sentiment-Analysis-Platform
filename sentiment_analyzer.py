import pandas as pd
from transformers import pipeline
import logging
from pathlib import Path
import os

class MultiModelSentimentAnalyzer:
    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Initialize different sentiment analyzers
        self.models = {
            'distilbert': pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                top_k=1
            ),
            'roberta': pipeline(
                "sentiment-analysis",
                model="siebert/sentiment-roberta-large-english",
                top_k=1
            ),
            'twitter': pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=1
            ),
            'finbert': pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=1
            ),
            'bert_multi': pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                top_k=1
            )
        }
        
        # Get path to articles file
        self.project_root = Path(__file__).parent.parent.parent
        self.articles_file = self.project_root / "data" / "processed" / "all_articles.csv"
        
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Looking for articles at: {self.articles_file}")

    def get_sentiments(self, text: str) -> dict:
        results = {}
        
        for model_name, analyzer in self.models.items():
            try:
                result = analyzer(text)[0][0]
                
                # Convert to consistent categories
                label = result['label']
                score = result['score']
                
                if 'POSITIVE' in label.upper() and score > 0.75:
                    sentiment = 'positive'
                elif 'NEGATIVE' in label.upper() and score > 0.75:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                results[f'sentiment_{model_name}'] = sentiment
                results[f'confidence_{model_name}'] = round(score, 3)
                
            except Exception as e:
                self.logger.error(f"Error with {model_name} model: {str(e)}")
                results[f'sentiment_{model_name}'] = 'error'
                results[f'confidence_{model_name}'] = 0.0
                
        return results

    def analyze_articles(self):
        try:
            if not self.articles_file.exists():
                self.logger.error(f"File not found: {self.articles_file}")
                return
            
            # Read the CSV file
            df = pd.read_csv(self.articles_file)
            self.logger.info(f"Loaded {len(df)} articles")
            
            # Process each article with all models
            results = []
            total = len(df)
            
            for idx, row in df.iterrows():
                # Log progress
                if idx % 10 == 0:
                    self.logger.info(f"Processing article {idx}/{total}")
                
                # Get sentiments from all models
                sentiments = self.get_sentiments(row['title'])
                results.append(sentiments)
            
            # Convert results to DataFrame and join with original
            results_df = pd.DataFrame(results)
            df = pd.concat([df, results_df], axis=1)
            
            # Calculate consensus sentiment
            sentiment_columns = [col for col in df.columns if col.startswith('sentiment_')]
            df['sentiment_consensus'] = df[sentiment_columns].apply(
                lambda x: max(set(x), key=list(x).count)
            )
            
            # Save results
            try:
                df.to_csv(self.articles_file, index=False, encoding='utf-8')
                self.logger.info(f"Saved results to: {self.articles_file}")
            except PermissionError:
                new_file = self.articles_file.parent / "all_articles_multi_sentiment.csv"
                df.to_csv(new_file, index=False, encoding='utf-8')
                self.logger.info(f"Permission denied on original file. Saved to: {new_file}")
                
            # Print summary
            self.logger.info("\nSentiment Analysis Summary:")
            for model in self.models.keys():
                col = f'sentiment_{model}'
                counts = df[col].value_counts()
                self.logger.info(f"\n{model} results:")
                for sentiment, count in counts.items():
                    self.logger.info(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
            
            consensus_counts = df['sentiment_consensus'].value_counts()
            self.logger.info("\nConsensus Results:")
            for sentiment, count in consensus_counts.items():
                self.logger.info(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
                
        except Exception as e:
            self.logger.error(f"Error processing articles: {str(e)}")
            raise

if __name__ == "__main__":
    analyzer = MultiModelSentimentAnalyzer()
    analyzer.analyze_articles()