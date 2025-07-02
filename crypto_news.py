import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from datetime import datetime
from collections import Counter
from typing import List, Dict, Optional, Tuple
import sqlite3
from sqlite3 import Error

class CryptoNewsAnalyzer:
    def __init__(self, api_key: str, db_path: str = 'crypto_news.db'):
        self.api_key = api_key
        self.base_url = "https://min-api.cryptocompare.com/data/v2/news/"
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for storing news articles."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    url TEXT,
                    source TEXT,
                    published_on INTEGER,
                    body TEXT,
                    sentiment_score REAL,
                    category TEXT,
                    tags TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS coins_mentioned (
                    article_id TEXT,
                    coin_symbol TEXT,
                    FOREIGN KEY (article_id) REFERENCES articles (id)
                )
            ''')
            
            conn.commit()
            
        except Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def get_crypto_news(self, limit: int = 20) -> List[Dict]:
        """Fetch and process cryptocurrency news with enhanced features."""
        headers = {'Authorization': f'Apikey {self.api_key}'}
        params = {
            'lang': 'EN',
            'categories': 'ALL_CRYPTO',
            'excludeCategories': 'ALL_NON_CRYPTO',
            'languages': 'en',
            'limit': limit
        }
        
        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json().get('Data', [])
            
            processed_articles = []
            for article in data:
                processed = self._process_article(article)
                processed_articles.append(processed)
                self._store_article(processed)
            
            return processed_articles
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return []

    def _process_article(self, article: Dict) -> Dict:
        """Process and enrich article data with sentiment analysis and other metrics."""
        # Basic article info
        processed = {
            'id': article.get('id'),
            'title': article.get('title'),
            'url': article.get('url'),
            'source': article.get('source'),
            'published_on': article.get('published_on', 0),
            'body': article.get('body', ''),
            'category': article.get('categories', ''),
            'tags': article.get('tags', '')
        }
        
        # Sentiment analysis
        sentiment = self._analyze_sentiment(processed['title'] + ' ' + processed['body'])
        processed['sentiment_score'] = sentiment
        
        # Extract mentioned coins
        mentioned_coins = self._extract_coins(processed['title'] + ' ' + processed['body'])
        processed['mentioned_coins'] = mentioned_coins
        
        return processed

    def _analyze_sentiment(self, text: str) -> float:
        """Perform sentiment analysis on text using TextBlob."""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def _extract_coins(self, text: str) -> List[str]:
        """Extract cryptocurrency mentions from text."""
        # This is a simplified version - you might want to use a comprehensive coin list
        common_coins = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE', 'SHIB', 'MATIC']
        mentioned = [coin for coin in common_coins if coin.lower() in text.lower()]
        return mentioned

    def _store_article(self, article: Dict) -> None:
        """Store article in SQLite database."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert or replace article
            cursor.execute('''
                INSERT OR REPLACE INTO articles 
                (id, title, url, source, published_on, body, sentiment_score, category, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article['id'],
                article['title'],
                article['url'],
                article['source'],
                article['published_on'],
                article['body'],
                article['sentiment_score'],
                article['category'],
                article['tags']
            ))
            
            # Store mentioned coins
            for coin in article.get('mentioned_coins', []):
                cursor.execute('''
                    INSERT OR IGNORE INTO coins_mentioned (article_id, coin_symbol)
                    VALUES (?, ?)
                ''', (article['id'], coin))
            
            conn.commit()
            
        except Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def get_trending_coins(self, hours: int = 24) -> List[Tuple[str, int]]:
        """Get most mentioned coins in the last X hours."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get articles from the last X hours
            time_threshold = int(datetime.now().timestamp()) - (hours * 3600)
            
            cursor.execute('''
                SELECT cm.coin_symbol, COUNT(*) as mention_count
                FROM coins_mentioned cm
                JOIN articles a ON cm.article_id = a.id
                WHERE a.published_on >= ?
                GROUP BY cm.coin_symbol
                ORDER BY mention_count DESC
                LIMIT 10
            ''', (time_threshold,))
            
            return cursor.fetchall()
            
        except Error as e:
            print(f"Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_sentiment_analysis(self, coin: str = None) -> Dict:
        """Get sentiment analysis for all news or a specific coin."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if coin:
                # Get sentiment for a specific coin
                cursor.execute('''
                    SELECT AVG(a.sentiment_score) as avg_sentiment, 
                           COUNT(*) as article_count
                    FROM articles a
                    JOIN coins_mentioned cm ON a.id = cm.article_id
                    WHERE cm.coin_symbol = ?
                ''', (coin,))
            else:
                # Get overall sentiment
                cursor.execute('''
                    SELECT AVG(sentiment_score) as avg_sentiment,
                           COUNT(*) as article_count
                    FROM articles
                ''')
            
            result = cursor.fetchone()
            return {
                'average_sentiment': result[0] if result[0] else 0,
                'article_count': result[1] if result[1] else 0,
                'coin': coin if coin else 'all'
            }
            
        except Error as e:
            print(f"Database error: {e}")
            return {'average_sentiment': 0, 'article_count': 0, 'coin': coin if coin else 'all'}
        finally:
            if conn:
                conn.close()

    def plot_sentiment_trend(self, days: int = 7) -> None:
        """Plot sentiment trend over time."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get daily sentiment
            df = pd.read_sql('''
                SELECT 
                    date(datetime(published_on, 'unixepoch')) as date,
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as article_count
                FROM articles
                WHERE published_on >= ?
                GROUP BY date
                ORDER BY date
            ''', conn, params=[int(datetime.now().timestamp()) - (days * 24 * 3600)])
            
            if df.empty:
                print("No data available for the specified period.")
                return
                
            # Create plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Sentiment line
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Average Sentiment', color=color)
            ax1.plot(df['date'], df['avg_sentiment'], color=color, marker='o')
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Article count bars
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Number of Articles', color=color)
            ax2.bar(df['date'], df['article_count'], color=color, alpha=0.3)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Formatting
            plt.title(f'Crypto News Sentiment Trend (Last {days} Days)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        except Error as e:
            print(f"Error generating plot: {e}")
        finally:
            if conn:
                conn.close()

# Example usage
if __name__ == "__main__":
    # Initialize with your API key
    analyzer = CryptoNewsAnalyzer(api_key='YOUR_API_KEY')
    
    # Fetch and analyze latest news
    print("Fetching latest crypto news...")
    articles = analyzer.get_crypto_news(limit=20)
    print(f"Processed {len(articles)} articles.")
    
    # Get trending coins
    print("\nTrending coins in the last 24 hours:")
    trending = analyzer.get_trending_coins(hours=24)
    for coin, count in trending:
        print(f"{coin}: {count} mentions")
    
    # Get sentiment analysis
    print("\nSentiment Analysis:")
    overall = analyzer.get_sentiment_analysis()
    print(f"Overall sentiment: {overall['average_sentiment']:.2f} (based on {overall['article_count']} articles)")
    
    # Plot sentiment trend
    print("\nGenerating sentiment trend plot...")
    analyzer.plot_sentiment_trend(days=7)