-- ============================================================
-- NASDAQ Sector Sentiment Table
-- Stores daily sector-level sentiment scores from news sources
-- Used by ML model to adjust Buy/Sell signals based on market mood
-- ============================================================

IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'nasdaq_sector_sentiment' AND TABLE_SCHEMA = 'dbo')
BEGIN
    CREATE TABLE dbo.nasdaq_sector_sentiment (
        id                      INT IDENTITY(1,1) PRIMARY KEY,
        trading_date            DATE NOT NULL,
        sector                  VARCHAR(100) NOT NULL,      -- e.g., 'Technology', 'Healthcare', 'Financial Services'
        
        -- Aggregated sentiment scores (range: -1.0 to +1.0)
        sentiment_score         FLOAT NOT NULL DEFAULT 0,   -- Weighted avg of VADER + FinBERT
        vader_score             FLOAT DEFAULT 0,            -- VADER compound score
        finbert_score           FLOAT DEFAULT 0,            -- FinBERT score (positive - negative)
        
        -- Sentiment distribution
        positive_ratio          FLOAT DEFAULT 0,            -- % of positive headlines
        negative_ratio          FLOAT DEFAULT 0,            -- % of negative headlines
        neutral_ratio           FLOAT DEFAULT 0,            -- % of neutral headlines
        
        -- Volume and confidence
        news_count              INT DEFAULT 0,              -- Number of articles analyzed
        source_count            INT DEFAULT 0,              -- Number of distinct sources
        confidence              FLOAT DEFAULT 0,            -- Confidence in score (0-1, based on news_count)
        
        -- Derived features (pre-computed for ML)
        sentiment_momentum_3d   FLOAT DEFAULT 0,            -- 3-day sentiment change
        sentiment_momentum_7d   FLOAT DEFAULT 0,            -- 7-day sentiment change
        sentiment_vs_avg_30d    FLOAT DEFAULT 0,            -- Current vs 30-day avg (z-score style)
        
        -- Market-level sentiment (same for all sectors on a given date)
        market_sentiment_score  FLOAT DEFAULT 0,            -- Overall US market sentiment
        market_news_count       INT DEFAULT 0,              -- Overall market news volume
        
        -- Metadata
        sources                 VARCHAR(500) DEFAULT '',    -- Comma-separated source names
        last_updated            DATETIME DEFAULT GETDATE(),
        
        -- Unique constraint: one row per sector per date
        CONSTRAINT UQ_nasdaq_sector_sentiment UNIQUE (trading_date, sector)
    );

    -- Index for fast lookups by date (ML training data loading)
    CREATE NONCLUSTERED INDEX IX_nasdaq_sector_sentiment_date 
    ON dbo.nasdaq_sector_sentiment(trading_date) INCLUDE (sector, sentiment_score, news_count);
    
    -- Index for sector-based queries
    CREATE NONCLUSTERED INDEX IX_nasdaq_sector_sentiment_sector
    ON dbo.nasdaq_sector_sentiment(sector, trading_date) INCLUDE (sentiment_score, confidence);

    PRINT 'Table nasdaq_sector_sentiment created successfully with indexes';
END
ELSE
BEGIN
    PRINT 'Table nasdaq_sector_sentiment already exists';
END
