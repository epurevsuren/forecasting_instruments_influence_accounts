import os
import csv
import time
import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

TICKERS = {
    'SPY': ('SPY', 'us'), 'VIX': ('^VIX', 'us'),
    'OIL': ('CL=F', '24h'), 'GOLD': ('GC=F', '24h'), 'BTC': ('BTC-USD', '24h'),
    'QQQ': ('QQQ', 'us'), 'DIA': ('DIA', 'us'),
    'XLI': ('XLI', 'us'), 'XLF': ('XLF', 'us'), 'XLE': ('XLE', 'us'),
    'COPPER': ('HG=F', '24h'), 'NATGAS': ('NG=F', '24h'),
    'EUR_USD': ('EURUSD=X', '24h'), 'USD_JPY': ('JPY=X', '24h'), 'GBP_USD': ('GBPUSD=X', '24h'),
    'USD_CNY': ('CNY=X', '24h'), 'USD_CAD': ('CAD=X', '24h'), 'USD_MXN': ('MXN=X', '24h'),
    'USD_CHF': ('CHF=X', '24h'), 'AUD_USD': ('AUDUSD=X', '24h'),
    'US10Y': ('^TNX', 'us'), 'US2Y': ('^IRX', 'us'),
    'ETH': ('ETH-USD', '24h'),
}

# --- 1. NEW DATA STRUCTURE: Define a sub-model for individual ticker impacts ---
class TickerImpact(BaseModel):
    ticker: str = Field(description="The specific ticker symbol from the TICKERS dictionary.")
    impact_percentage: float = Field(description="The estimated impact percentage for this specific ticker (e.g., 1.5 for 1.5%, -0.5 for -0.5%).")

# --- 2. UPDATE MAIN MODEL: Use a list of the sub-models ---
class TweetImpactAnalysis(BaseModel):
    impacted: bool = Field(description="True if the tweet likely caused a measurable market impact, False otherwise.")
    detailed_impacts: list[TickerImpact] = Field(description="List of specific impacted tickers and their respective impact percentages. Empty if none.")
    reasoning: str = Field(description="A brief 1-2 sentence explanation of why this tweet did or did not impact the market.")

def analyze_tweet(text: str, date: str, account: str) -> TweetImpactAnalysis:
    prompt = f"""
    Analyze the following tweet and determine its impact on the financial markets.
    
    Tweet Data:
    - Account: {account}
    - Date (NY Time): {date}
    - Tweet Text: "{text}"
    
    Available Tickers to choose from:
    {list(TICKERS.keys())}
    
    Instructions:
    1. Determine if this tweet contained highly material news that would move the market.
    2. Identify which specific tickers from the list above would be most directly impacted.
    3. For EACH impacted ticker, estimate its specific rough percentage impact (e.g., VIX might be 5.0% while SPY is -1.2%).
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TweetImpactAnalysis,
            temperature=0.1,
        ),
    )
    return TweetImpactAnalysis.model_validate_json(response.text)

def process_csv(input_filepath: str, output_filepath: str):
    print(f"Loading data from {input_filepath}...")
    df = pd.read_csv(input_filepath)
    
    required_cols = ['id', 'account', 'date', 'text', 'url']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # --- 3. UPDATE CSV COLUMNS: Replaced old columns with 'ticker_impacts' ---
    fieldnames = required_cols + ['is_impactful', 'ticker_impacts', 'reasoning']
    
    processed_ids = set()
    file_exists = os.path.exists(output_filepath)
    
    if file_exists:
        try:
            existing_df = pd.read_csv(output_filepath, usecols=['id'])
            processed_ids = set(existing_df['id'].astype(str).tolist())
            print(f"Found existing output file. {len(processed_ids)} rows already processed. Resuming...")
        except Exception:
            print("Existing file found but couldn't parse IDs cleanly. Starting fresh.")
            file_exists = False

    total_tweets = len(df)
    print(f"Found {total_tweets} total tweets in input file. Starting analysis loop...")

    with open(output_filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists or os.path.getsize(output_filepath) == 0:
            writer.writeheader()

        for index, row in df.iterrows():
            row_id_str = str(row['id'])
            
            if row_id_str in processed_ids:
                continue
                
            print(f"Processing row {index + 1}/{total_tweets} (ID: {row['id']})...")
            row_data = {col: row[col] for col in required_cols}
            
            max_retries = 5
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    analysis = analyze_tweet(str(row['text']), str(row['date']), str(row['account']))
                    
                    row_data['is_impactful'] = analysis.impacted
                    
                    # --- 4. FORMAT OUTPUT: Convert the list of objects into a readable string ---
                    impact_strings = []
                    for impact_item in analysis.detailed_impacts:
                        impact_strings.append(f"{impact_item.ticker}: {impact_item.impact_percentage}")
                    
                    # This creates a string like: "SPY: -1.2, VIX: 4.5, OIL: 2.0"
                    row_data['ticker_impacts'] = ", ".join(impact_strings) if impact_strings else ""
                    
                    row_data['reasoning'] = analysis.reasoning
                    success = True
                    
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        retry_count += 1
                        print(f"  [Rate Limit Hit] 429 Exhausted. Retrying row in 60s... (Attempt {retry_count}/{max_retries})")
                        time.sleep(60)
                    else:
                        print(f"  Error processing row {index + 1}: {e}")
                        row_data['is_impactful'] = "ERROR"
                        row_data['ticker_impacts'] = "ERROR"
                        row_data['reasoning'] = err_msg
                        success = True

            if not success:
                row_data['is_impactful'] = "ERROR_TIMEOUT"
                row_data['ticker_impacts'] = "ERROR_TIMEOUT"
                row_data['reasoning'] = "Max retries exceeded on 429 rate limits."

            writer.writerow(row_data)
            f.flush() 
            
            time.sleep(12.5)

    print(f"\nProcessing complete or caught up! File saved at: {output_filepath}")

if __name__ == "__main__":
    INPUT_CSV = "truth_social.csv"          
    OUTPUT_CSV = "truth_social_analyzed_detailed.csv" # Changed output name to avoid conflicting with your old columns
    
    process_csv(INPUT_CSV, OUTPUT_CSV)