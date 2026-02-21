import os
import sys
# 1. CONNECTION FIX (Critical for Mac/Hotspots)
os.environ["GRPC_DNS_RESOLVER"] = "native"

from dotenv import load_dotenv
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# 2. NEW LIBRARY IMPORT
from google import genai

# Load Environment Variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Fallback if .env fails
if not API_KEY:
    print("⚠️  .env file not found or empty.")
    API_KEY = input("Paste your API Key here: ").strip()

# 3. SETUP CLIENT
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"❌ Error setting up Gemini Client: {e}")
    sys.exit(1)

# --- DATA FETCHING (nba_api) ---
def get_last_5_games(player_name):
    print(f"\n🏀 Searching for {player_name}...")
    nba_players = players.find_players_by_full_name(player_name)
    if not nba_players:
        return None, "Player not found."
    
    player_id = nba_players[0]['id']
    try:
        # Fetch 2025-26 Season Stats
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2025-26')
        df = gamelog.get_data_frames()[0]
        
        if df.empty: 
            return None, "No games played this season."
            
        # Clean Data for the AI
        subset = df[['GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'REB', 'AST', 'FG3M', 'MIN']].head(5)
        return nba_players[0]['full_name'], subset
    except Exception as e:
        return None, f"NBA API Error: {e}"

# --- PREDICTION LOGIC ---
def get_prediction(player_name, line, stat_type, stats_df):
    print(f"🤖 Analyzing with Gemini 2.5 Flash...")
    
    prompt = f"""
    You are a professional sports bettor.
    
    TASK: Predict OVER or UNDER {line} {stat_type} for {player_name}.
    
    DATA (Last 5 Games):
    {stats_df.to_string(index=False)}
    
    INSTRUCTIONS:
    - Analyze the trend (is he heating up?).
    - Check consistency (how often did he hit this line?).
    - Provide a clear recommendation.
    
    OUTPUT:
    "PREDICTION: [OVER/UNDER]"
    "CONFIDENCE: [0-100]%"
    "REASONING: [One sentence]"
    """

    try:
        # 4. MODEL CALL (Updated to the one found in your list)
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Generation Error: {e}"

# --- MAIN APP ---
if __name__ == "__main__":
    print("="*40)
    print("🚀 NBA PREDICTOR")
    print("="*40)

    while True:
        p_name = input("\nEnter Player Name (or 'q' to quit): ").strip()
        if p_name.lower() == 'q': break
        
        real_name, stats = get_last_5_games(p_name)
        
        if stats is not None:
            print(f"✅ Found data for {real_name}")
            stat = input("Enter Stat (e.g. Points, Rebounds): ")
            line = input(f"Enter Line for {stat} (e.g. 24.5): ")
            
            prediction = get_prediction(real_name, line, stat, stats)
            print("\n" + "-"*40)
            print(prediction)
            print("-" * 40)
        else:
            print(f"❌ {stats}")print('Prian is testing')
