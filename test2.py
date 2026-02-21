import os
import sys
import re
import json
import time
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from openai import OpenAI

# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found in .env")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Strong Mac-Compatible Headers ---
CUSTOM_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}

# --- Fetch last 5 games (2025-26) ---
def get_last_5_games(player_name, retries=3):
    for attempt in range(retries):
        try:
            nba_players = players.find_players_by_full_name(player_name)
            if not nba_players:
                return None, f"Could not find '{player_name}'."

            player_id = nba_players[0]["id"]
            full_name = nba_players[0]["full_name"]

            # small delay to reduce rate limiting
            time.sleep(1)

            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season="2025-26",
                headers=CUSTOM_HEADERS,
                timeout=15
            )

            df = gamelog.get_data_frames()[0]

            if df.empty:
                return full_name, pd.DataFrame()

            subset = df[
                ["GAME_DATE", "MATCHUP", "WL", "PTS", "REB", "AST", "FG3M", "MIN"]
            ].head(5)

            return full_name, subset

        except Exception:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                return None, "NBA API blocked or timed out after multiple attempts."

# --- AI Extraction ---
def extract_details_with_ai(user_input):
    prompt = f'Return ONLY JSON with keys: "player", "stat", "line", "direction". Input: "{user_input}"'

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = re.sub(r"```(?:json)?\n?|\n?```", "", content).strip()

        return json.loads(content)

    except:
        return None

# --- Regex Fallback ---
def fallback_extract(message):
    match = re.search(r"([A-Za-z ]+) (over|under) ([0-9]+\.?[0-9]*)", message, re.I)

    if match:
        return {
            "player": match.group(1).strip(),
            "direction": match.group(2).lower(),
            "line": float(match.group(3)),
            "stat": "Points",
        }

    return {
        "player": "",
        "stat": "Points",
        "line": 20.5,
        "direction": "over",
    }

# --- Main Chat Logic ---
def chat_handler(message):
    details = extract_details_with_ai(message)

    if not details or not details.get("player"):
        details = fallback_extract(message)

    p_name = details.get("player", "Unknown")
    stat_type = details.get("stat", "Points")
    line = details.get("line", 20.5)
    direction = details.get("direction", "over")

    real_name, stats_df = get_last_5_games(p_name)

    if not isinstance(stats_df, pd.DataFrame):
        return f"⚠️ {stats_df if stats_df else real_name}"

    if stats_df.empty:
        return f"⚠️ Found {real_name}, but no logs yet for 2025-26."

    formatted_stats = stats_df.to_string(index=False)

    analysis_prompt = f"""
    Analyze {real_name} relative to {line} {stat_type} ({direction})
    using the last 5 games:

    {formatted_stats}
    """

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.7,
        )

        analysis = res.choices[0].message.content

    except:
        analysis = "AI analysis unavailable. Showing raw data below."

    return f"{analysis}\n\n**Performance Data for {real_name}:**\n\n```\n{formatted_stats}\n```"

# --- Gradio Respond (Modern Messages Format) ---
def respond(message, chat_history):
    chat_history = chat_history or []

    bot_message = chat_handler(message)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})

    return "", chat_history

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange")) as demo:
    gr.Markdown("# 🚀 Agentic-Court: NBA Player Analyst")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(
        label="Query Player Performance",
        placeholder="E.g. 'Steph Curry over 4.5 3 pointers'",
    )
    clear = gr.Button("Clear History")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()