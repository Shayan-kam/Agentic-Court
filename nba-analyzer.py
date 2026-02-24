import os
import sys
import re
import json
import time
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from nba_api.stats.static import players
from nba_api.stats.endpoints import (
    playergamelog,
    playernextngames,
    playerdashboardbygeneralsplits
)
from openai import OpenAI


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found in .env")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# --- NBA Headers ---
CUSTOM_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

CURRENT_SEASON = "2025-26"  # Update each season


# --- Matchup Context ---
def get_matchup_context(player_id, player_team_abbr):
    try:
        next_games = playernextngames.PlayerNextNGames(
            player_id=player_id,
            headers=CUSTOM_HEADERS
        ).get_data_frames()[0]

        if next_games.empty:
            return "Unknown", pd.DataFrame()

        game = next_games.iloc[0]
        visitor = game['VISITOR_TEAM_ABBREVIATION']
        home = game['HOME_TEAM_ABBREVIATION']
        next_opp = visitor if home == player_team_abbr else home

        vs_dash = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
            player_id=player_id,
            per_mode_detailed='PerGame',
            headers=CUSTOM_HEADERS
        )

        dfs = vs_dash.get_data_frames()

        opponent_df = None

        for df in dfs:
            # Opponent splits table contains BOTH GROUP_VALUE and GP
            if "GROUP_VALUE" in df.columns and "GP" in df.columns:
                # And group values are team abbreviations (3 letters)
                if df["GROUP_VALUE"].str.len().max() == 3:
                    opponent_df = df
                    break

        if opponent_df is None:
            return next_opp, pd.DataFrame()

        matchup_stats = opponent_df[opponent_df["GROUP_VALUE"] == next_opp]

        return next_opp, matchup_stats

    except Exception as e:
        print(f"⚠️ Matchup fetch error: {e}")
        return "Unknown", pd.DataFrame()


# --- Combined Fetcher ---
def get_player_full_analysis_data(player_name, retries=4):
    for attempt in range(retries):
        try:
            nba_players = players.find_players_by_full_name(player_name)
            if not nba_players:
                return None, f"Could not find '{player_name}'.", None, None

            p_id = nba_players[0]["id"]
            full_name = nba_players[0]["full_name"]

            time.sleep((attempt * 3) + 1)

            gamelog = playergamelog.PlayerGameLog(
                player_id=p_id,
                season=CURRENT_SEASON,
                season_type_all_star="Regular Season",
                headers=CUSTOM_HEADERS
            )

            full_log = gamelog.get_data_frames()[0]

            if full_log.empty:
                return full_name, pd.DataFrame(), "Unknown", pd.DataFrame()

            l5_df = full_log.head(5)

            matchup_str = l5_df.iloc[0]["MATCHUP"]
            current_team = matchup_str.split(" ")[0]

            next_opp, hist_df = get_matchup_context(p_id, current_team)

            return full_name, l5_df, next_opp, hist_df

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                return None, f"Error: {e}", None, None


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
    except Exception as e:
        print(f"AI extraction error: {e}")
        return None


# --- Regex Fallback ---
def fallback_extract(message):
    match = re.search(r"([A-Za-z ]+) (over|under) ([0-9]+\.?[0-9]*)", message, re.I)
    return {
        "player": match.group(1).strip() if match else "",
        "direction": match.group(2).lower() if match else "over",
        "line": float(match.group(3)) if match else 20.5,
        "stat": "Points",
    }


# --- Main Chat Logic ---
def chat_handler(message):
    details = extract_details_with_ai(message) or fallback_extract(message)
    p_name = details.get("player", "Unknown")
    stat_type = details.get("stat", "Points")
    line = details.get("line", 20.5)
    direction = details.get("direction", "over")

    real_name, l5_df, next_opp, hist_df = get_player_full_analysis_data(p_name)

    if l5_df is None or isinstance(l5_df, str):
        return f"⚠️ {l5_df if l5_df else 'Error'}"

    l5_stats = l5_df[["GAME_DATE", "MATCHUP", "WL", "PTS", "REB", "AST", "MIN"]].to_string(index=False)

    if not hist_df.empty and all(col in hist_df.columns for col in ["GP", "PTS", "REB", "AST", "FG_PCT"]):
        vs_stats = hist_df[["GP", "PTS", "REB", "AST", "FG_PCT"]].to_string(index=False)
    else:
        vs_stats = f"No career data vs {next_opp}."

    analysis_prompt = (
        f"Analyze {real_name} for {line} {stat_type} ({direction}) vs {next_opp}.\n\n"
        f"Recent:\n{l5_stats}\n\n"
        f"Career vs Opponent:\n{vs_stats}"
    )

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.7,
        )
        analysis = res.choices[0].message.content
    except Exception as e:
        print(f"AI analysis error: {e}")
        analysis = "AI analysis unavailable."

    return f"### 🏀 Analysis: {real_name} vs. {next_opp}\n\n{analysis}\n\n**Raw Data (Last 5):**\n```\n{l5_stats}\n```"


# --- Gradio Respond (Dictionary Format) ---
def respond(message, chat_history):
    if chat_history is None:
        chat_history = []

    bot_message = chat_handler(message)

    chat_history.append({
        "role": "user",
        "content": message
    })

    chat_history.append({
        "role": "assistant",
        "content": bot_message
    })

    return "", chat_history


with gr.Blocks(theme=gr.themes.Default(primary_hue="orange")) as demo:
    gr.Markdown("# 🚀 Agentic-Court: Predictive NBA Matchup Analyst")

    chatbot = gr.Chatbot(value=[])
    msg = gr.Textbox(
        label="Query Player Performance",
        placeholder="E.g. 'Steph Curry over 4.5 3 pointers'",
    )
    clear = gr.Button("Clear History")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)


if __name__ == "__main__":
    demo.launch()