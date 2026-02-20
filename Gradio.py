import os
import sys
import re
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import openai

# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found in .env")
    sys.exit(1)
openai.api_key = OPENAI_API_KEY

# --- Fetch last 5 games ---
def get_last_5_games(player_name):
    nba_players = players.find_players_by_full_name(player_name)
    if not nba_players:
        return None, f"Could not find '{player_name}'. Try a full name like 'Stephen Curry'."
    player_id = nba_players[0]['id']
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2025-26')
        df = gamelog.get_data_frames()[0]
        if df.empty:
            return None, "No games played in the 2025-26 season yet."
        subset = df[['GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'REB', 'AST', 'FG3M', 'MIN']].head(5)
        return nba_players[0]['full_name'], subset
    except Exception as e:
        return None, f"NBA API Error: {e}"

# --- Fallback parser with over/under ---
def fallback_extract(message):
    """
    Parse "Player over/under 25.5 points" style messages
    """
    match = re.match(
        r"([A-Za-z ]+) (over|under) ([0-9]+\.?[0-9]*) (points|rebounds|assists|3 pointers)?",
        message,
        re.I
    )
    if match:
        player = match.group(1).strip()
        direction = match.group(2).lower()   # over or under
        line = float(match.group(3))
        stat = match.group(4).capitalize() if match.group(4) else "Points"
        return {"player": player, "stat": stat, "line": line, "direction": direction}
    return {"player": "", "stat": "Points", "line": 20.5, "direction": "over"}

# --- GPT-4o-mini AI extraction ---
def extract_details_with_ai(user_input):
    """
    Ask GPT-4o-mini to extract player, stat, line, direction in JSON
    """
    prompt = f"""
Extract NBA betting details from the following user input.
Return a JSON with keys: "player", "stat", "line" (number), "direction" ("over" or "under").
Default stat: Points, default line: 20.5.

User input: "{user_input}"
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        import json
        text = response.choices[0].message.content
        data = json.loads(text)
        return data
    except Exception as e:
        print(f"AI Extraction Error: {e}")
        return None

# --- Chat handler ---
def chat_handler(message, history):
    # Step 1: AI extraction
    details = extract_details_with_ai(message)

    # Step 2: Fallback parser if AI fails
    if not details or len(details.get("player", "")) < 2:
        details = fallback_extract(message)

    player_name = details['player']
    stat_type = details['stat']
    line = details['line']
    direction = details.get("direction", "over")

    # Step 3: Fetch NBA data
    real_name, stats_df = get_last_5_games(player_name)
    if stats_df is None:
        return f"⚠️ {real_name}"

    # Step 4: AI analysis with fallback
    analysis_prompt = f"""
You are a professional sports analyst.
Predict {direction.upper()} or UNDER {line} {stat_type} for {real_name} based on these 5 games:
{stats_df.to_string(index=False)}
Respond with a short recommendation.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.7
        )
        analysis = response.choices[0].message.content
    except Exception as e:
        # fallback simple average
        avg_column = {"Points":"PTS","Rebounds":"REB","Assists":"AST","3 Pointers":"FG3M"}.get(stat_type,"PTS")
        avg = stats_df[avg_column].mean()
        if direction == "over":
            if avg > line:
                analysis = f"{real_name} is likely to go OVER {line} {stat_type} (avg {avg:.1f})."
            else:
                analysis = f"{real_name} might go UNDER {line} {stat_type} (avg {avg:.1f})."
        else:  # direction == under
            if avg < line:
                analysis = f"{real_name} is likely to go UNDER {line} {stat_type} (avg {avg:.1f})."
            else:
                analysis = f"{real_name} might go OVER {line} {stat_type} (avg {avg:.1f})."

    return f"{analysis}\n\n**Data for {real_name}:**\n\n{stats_df.to_string(index=False)}"

# --- Gradio respond function ---
def respond(message, chat_history):
    response = chat_handler(message, chat_history)
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    return chat_history, ""  # clear input box

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Agentic-Court: NBA Predictor (GPT-4o-mini)")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Ask something like: 'LeBron over 25.5 points'")
    submit_btn = gr.Button("Send")

    submit_btn.click(respond, [user_input, chatbot], [chatbot, user_input])
    user_input.submit(respond, [user_input, chatbot], [chatbot, user_input])

# Launch with theme
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(primary_hue="orange"))
