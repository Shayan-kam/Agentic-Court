import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel

# 1. Setup
load_dotenv(override=True)
client = OpenAI()

# 2. Load Data
reader = PdfReader("./LeBron_James_Career_Stats.pdf")
Player_stats = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        Player_stats += text

# 3. System Prompts
system_prompt = f"""
You are acting as a proffesional sports analyst. You are answering questions based on the PDF with a basektball
players seasonal infromation, 
particularly information related to Lebron career, background, skills and experience. 
Your responsibility is to represent a sports Analyst who decifiers through the infromation provided
and creates Mathematical and beyond calulations to prdecit Lebrons preformance for the next season. 
You are to provide what his precidted stats would be in a easily readable format. 
Be professional and engaging, as if talking to a client. 
If you don't know the answer, say so.


## Player Profile:
{Player_stats}

With this context, please chat with the user, always staying in character as a sports analyst.
"""

evaluator_system_prompt = f"""
You are an evaluator that decides whether a response to a question is acceptable. 
The Agent is playing the role of a sports analyst who predicts Lebrons prefromace for the next season
based on this infromation {Player_stats}. 
The Agent must be professional and concise. 
Evaluate if the latest response is high quality and accurate based on the provided context.


## Player Profile:
{Player_stats}
"""

# 4. Evaluation Logic
class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

def evaluate(reply, message, history) -> Evaluation:
    eval_user_content = f"History: {history}\nLatest: {message}\nResponse: {reply}"
    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {"role": "user", "content": eval_user_content}
    ]
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=Evaluation
    )
    return completion.choices[0].message.parsed

def rerun(reply, message, history, feedback):
    updated_prompt = system_prompt + f"\n\nReject feedback: {feedback}"
    messages = [{"role": "system", "content": updated_prompt}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content

# 5. Main Chat Function (REFORMATTED FOR COMPATIBILITY)
def chat(message, history):
    # This loop ensures the history works even if Gradio version is 'weird'
    formatted_history = []
    for turn in history:
        # Check if history is old [user, bot] format or new {role, content} format
        if isinstance(turn, dict):
            formatted_history.append(turn)
        else:
            formatted_history.append({"role": "user", "content": turn[0]})
            formatted_history.append({"role": "assistant", "content": turn[1]})
    
    messages = [{"role": "system", "content": system_prompt}] + formatted_history + [{"role": "user", "content": message}]
    
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    reply = response.choices[0].message.content

    evaluation = evaluate(reply, message, formatted_history)
    
    if evaluation.is_acceptable:
        return reply
    else:
        return rerun(reply, message, formatted_history, evaluation.feedback)

# 6. Launch UI (REMOVED 'type' ARGUMENT)
if __name__ == "__main__":
    # Removing 'type' avoids the TypeError entirely
    gr.ChatInterface(chat).launch()