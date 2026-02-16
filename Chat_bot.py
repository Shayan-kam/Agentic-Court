import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel

# 1. Setup and Authentication
load_dotenv(override=True)
client = OpenAI() # Uses OPENAI_API_KEY from .env

# 2. Load Data
# Ensure these files exist in a folder named 'me'
reader = PdfReader("me/linkedin.pdf")
linkedin_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin_text += text

with open("me/summary.txt", "r", encoding="utf-8") as f:
    summary_text = f.read()

name = "Shayan Kamalesh"

# 3. Define System Prompts
system_prompt = f"""
You are acting as {name}. You are answering questions on {name}'s website, 
particularly questions related to {name}'s career, background, skills and experience. 
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. 
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. 
Be professional and engaging, as if talking to a potential client or future employer. 
If you don't know the answer, say so.

## Summary:
{summary_text}

## LinkedIn Profile:
{linkedin_text}

With this context, please chat with the user, always staying in character as {name}.
"""

evaluator_system_prompt = f"""
You are an evaluator that decides whether a response to a question is acceptable. 
The Agent is playing the role of {name}. 
The Agent must be professional and engaging. 
Evaluate if the latest response is high quality and accurate based on the provided context.

## Summary:
{summary_text}

## LinkedIn Profile:
{linkedin_text}
"""

# 4. Evaluation Logic
class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

def evaluate(reply, message, history) -> Evaluation:
    eval_user_content = f"""
    Conversation History: {history}
    User's latest message: {message}
    Agent's latest response: {reply}
    Please evaluate the response.
    """
    
    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {"role": "user", "content": eval_user_content}
    ]
    
    # Using structured outputs (parsing into the Evaluation class)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=Evaluation
    )
    return completion.choices[0].message.parsed

def rerun(reply, message, history, feedback):
    updated_prompt = system_prompt + f"""
    \n\n## Previous answer rejected
    Your last attempt was rejected for: {feedback}
    Your attempted answer was: {reply}
    Please try again, addressing the feedback.
    """
    messages = [{"role": "system", "content": updated_prompt}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content

# 5. Main Chat Function
def chat(message, history):
    # Clean history for non-OpenAI providers if necessary
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    
    # Generate initial response
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    reply = response.choices[0].message.content

    # Evaluate response
    evaluation = evaluate(reply, message, history)
    
    if evaluation.is_acceptable:
        print("Passed evaluation")
        return reply
    else:
        print(f"Failed evaluation: {evaluation.feedback}")
        # Retry once with feedback
        return rerun(reply, message, history, evaluation.feedback)

# 6. Launch UI
if __name__ == "__main__":
    gr.ChatInterface(chat, type="messages").launch()