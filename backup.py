  import os
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request
from dotenv import load_dotenv
from openai import OpenAI  # OpenAI Python SDK v0.27.0+
from slack_sdk import WebClient

# Import your existing FAQ search function
from search_faq import search_faq
# Load environment variables
load_dotenv()

# Initialize Slack Bolt app
app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"]
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Threshold for FAQ search confidence (adjust based on your index)
# SIMILARITY_THRESHOLD = 0.25  # Example threshold

def call_gpt4_response(user_text: str) -> str:
    """
    Call GPT-4 to generate a helpful response or classify the message topic.
    """
    try:
        # Example prompt for answering or classification (you can customize)
        system_prompt = (
            "You are a helpful assistant. Answer the user's question clearly and concisely."
        )

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling GPT-4: {e}")
        return "Sorry, I'm having trouble responding right now."

CATEGORY_CONTACTS = {
    "checkout": "@Tom Richmond",
    "client_services": "@Adam Hepper"
}

def classify_category(question):
    """
    Very simple keyword-based classifier (can be replaced with GPT if needed).
    """
    q_lower = question.lower()
    if "checkout" in q_lower or "payment" in q_lower or "terms" in q_lower:
        return "checkout"
    elif "campaign" in q_lower or "subscription" in q_lower or "client" in q_lower or "services" in q_lower:
        return "client_services"
    else:
        return None


DISTANCE_THRESHOLD = 1.1  # Adjust this threshold based on your tests

SIMILARITY_THRESHOLD = 0.80  # Adjust as needed

def classify_category_gpt(query):
    prompt = f"""
    Classify the following customer question into one of these categories:
    - checkout
    - client services
    - unknown

    Question: "{query}"

    Only respond with one category name exactly.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # cheaper & faster than full GPT-4
        messages=[
            {"role": "system", "content": "You are a classification assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    category = response.choices[0].message.content.strip().lower()
    return category if category in CATEGORY_CONTACTS else None


@app.message("")
def handle_message_events(message, say, client: WebClient):
    user_text = message.get("text", "").strip()
    if not user_text:
        say("Please send a valid message.")
        return

    channel_id = message["channel"]

    # Step 1: Search FAISS for matching FAQ
    results = search_faq(user_text, top_k=1)

    if results and len(results) > 0:
        top_result = results[0]
        distance = top_result.get("score", None)
        print(f"DEBUG: distance={distance}, question={top_result.get('question')}")

        if distance is not None and distance <= DISTANCE_THRESHOLD:
            answer = top_result.get("answer")

            # STEP 1️⃣ → Post the FIRST message (main reply, outside thread)
            if isinstance(answer, str):
                main_message = client.chat_postMessage(
                    channel=channel_id,
                    text=answer
                )
                return

            elif isinstance(answer, list):
                # ✅ Post a general intro message (main reply)
                main_message = client.chat_postMessage(
                    channel=channel_id,
                    text="To solve this, follow these steps:"
                )

                # Extract parent_ts for threading
                parent_ts = main_message["ts"]

                # STEP 2️⃣ → Send each step as a THREAD reply
                for step in answer:
                    blocks = []

                    # If step has text → add section block
                    if "text" in step and step["text"]:
                        blocks.append({
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": step["text"]
                            }
                        })

                    # If step has image → add image block
                    if "image" in step and step["image"]:
                        blocks.append({
                            "type": "image",
                            "image_url": step["image"],
                            "alt_text": "FAQ Image"
                        })

                    if blocks:
                        client.chat_postMessage(
                            channel=channel_id,
                            thread_ts=parent_ts,
                            blocks=blocks
                        )

                return  # ✅ Done

    # Step 2: No match → fallback to GPT classification
    category = classify_category(user_text)

    if category and category in CATEGORY_CONTACTS:
        contact = CATEGORY_CONTACTS[category]
        client.chat_postMessage(
            channel=channel_id,
            text=f"I currently don't know that answer, but {contact} can assist you."
        )
    else:
        category = classify_category_gpt(user_text)
        if category:
            client.chat_postMessage(
                channel=channel_id,
                text=f"I currently don't know that answer, but {CATEGORY_CONTACTS[category]} can assist you."
            )
        else:
            client.chat_postMessage(
                channel=channel_id,
                text="I currently don't know that answer."
            )
   

# Flask app for Slack events
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

if __name__ == "__main__":
    flask_app.run(port=3000)

  
  
  
  
  
  
  
  
  
  
  
  
  
  {
    "question": "How do I add pixels?",
    "answer": "Go to Stores",
    "text_1": "Go to Stores",
    "image_1": "https://recstep.com/bot/go_to_store.png",
    "text_2": "Click on the title:",
    "image_2": "https://recstep.com/bot/title.png",
    "text_3": "Go to “Conversion tracking” tab",
    "image_3": "https://recstep.com/bot/title.png/conversion_tracking.png",
    "text_4": "Click on “add tracking” include all relevant information and click on “add”",
    "image_4": "https://recstep.com/bot/title.png/add_tracking_form.png"
  },


import os
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request
from dotenv import load_dotenv
from openai import OpenAI  # OpenAI Python SDK v0.27.0+

# Import your existing FAQ search function
from search_faq import search_faq
# Load environment variables
load_dotenv()

# Initialize Slack Bolt app
app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"]
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Threshold for FAQ search confidence (adjust based on your index)
SIMILARITY_THRESHOLD = 0.25  # Example threshold

def call_gpt4_response(user_text: str) -> str:
    """
    Call GPT-4 to generate a helpful response or classify the message topic.
    """
    try:
        # Example prompt for answering or classification (you can customize)
        system_prompt = (
            "You are a helpful assistant. Answer the user's question clearly and concisely."
        )

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling GPT-4: {e}")
        return "Sorry, I'm having trouble responding right now."

CATEGORY_CONTACTS = {
    "checkout": "@Tom Richmond",
    "client_services": "@Adam Hepper"
}

def classify_category(question):
    """
    Very simple keyword-based classifier (can be replaced with GPT if needed).
    """
    q_lower = question.lower()
    if "checkout" in q_lower or "payment" in q_lower or "terms" in q_lower:
        return "checkout"
    elif "campaign" in q_lower or "subscription" in q_lower or "client" in q_lower or "services" in q_lower:
        return "client_services"
    else:
        return None


DISTANCE_THRESHOLD = 1.1  # Adjust this threshold based on your tests

@app.message("")
def handle_message_events(message, say):
    user_text = message.get("text", "").strip()
    if not user_text:
        say("Please send a valid message.")
        return

    results = search_faq(user_text, top_k=1)
    if results and len(results) > 0:
        top_result = results[0]
        distance = top_result.get("score", None)
        print(f"DEBUG: distance={distance}, question={top_result.get('question')}")

        if distance is not None and distance <= DISTANCE_THRESHOLD:
            say(top_result["answer"])
            return

    # Fallback to GPT-4 if no good FAQ match
    category = classify_category(user_text)

    if category and category in CATEGORY_CONTACTS:
        contact = CATEGORY_CONTACTS[category]
        say(f"I currently don't know that answer, but {contact} can assist you.")
    else:
        say("I currently don't know that answer.")
    # gpt_response = call_gpt4_response(user_text)
    # say(gpt_response)

    # If no good FAQ match found, fallback GPT-4
   

# Flask app for Slack events
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

if __name__ == "__main__":
    flask_app.run(port=3000)
