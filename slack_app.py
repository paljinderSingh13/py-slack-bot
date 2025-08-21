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
def handle_message_events(message, say):
    user_text = message.get("text", "").strip()
    if not user_text:
        say("Please send a valid message.")
        return

    # Step 1: Search FAISS
    results = search_faq(user_text, top_k=1)

    if results and len(results) > 0:
        top_result = results[0]
        distance = top_result.get("score", None)
        print(f"DEBUG: distance={distance}, question={top_result.get('question')}")

        if distance is not None and distance <= DISTANCE_THRESHOLD:
            if top_result.get("image_url"): 
                say({
                    "blocks": [
                        {
                            "type": "image",
                            "image_url": top_result["image_url"],
                            "alt_text": "FAQ Image"
                        }
                    ]
                })
            answer = top_result.get("answer")

            # Case 1: Answer is a simple string
            if isinstance(answer, str):
                say(answer)

            # Case 2: Answer is a list of steps with text + images
            elif isinstance(answer, list):
                for step in answer:
                    blocks = []

                    # If step has text, add section block
                    if "text" in step and step["text"]:
                        blocks.append({
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": step["text"]
                            }
                        })

                    # If step has image, add image block
                    if "image" in step and step["image"]:
                        blocks.append({
                            "type": "image",
                            "image_url": step["image"],
                            "alt_text": "FAQ Image"
                        })

                    if blocks:
                        say({"blocks": blocks})

            return 

    # Step 2: No match → classify via GPT
    category = classify_category(user_text)

    if category and category in CATEGORY_CONTACTS:
        contact = CATEGORY_CONTACTS[category]

        say(f"I currently don't know that answer, but {contact} can assist you.")

    else:
        category = classify_category_gpt(user_text)
        if category:
            say(f"I currently don't know that answer, but {CATEGORY_CONTACTS[category]} can assist you.")
        else:
            say(f"I currently don't know that answer.")
   

# Flask app for Slack events
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

if __name__ == "__main__":
    flask_app.run(port=3000)
