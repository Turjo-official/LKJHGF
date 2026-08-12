import os
import sys
from flask import Flask, request, jsonify

# Compatibility check for renamed ddgs package
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from google import genai

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

@app.route('/', methods=['GET', 'HEAD'])
def home():
    return jsonify({
        "status": "API is online",
        "api_key_configured": bool(api_key)
    }), 200

@app.route('/chat', methods=['POST'])
def chat():
    if not api_key or not client:
        print("ERROR: GEMINI_API_KEY is not set in Render Environment Variables!", file=sys.stderr)
        return jsonify({"error": "GEMINI_API_KEY missing on server"}), 500

    # Parse payload (Supports standard JSON OR raw string from ESP32)
    user_prompt = ""
    try:
        data = request.get_json(silent=True)
        if isinstance(data, dict) and "prompt" in data:
            user_prompt = data["prompt"]
        else:
            # Fallback if ESP32 sends raw text string directly
            user_prompt = request.get_data(as_text=True).strip()
    except Exception as parse_err:
        print(f"Payload parse warning: {parse_err}", file=sys.stderr)

    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Search Attempt (Safely catches cloud IP blocks)
    search_context = ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(user_prompt, max_results=2))
            for r in results:
                search_context += f"Title: {r['title']}\nSnippet: {r['body']}\n\n"
    except Exception as search_err:
        print(f"DuckDuckGo search skipped: {search_err}", file=sys.stderr)

    # Call Gemini API
    try:
        full_query = f"Live Web Data:\n{search_context}\n\nQuestion: {user_prompt}" if search_context else user_prompt
        
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=full_query,
            config={
                "system_instruction": "You are an assistant for an ESP32 microcontroller. Provide short, concise answers under 150 characters without markdown formatting.",
                "max_output_tokens": 100,
            }
        )
        return jsonify({"response": response.text}), 200

    except Exception as e:
        print(f"Gemini API Exception: {e}", file=sys.stderr)
        return jsonify({"error": f"Gemini Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
