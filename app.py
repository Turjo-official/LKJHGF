import os
from flask import Flask, request, jsonify
from duckduckgo_search import DDGS
from google import genai

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "API is online",
        "api_key_configured": bool(api_key)
    }), 200

@app.route('/chat', methods=['POST'])
def chat():
    if not client:
        return jsonify({"error": "GEMINI_API_KEY is not set in Render environment variables."}), 500

    data = request.json or {}
    user_prompt = data.get("prompt", "")

    if not user_prompt:
        return jsonify({"error": "No prompt provided. Send JSON: {'prompt': 'your question'}"}), 400

    search_context = ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(user_prompt, max_results=3))
            for r in results:
                search_context += f"Title: {r['title']}\nSnippet: {r['body']}\n\n"
    except Exception as search_err:
        search_context = f"Search unavailable: {str(search_err)}"

    try:
        full_query = f"Live Web Data:\n{search_context}\n\nQuestion: {user_prompt}"
        
        # Exact valid model name required here
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=full_query,
            config={
                "system_instruction": "You are a helpful assistant. Provide concise, direct answers.",
                "max_output_tokens": 300,
            }
        )
        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({"error": f"Gemini API Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
