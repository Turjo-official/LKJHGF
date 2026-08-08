import os
from flask import Flask, request, jsonify
from duckduckgo_search import DDGS
from google import genai

app = Flask(__name__)

# Initialize Gemini Client
# Ensure GEMINI_API_KEY is set in your environment
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or "prompt" not in data:
        return jsonify({"response": "No prompt provided"}), 400

    user_prompt = data["prompt"]

    # 1. Fetch search context
    search_context = ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(user_prompt, max_results=3))
            for r in results:
                search_context += f"Title: {r['title']}\nSnippet: {r['body']}\n\n"
    except Exception:
        search_context = "No live search context available."

    # 2. Construct the prompt for Gemini
    full_prompt = f"""
    You are a helpful assistant. Use the following live web data to answer the user's question.
    
    Live Web Data:
    {search_context}
    
    User Question: {user_prompt}
    """

    # 3. Call Gemini API
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
            config={
                "system_instruction": "Provide concise, direct answers based on provided search context.",
                "max_output_tokens": 300,
            }
        )
        ai_reply = response.text
    except Exception as e:
        ai_reply = f"Gemini Error: {str(e)}"

    return jsonify({"response": ai_reply})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)