import os
import time
import json
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

API_KEY = os.getenv("API_KEY")
# Case sensitivity: Some NIM versions prefer lowercase 'pro'
DEFAULT_MODEL = "deepseek-ai/deepseek-v4-pro"
NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

app = Flask(__name__)
CORS(app)

@app.route('/', methods=["GET"])
def health_check():
    return "NVIDIA Pro-Ready Proxy Online", 200

@app.route('/v1/chat/completions', methods=["POST"])
@app.route('/chat/completions', methods=["POST"])
def handle_proxy():
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        current_model = data.get("model", DEFAULT_MODEL)

        payload = {
            "model": current_model,
            "messages": messages,
            "stream": True, # Force streaming for Pro
            "temperature": data.get("temperature", 0.9),
            "max_tokens": data.get("max_tokens", 4096),
            "chat_template_kwargs": {
                "enable_thinking": True,
                "thinking": True
            }
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        def stream_response():
            # --- THE HEARTBEAT HACK ---
            # Send an SSE comment immediately. Standard RP tools (SillyTavern/Janitor) 
            # ignore lines starting with ':', but the HF Gateway sees this as "data" 
            # and keeps the connection alive.
            yield ": connection established\n\n"
            yield ": model is thinking...\n\n"

            try:
                # Use a longer read timeout for Pro (600s)
                with requests.post(NIM_ENDPOINT, headers=headers, json=payload, stream=True, timeout=(15, 600)) as r:
                    if r.status_code != 200:
                        yield f"data: {json.dumps({'error': 'NVIDIA Error', 'details': r.text})}\n\n"
                        return

                    for chunk in r.iter_lines():
                        if chunk:
                            # Pass through the NVIDIA tokens
                            yield chunk.decode('utf-8') + "\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(stream_with_context(stream_response()), content_type='text/event-stream')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
