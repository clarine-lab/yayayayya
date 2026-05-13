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
            # Initial heartbeat
            yield ": connection established\n\n"
            
            try:
                with requests.post(NIM_ENDPOINT, headers=headers, json=payload, stream=True, timeout=(15, 600)) as r:
                    if r.status_code != 200:
                        err_snippet = r.raw.read(500).decode('utf-8', 'ignore')
                        yield f"data: {json.dumps({'error': 'NVIDIA Error', 'details': err_snippet})}\n\n"
                        return

                    for line in r.iter_lines():
                        if not line:
                            continue
                            
                        decoded_line = line.decode('utf-8').strip()

                        # 1. Catch the DONE signal and terminate cleanly
                        if "[DONE]" in decoded_line:
                            yield "data: [DONE]\n\n"
                            break

                        # 2. Process only properly formatted SSE data lines
                        if decoded_line.startswith("data: "):
                            json_str = decoded_line[6:] # Strip "data: "
                            
                            try:
                                data_obj = json.loads(json_str)
                                choices = data_obj.get("choices", [])
                                
                                # Filter out empty choices (NVIDIA's trailing usage chunks)
                                if not choices:
                                    continue
                                    
                                delta = choices[0].get("delta", {})
                                finish_reason = choices[0].get("finish_reason")
                                
                                # Extract the actual story content (if it exists)
                                content = delta.get("content")
                                
                                # If there is no story content yet, the model is still thinking
                                if content is None and finish_reason is None:
                                    # Send a heartbeat so JanitorAI doesn't time out
                                    yield ": heartbeat\n\n"
                                    continue
                                
                                # Rebuild a clean 'delta' that JanitorAI can easily read
                                clean_delta = {}
                                if content is not None:
                                    clean_delta["content"] = content
                                    
                                # Rebuild the chunk to perfectly match standard OpenAI format
                                clean_chunk = {
                                    "id": data_obj.get("id", "chatcmpl-proxy"),
                                    "object": "chat.completion.chunk",
                                    "created": data_obj.get("created", 0),
                                    "model": data_obj.get("model", current_model),
                                    "choices": [{
                                        "index": choices[0].get("index", 0),
                                        "delta": clean_delta,
                                        "finish_reason": finish_reason
                                    }]
                                }
                                
                                yield f"data: {json.dumps(clean_chunk)}\n\n"
                                
                            except json.JSONDecodeError:
                                # Fallback: if NVIDIA sends weird raw text, just pass it through safely
                                yield f"{decoded_line}\n\n"

            except requests.exceptions.Timeout:
                yield f"data: {json.dumps({'error': 'NVIDIA Timeout', 'details': 'Model took too long.'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': 'Proxy Loop Error', 'details': str(e)})}\n\n"

            except requests.exceptions.Timeout:
                yield f"data: {json.dumps({'error': 'NVIDIA Timeout', 'details': 'The model took too long.'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': 'Proxy Loop Error', 'details': str(e)})}\n\n"

        return Response(stream_with_context(stream_response()), content_type='text/event-stream')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
