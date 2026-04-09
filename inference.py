import os
import httpx
import json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_key")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

def main():
    print("[START] Initializing SRE Agent")
    
    # Connect to your FastAPI environment
    client = httpx.Client(timeout=30.0)

    try:
        reset_resp = client.post(f"{API_BASE_URL}/reset", json={})
        reset_resp.raise_for_status()
        state = reset_resp.json()
        print(f"Environment reset successful. Task ID: {state.get('task_id')}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to /reset endpoint: {e}")
        return

    done = False
    step = 0

    # Initialize LLM
    try:
        llm_client = OpenAI(api_key=API_KEY)
        use_llm = True
    except Exception as e:
        print(f"LLM initialization skipped: {e}")
        use_llm = False

    # Run the episode
    while not done and step < 15:
        step += 1
        print(f"[STEP] Running step {step}")

        # Default fallback action
        action_payload = {"action_type": "NO_OP", "params": {}}

        if use_llm:
            prompt = (
                f"You are an SRE agent resolving a server incident. "
                f"Current observation: {json.dumps(state.get('observation', {}))}. "
                "Output ONLY valid JSON representing your next action. "
                "Example: {\"action_type\": \"NO_OP\", \"params\": {}}"
            )
            try:
                completion = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                raw_response = completion.choices[0].message.content.strip()
                
                # Clean up markdown formatting if the LLM adds it
                if raw_response.startswith("
http://googleusercontent.com/immersive_entry_chip/0
http://googleusercontent.com/immersive_entry_chip/1
