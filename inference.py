import os
import time
import httpx
import json
from openai import OpenAI

LLM_BASE_URL = os.environ["API_BASE_URL"]
LLM_API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

CANDIDATE_URLS = [
    os.environ.get("ENV_BASE_URL"),
    "http://environment:8000",
    "http://env:8000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://host.docker.internal:8000"
]

CANDIDATE_URLS = [u.rstrip("/") for u in CANDIDATE_URLS if u]

def connect(client):
    for base in CANDIDATE_URLS:
        for _ in range(5):
            try:
                resp = client.post(f"{base}/reset", json={})
                resp.raise_for_status()
                return base, resp.json()
            except:
                time.sleep(2)
    return None, None

def main():
    print("[START] Initializing SRE Agent")
    
    client = httpx.Client(timeout=60.0)
    base_url, state = connect(client)
    
    if not state:
        raise RuntimeError("Could not connect to the environment server.")
    
    print(f"[CONNECTED] {base_url}")
    
    llm_client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )

    done = False
    step = 0

    while not done and step < 15:
        step += 1
        print(f"[STEP] Running step {step}")

        prompt = (
            f"You are an SRE agent. Current observation: {json.dumps(state.get('observation', {}))}. "
            "Output ONLY valid JSON representing your next action. "
            "Example: {\"action_type\": \"NO_OP\", \"params\": {}}"
        )
        
        completion = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        raw_response = completion.choices[0].message.content.strip()
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()
        
        try:
            action_payload = json.loads(raw_response)
        except:
            action_payload = {"action_type": "NO_OP", "params": {}}

        step_resp = client.post(f"{base_url}/step", json=action_payload)
        step_resp.raise_for_status()
        state = step_resp.json()
        done = state.get("done", False)

    print("[END] Episode finished")

if __name__ == "__main__":
    main()
