import os
import httpx
import json
from openai import OpenAI

LLM_BASE_URL = os.getenv("API_BASE_URL")
LLM_API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "dummy_key")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

ENV_BASE_URL = "http://localhost:8000"

def decide_action(state):
    obs = state.get("observation", {})
    text = json.dumps(obs).lower()

    if "error" in text or "fail" in text:
        return {"action_type": "RESTART_SERVICE", "params": {}}
    if "not running" in text or "stopped" in text:
        return {"action_type": "START_SERVICE", "params": {}}
    if "high cpu" in text or "high memory" in text:
        return {"action_type": "SCALE_UP", "params": {}}
    return {"action_type": "NO_OP", "params": {}}

def main():
    print("[START] Initializing SRE Agent")
    
    client = httpx.Client(timeout=30.0)

    try:
        reset_resp = client.post(f"{ENV_BASE_URL}/reset", json={})
        reset_resp.raise_for_status()
        state = reset_resp.json()
        print(f"Environment reset successful. Task ID: {state.get('task_id')}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to /reset endpoint: {e}")
        return

    try:
        llm_client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY
        )
        use_llm = True
    except Exception as e:
        print(f"LLM initialization skipped: {e}")
        use_llm = False

    done = False
    step = 0

    while not done and step < 20:
        step += 1
        print(f"[STEP] Running step {step}")

        action_payload = decide_action(state)

        if use_llm:
            try:
                prompt = (
                    f"You are an SRE agent resolving a server incident. "
                    f"Current observation: {json.dumps(state.get('observation', {}))}. "
                    "Output ONLY valid JSON representing your next action."
                )
                completion = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                raw_response = completion.choices[0].message.content.strip()
                raw_response = raw_response.replace("```json", "").replace("```", "").strip()
                action_payload = json.loads(raw_response)
            except Exception as e:
                print(f"LLM decision failed, using rule-based action: {e}")

        try:
            step_resp = client.post(f"{ENV_BASE_URL}/step", json=action_payload)
            step_resp.raise_for_status()
            state = step_resp.json()
            done = state.get("done", False)
        except Exception as e:
            print(f"[ERROR] Action step failed: {e}")
            break

    print("[END] Episode finished")

if __name__ == "__main__":
    main()
