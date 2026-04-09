import os
import httpx
import json
from openai import OpenAI

LLM_BASE_URL = os.environ["API_BASE_URL"]
LLM_API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

ENV_BASE_URL = "http://localhost:8000"

def main():
    print("[START] Initializing SRE Agent")
    
    client = httpx.Client(timeout=30.0)

    try:
        reset_resp = client.post(f"{ENV_BASE_URL}/reset", json={})
        reset_resp.raise_for_status()
        state = reset_resp.json()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    llm_client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )

    done = False
    step = 0

    while not done and step < 20:
        step += 1
        print(f"[STEP] {step}")

        prompt = (
            "You are an SRE agent. "
            f"Observation: {json.dumps(state.get('observation', {}))}. "
            "Return ONLY valid JSON with action_type and params. "
            "Example: {\"action_type\":\"NO_OP\",\"params\":{}}"
        )

        try:
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = completion.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            action_payload = json.loads(raw)
        except Exception as e:
            print(f"[ERROR] LLM failed: {e}")
            action_payload = {"action_type": "NO_OP", "params": {}}

        try:
            step_resp = client.post(f"{ENV_BASE_URL}/step", json=action_payload)
            step_resp.raise_for_status()
            state = step_resp.json()
            done = state.get("done", False)
        except Exception as e:
            print(f"[ERROR] Step failed: {e}")
            break

    print("[END] Episode finished")

if __name__ == "__main__":
    main()
