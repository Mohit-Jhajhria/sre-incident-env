import os
import httpx
import json
from openai import OpenAI

# 1. The grader injects these specifically for the LLM
LLM_BASE_URL = os.getenv("API_BASE_URL")
LLM_API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "dummy_key")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# 2. Your FastAPI environment is guaranteed to be running here
ENV_BASE_URL = "http://localhost:8000"

def main():
    print("[START] Initializing SRE Agent")
    
    # Connect to your FastAPI environment
    client = httpx.Client(timeout=30.0)

    try:
        reset_resp = client.post(f"{ENV_BASE_URL}/reset", json={})
        reset_resp.raise_for_status()
        state = reset_resp.json()
        print(f"Environment reset successful. Task ID: {state.get('task_id')}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to /reset endpoint: {e}")
        return

    done = False
    step = 0

    # Initialize LLM exactly as the grader demands
    try:
        llm_client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY
        )
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
                
                # Clean up markdown formatting safely
                raw_response = raw_response.replace("```json", "").replace("```", "").strip()
                
                action_payload = json.loads(raw_response)
            except Exception as e:
                print(f"LLM decision failed, falling back to NO_OP: {e}")

        # Send the action to your environment
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
