import sys
import os
import uvicorn

# Point the validator to our actual API file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app

def main():
    """Entry point for the OpenEnv validator."""
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
