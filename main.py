import os
import uvicorn
from src.api.webhook import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api.webhook:app", host="0.0.0.0", port=port)
