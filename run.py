import uvicorn
from app.main import app  # Import your FastAPI/Flask instance directly

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)