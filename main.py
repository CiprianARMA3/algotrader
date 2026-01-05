import os
import sys

# Add the backend and backend/app directories to the Python path
# This ensures that internal imports like 'from app.api...' work correctly
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "backend"))
sys.path.append(os.path.join(base_path, "backend", "app"))

# Import the FastAPI app from your existing backend code
from backend.app.main import app

# This allows the app to run if 'python main.py' is called directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
