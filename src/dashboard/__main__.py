"""Allow running with: uv run python -m src.dashboard"""

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.dashboard.app import create_app

app = create_app()
app.run(debug=True, port=5000)
