from flask import Flask, jsonify, render_template
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route("/api/test")
def test_api():
    """Test API endpoint."""
    return jsonify({
        "success": True,
        "message": "API is working!"
    })

if __name__ == "__main__":
    logger.info("Starting simple Flask server on http://0.0.0.0:5001")
    # Run the app with explicit host and port, binding to all interfaces
    app.run(debug=True, port=5001, host="0.0.0.0")