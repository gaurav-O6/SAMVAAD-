"""Root entry point for the SAMVAAD Flask app."""

from templates.app import app


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SAMVAAD Backend is running!")
    print("  Open your browser at:  http://localhost:5000")
    print("  Mode: Landmark-based (maximum accuracy)")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
