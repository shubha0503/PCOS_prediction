# Static PCOS assessment frontend

This frontend uses plain HTML, CSS, and JavaScript. It calls the FastAPI service in the project root.

1. In the project root, start the API: `uvicorn api:app --reload --host 127.0.0.1 --port 8000`.
2. In this directory, serve the static files: `python -m http.server 5500`.
3. Open `http://127.0.0.1:5500` and use the API URL `http://127.0.0.1:8000` shown on the page.

The API URL can be changed in the page if the service is running elsewhere.
