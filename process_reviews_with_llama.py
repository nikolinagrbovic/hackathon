import os
import json
import requests

REVIEWS_FOLDER = "reviews"
OUTPUT_FILE = "processed_destinations.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def ask_llama(city_name, review_text):
    prompt = f"""
You are an AI travel emotion extractor.

Analyze this travel review text and return ONLY valid JSON.

City/destination name: {city_name}

Review text:
{review_text}

Return this exact JSON structure:
{{
  "name": "{city_name}",
  "country": "",
  "tags": [],
  "moods": [],
  "activities": [],
  "environment": [],
  "climate": "",
  "crowd": "",
  "budget": "",
  "pace": "",
  "best_for": []
}}

Rules:
- Fill country based on the city/destination name.
- Use short lowercase tags.
- climate can be: warm, cold, mild, tropical, dry, unknown.
- crowd can be: low, medium, high, unknown.
- budget can be: low, medium, high, unknown.
- pace can be: slow, balanced, energetic, unknown.
- Return JSON only.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()
    text = data["response"].strip()

    return json.loads(text)


def main():
    results = []

    for filename in os.listdir(REVIEWS_FOLDER):
        if filename.endswith(".txt"):
            city_name = filename.replace(".txt", "")

            file_path = os.path.join(REVIEWS_FOLDER, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                review_text = f.read()

            print(f"Processing {city_name}...")

            try:
                extracted = ask_llama(city_name, review_text)
                extracted["raw_text"] = review_text
                results.append(extracted)

            except Exception as e:
                print(f"Error processing {city_name}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done! Created processed_destinations.json")


if __name__ == "__main__":
    main()