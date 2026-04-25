from pathlib import Path
import json
import requests
import time

PLACES_PATH = Path("places.json")
OUTPUT_PATH = Path("coords.json")
MODEL = "llama3.2:3b"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def ask_llama_json(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=120,
        )

        response.raise_for_status()
        raw = response.json().get("response", "{}")

        return json.loads(raw)

    except Exception as e:
        print("LLAMA ERROR:", e)
        return {}


def valid_coords(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
        and -90 <= value[0] <= 90
        and -180 <= value[1] <= 180
    )


def get_coordinates_for_place(place):
    city = place.get("city", "")
    country = place.get("country", "")
    landmarks = place.get("landmarks", [])

    prompt = f"""
Find approximate real-world coordinates.

City: {city}
Country: {country}
Landmarks: {json.dumps(landmarks, ensure_ascii=False)}

Return ONLY JSON.

Format:
{{
  "coordinates": [lat, lng],
  "landmarks": {{
    "Landmark": [lat, lng]
  }}
}}
"""

    ai = ask_llama_json(prompt)

    city_coords = ai.get("coordinates", [])
    landmark_coords = ai.get("landmarks", {})

    if not valid_coords(city_coords):
        city_coords = []

    clean_landmarks = {}

    for lm in landmarks:
        coords = landmark_coords.get(lm)
        if valid_coords(coords):
            clean_landmarks[lm] = coords

    return city_coords, clean_landmarks


def main():
    places = load_json(PLACES_PATH)

    coords_data = {}

    for place in places:
        city = place["city"]

        print(f"Geocoding {city}...")

        city_coords, landmark_coords = get_coordinates_for_place(place)

        coords_data[city] = {
            "coordinates": city_coords,
            "landmarks": landmark_coords,
        }

        time.sleep(0.5)

    save_json(OUTPUT_PATH, coords_data)

    print("\nDone -> coords.json")


if __name__ == "__main__":
    main()