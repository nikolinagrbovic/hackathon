from pathlib import Path
from collections import defaultdict
import json
import requests

DATA_PATH = Path("data.json")
OUTPUT_PATH = Path("places.json")
MODEL = "llama3.2:3b"

CITY_IMAGES = {
    "Barcelona": "https://images.unsplash.com/photo-1583422409516-2895a77efded?auto=format&fit=crop&w=900&q=80",
    "Madrid": "https://images.unsplash.com/photo-1539037116277-4db20889f2d4?auto=format&fit=crop&w=900&q=80",
    "Paris": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=900&q=80",
    "Rome": "https://images.unsplash.com/photo-1529260830199-42c24126f198?auto=format&fit=crop&w=900&q=80",
    "Berlin": "https://images.unsplash.com/photo-1560969184-10fe8719e047?auto=format&fit=crop&w=900&q=80",
    "Athens": "https://images.unsplash.com/photo-1555993539-1732b0258235?auto=format&fit=crop&w=900&q=80",
    "Oslo": "https://images.unsplash.com/photo-1605283176560-37dcb61c0cce?auto=format&fit=crop&w=900&q=80",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def group_by_city(data):
    grouped = defaultdict(lambda: {
        "city": "",
        "country": "",
        "reviews": [],
        "landmarks": []
    })

    for item in data:
        city = item.get("city", "").strip()
        country = item.get("country", "").strip()
        landmark = item.get("landmark", "").strip()
        review = item.get("review", "").strip()

        if not city:
            continue

        key = f"{city}, {country}"

        grouped[key]["city"] = city
        grouped[key]["country"] = country

        if landmark and landmark not in grouped[key]["landmarks"]:
            grouped[key]["landmarks"].append(landmark)

        if review:
            grouped[key]["reviews"].append(review)

    return list(grouped.values())


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

        print("\nLLAMA RAW:")
        print(raw)

        return json.loads(raw)

    except Exception as e:
        print("LLAMA ERROR:", e)
        return {}


def clean_string(value):
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def clean_list(value, limit=None):
    if not isinstance(value, list):
        return []

    result = []

    for item in value:
        if isinstance(item, str):
            clean = item.strip().lower()
            if clean and clean not in result:
                result.append(clean)

    if limit:
        return result[:limit]

    return result


def enforce_story(text, max_words=28):
    if not isinstance(text, str):
        return ""

    words = text.strip().split()
    return " ".join(words[:max_words])


def create_place_profile(city_group):
    city = city_group["city"]
    country = city_group["country"]
    landmarks = city_group["landmarks"]
    reviews = city_group["reviews"]

    prompt = f"""
You transform raw traveler stories into one structured travel recommendation profile.

City: {city}
Country: {country}
Landmarks: {", ".join(landmarks)}

Traveler stories:
{chr(10).join("- " + review for review in reviews)}

Return ONLY valid JSON.
Do NOT return markdown.
Do NOT invent new landmarks.
Use only lowercase strings in arrays.

Important:
This profile will be used for matching a user's desired feeling with destinations.

Rules:
- story:whole sentence and max 30 words
- mood: one of romantic, adventurous, peaceful, mysterious, cultural, playful
- climate: one of warm, cold, mild
- pace: one of slow, energetic, balanced
- crowd: one of quiet, lively, crowded
- tags: 10 to 14 short lowercase keywords
- activities: 4 to 8 lowercase keywords
- sensory: 4 to 8 lowercase sensory keywords
- travel_style: 3 to 6 lowercase keywords
- avoid generic words like city, travel, place, beautiful
- do not return objects inside arrays

JSON format:
{{
  "story": "",
  "mood": "",
  "climate": "",
  "pace": "",
  "crowd": "",
  "tags": [],
  "activities": [],
  "sensory": [],
  "travel_style": []
}}
"""

    ai = ask_llama_json(prompt)

    return {
        "city": city,
        "country": country,
        "story": enforce_story(ai.get("story", "")),
        "mood": clean_string(ai.get("mood", "cultural")),
        "climate": clean_string(ai.get("climate", "mild")),
        "pace": clean_string(ai.get("pace", "balanced")),
        "crowd": clean_string(ai.get("crowd", "lively")),
        "tags": clean_list(ai.get("tags", []), 14),
        "activities": clean_list(ai.get("activities", []), 8),
        "sensory": clean_list(ai.get("sensory", []), 8),
        "travel_style": clean_list(ai.get("travel_style", []), 6),
        "landmarks": landmarks[:5],
        "image": CITY_IMAGES.get(city, ""),
        "flightUrl": f"https://www.skyscanner.net/transport/flights/?query={city}",
    }


def main():
    data = load_json(DATA_PATH)
    city_groups = group_by_city(data)

    places = []

    for city_group in city_groups:
        print(f"\nProcessing {city_group['city']}...")
        place = create_place_profile(city_group)
        places.append(place)

    save_json(OUTPUT_PATH, places)

    print(f"\nDone. Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()