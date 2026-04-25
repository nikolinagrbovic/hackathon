from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PLACES_PATH = Path("places.json")
MODEL = "llama3.2:3b"


class UserInput(BaseModel):
    text: str


def load_json(path: Path):
    if not path.exists():
        print("Missing file:", path.resolve())
        return []

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def ask_llama_json(prompt: str):
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
            timeout=90,
        )

        response.raise_for_status()
        raw = response.json().get("response", "{}")
        print("\nLLAMA RAW:")
        print(raw)

        return json.loads(raw)

    except Exception as e:
        print("LLAMA ERROR:", e)
        return {}


def normalize_tags(tags):
    clean = []

    if not isinstance(tags, list):
        return []

    for tag in tags:
        if isinstance(tag, str):
            value = tag.strip().lower()
            if value:
                clean.append(value)

    unique = []
    for tag in clean:
        if tag not in unique:
            unique.append(tag)

    return unique


def build_flight_url(city):
    return f"https://www.skyscanner.net/transport/flights/?query={city}"


def extract_user_profile(text: str):
    prompt = f"""
You extract travel intent from user input.

User input:
{text}

Return ONLY valid JSON.
Do NOT return markdown.

Rules:
- tags MUST be lowercase strings
- climate MUST be one of: warm, cold, mild, any
- pace MUST be one of: slow, energetic, balanced, any
- crowd MUST be one of: quiet, lively, crowded, any
- mood MUST be one of: romantic, adventurous, peaceful, mysterious, cultural, playful, any
- activities, sensory, travel_style MUST be arrays of lowercase strings
- do NOT return objects inside arrays

JSON format:
{{
  "tags": [],
  "mood": "any",
  "climate": "any",
  "pace": "any",
  "crowd": "any",
  "activities": [],
  "sensory": [],
  "travel_style": []
}}
"""

    ai = ask_llama_json(prompt)

    return {
        "tags": normalize_tags(ai.get("tags", []))[:10],
        "mood": str(ai.get("mood", "any")).lower().strip() or "any",
        "climate": str(ai.get("climate", "any")).lower().strip() or "any",
        "pace": str(ai.get("pace", "any")).lower().strip() or "any",
        "crowd": str(ai.get("crowd", "any")).lower().strip() or "any",
        "activities": normalize_tags(ai.get("activities", [])),
        "sensory": normalize_tags(ai.get("sensory", [])),
        "travel_style": normalize_tags(ai.get("travel_style", [])),
    }


def list_overlap_score(user_items, place_items, weight):
    score = 0
    matched = []
    place_set = set(place_items or [])

    for item in user_items or []:
        if item in place_set:
            score += weight
            matched.append(item)

    return score, matched


def similarity_score(user_profile, place):
    score = 0
    matched = []
    conflicts = []

    for field, weight in [
        ("tags", 3),
        ("activities", 3),
        ("sensory", 2),
        ("travel_style", 2),
    ]:
        field_score, field_matches = list_overlap_score(
            user_profile.get(field, []),
            place.get(field, []),
            weight,
        )
        score += field_score
        matched += field_matches

    for field, match_points, conflict_points in [
        ("mood", 2, 0),
        ("climate", 4, -5),
        ("pace", 3, -4),
        ("crowd", 3, -4),
    ]:
        user_value = user_profile.get(field, "any")
        place_value = place.get(field, "")

        if user_value != "any":
            if user_value == place_value:
                score += match_points
                matched.append(user_value)
            elif conflict_points < 0 and place_value:
                score += conflict_points
                conflicts.append(place_value)

    searchable_text = " ".join([
        place.get("story", ""),
        " ".join(place.get("landmarks", [])),
        " ".join(place.get("tags", [])),
        " ".join(place.get("activities", [])),
        " ".join(place.get("sensory", [])),
        " ".join(place.get("travel_style", [])),
    ]).lower()

    for tag in user_profile.get("tags", []):
        tag = tag.lower().strip()
        if tag and tag in searchable_text and tag not in matched:
            score += 0.6
            matched.append(tag)

    matched = list(dict.fromkeys(matched))
    conflicts = list(dict.fromkeys(conflicts))

    score += len(matched) * 0.4
    score -= len(conflicts) * 0.7

    max_score = (
        len(user_profile.get("tags", [])) * 3 +
        len(user_profile.get("activities", [])) * 3 +
        len(user_profile.get("sensory", [])) * 2 +
        len(user_profile.get("travel_style", [])) * 2 +
        len(user_profile.get("tags", [])) * 0.6 +
        8 * 0.4
    )

    if user_profile.get("mood") != "any":
        max_score += 2

    if user_profile.get("climate") != "any":
        max_score += 4

    if user_profile.get("pace") != "any":
        max_score += 3

    if user_profile.get("crowd") != "any":
        max_score += 3

    if max_score <= 0:
        percent = 50
    else:
        percent = round((score / max_score) * 100)

    percent = int(max(0, min(99, percent)))

    return score, percent, matched[:8], conflicts[:5]


def build_match_reason(matched, conflicts, place):
    if matched and conflicts:
        return f"Matched because of {', '.join(matched[:3])}, but may conflict with {', '.join(conflicts[:2])}."

    if matched:
        return f"Matched because of {', '.join(matched[:4])}."

    return f"Suggested because it has a {place.get('mood', 'travel')} mood and {place.get('pace', 'balanced')} pace."


def format_profile(place, score, percent, matched, conflicts):
    city = place["city"]
    country = place["country"]

    return {
        "name": f"{city}, {country}",
        "city": city,
        "country": country,
        "match": f"{percent}% feeling match",
        "matchReason": build_match_reason(matched, conflicts, place),
        "score": score,
        "story": place.get("story", ""),
        "tags": place.get("tags", []),
        "mood": place.get("mood", ""),
        "climate": place.get("climate", ""),
        "pace": place.get("pace", ""),
        "crowd": place.get("crowd", ""),
        "activities": place.get("activities", []),
        "sensory": place.get("sensory", []),
        "travel_style": place.get("travel_style", []),
        "matchedTags": matched,
        "conflicts": conflicts,
        "landmarks": place.get("landmarks", [])[:4],
        "image": place.get("image", ""),
        "flightUrl": place.get("flightUrl") or build_flight_url(city),
    }


@app.get("/")
def home():
    return {"message": "Backend is running with places.json"}


@app.post("/recommend")
def recommend(data: UserInput):
    print("\n========== NEW REQUEST ==========")

    user_profile = extract_user_profile(data.text)
    places = load_json(PLACES_PATH)

    ranked = []

    for place in places:
        score, percent, matched, conflicts = similarity_score(user_profile, place)
        ranked.append(format_profile(place, score, percent, matched, conflicts))

    ranked.sort(key=lambda item: item["score"], reverse=True)

    return {
        "input": data.text,
        "user_profile": user_profile,
        "results": ranked[:6],
    }
