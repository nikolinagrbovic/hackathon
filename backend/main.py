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

DATA_PATH = Path("data.json")
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

        elif isinstance(tag, dict):
            for key in ["tag", "name", "description", "value", "type"]:
                value = tag.get(key)
                if isinstance(value, str) and value.strip():
                    clean.append(value.strip().lower())
                    break

    unique = []
    for tag in clean:
        if tag not in unique:
            unique.append(tag)

    return unique


def enforce_max_words(text, max_words=30):
    if not isinstance(text, str):
        return ""

    words = text.strip().split()
    return " ".join(words[:max_words])


def build_flight_url(city):
    return f"https://www.skyscanner.net/transport/flights/?query={city}"


def group_reviews_by_city():
    data = load_json(DATA_PATH)
    grouped = {}

    for item in data:
        city = item.get("city", "").strip()
        country = item.get("country", "").strip()
        review = item.get("review", "").strip()
        landmark = item.get("landmark", "").strip()

        if not city:
            continue

        key = f"{city}, {country}"

        if key not in grouped:
            grouped[key] = {
                "city": city,
                "country": country,
                "name": key,
                "reviews": [],
                "landmarks": [],
            }

        if review:
            grouped[key]["reviews"].append(review)

        if landmark and landmark not in grouped[key]["landmarks"]:
            grouped[key]["landmarks"].append(landmark)

    return list(grouped.values())


def create_ai_city_profile(city_group):
    reviews_text = "\n".join(city_group["reviews"])
    landmarks_text = ", ".join(city_group["landmarks"])

    prompt = f"""
You are creating a travel emotion profile for one city.

City: {city_group["city"]}
Country: {city_group["country"]}
Landmarks: {landmarks_text}

Reviews:
{reviews_text}

Return ONLY valid JSON.
Do NOT return markdown.
Do NOT invent facts.

Rules:
- story MUST be exactly ONE sentence
- story MUST be max 30 words
- tags MUST be lowercase strings
- climate MUST be one of: warm, cold, mild
- pace MUST be one of: slow, energetic, balanced
- crowd MUST be one of: quiet, lively, crowded
- mood MUST be one of: romantic, adventurous, peaceful, mysterious, cultural, playful
- activities, sensory, travel_style MUST be arrays of lowercase strings
- do NOT return objects inside arrays

JSON format:
{{
  "story": "",
  "tags": [],
  "mood": "",
  "climate": "",
  "pace": "",
  "crowd": "",
  "activities": [],
  "sensory": [],
  "travel_style": []
}}
"""

    ai = ask_llama_json(prompt)

    story = enforce_max_words(ai.get("story", ""), 30)
    tags = normalize_tags(ai.get("tags", []))

    return {
        "story": story,
        "tags": tags[:14],
        "mood": str(ai.get("mood", "")).lower().strip(),
        "climate": str(ai.get("climate", "")).lower().strip(),
        "pace": str(ai.get("pace", "")).lower().strip(),
        "crowd": str(ai.get("crowd", "")).lower().strip(),
        "activities": normalize_tags(ai.get("activities", [])),
        "sensory": normalize_tags(ai.get("sensory", [])),
        "travel_style": normalize_tags(ai.get("travel_style", [])),
    }


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
  "mood": "",
  "climate": "",
  "pace": "",
  "crowd": "",
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


def list_overlap_score(user_items, city_items, weight):
    score = 0
    matched = []

    city_set = set(city_items or [])

    for item in user_items or []:
        if item in city_set:
            score += weight
            matched.append(item)

    return score, matched


def similarity_score(user_profile, city_profile):
    score = 0
    matched = []
    conflicts = []

    tag_score, tag_matches = list_overlap_score(
        user_profile.get("tags", []),
        city_profile.get("tags", []),
        3,
    )
    score += tag_score
    matched += tag_matches

    activity_score, activity_matches = list_overlap_score(
        user_profile.get("activities", []),
        city_profile.get("activities", []),
        3,
    )
    score += activity_score
    matched += activity_matches

    sensory_score, sensory_matches = list_overlap_score(
        user_profile.get("sensory", []),
        city_profile.get("sensory", []),
        2,
    )
    score += sensory_score
    matched += sensory_matches

    style_score, style_matches = list_overlap_score(
        user_profile.get("travel_style", []),
        city_profile.get("travel_style", []),
        2,
    )
    score += style_score
    matched += style_matches

    if user_profile.get("mood") != "any":
        if user_profile.get("mood") == city_profile.get("mood"):
            score += 2
            matched.append(user_profile["mood"])

    if user_profile.get("climate") != "any":
        if user_profile.get("climate") == city_profile.get("climate"):
            score += 4
            matched.append(user_profile["climate"])
        elif city_profile.get("climate"):
            score -= 5
            conflicts.append(city_profile["climate"])

    if user_profile.get("pace") != "any":
        if user_profile.get("pace") == city_profile.get("pace"):
            score += 3
            matched.append(user_profile["pace"])
        elif city_profile.get("pace"):
            score -= 4
            conflicts.append(city_profile["pace"])

    if user_profile.get("crowd") != "any":
        if user_profile.get("crowd") == city_profile.get("crowd"):
            score += 3
            matched.append(user_profile["crowd"])
        elif city_profile.get("crowd"):
            score -= 4
            conflicts.append(city_profile["crowd"])

    matched = list(dict.fromkeys(matched))
    conflicts = list(dict.fromkeys(conflicts))

    percent = max(10, min(98, 65 + score * 4))

    return score, percent, matched[:8], conflicts[:5]


def format_profile(city_profile, score, percent, matched, conflicts):
    city = city_profile["city"]
    country = city_profile["country"]

    return {
        "name": f"{city}, {country}",
        "city": city,
        "country": country,
        "match": f"{percent}% feeling match",
        "score": score,
        "story": city_profile.get("story", ""),
        "tags": city_profile.get("tags", []),
        "mood": city_profile.get("mood", ""),
        "climate": city_profile.get("climate", ""),
        "pace": city_profile.get("pace", ""),
        "crowd": city_profile.get("crowd", ""),
        "activities": city_profile.get("activities", []),
        "sensory": city_profile.get("sensory", []),
        "travel_style": city_profile.get("travel_style", []),
        "matchedTags": matched,
        "conflicts": conflicts,
        "landmarks": city_profile.get("landmarks", [])[:4],
        "image": "",
        "flightUrl": build_flight_url(city),
    }


@app.get("/")
def home():
    return {"message": "Backend is running"}


@app.post("/recommend")
def recommend(data: UserInput):
    print("\n========== NEW REQUEST ==========")

    user_profile = extract_user_profile(data.text)
    city_groups = group_reviews_by_city()

    ranked = []

    for city_group in city_groups:
        ai_profile = create_ai_city_profile(city_group)

        city_profile = {
            "city": city_group["city"],
            "country": city_group["country"],
            "reviews": city_group["reviews"],
            "landmarks": city_group["landmarks"],
            **ai_profile,
        }

        score, percent, matched, conflicts = similarity_score(user_profile, city_profile)

        ranked.append(format_profile(city_profile, score, percent, matched, conflicts))

    ranked.sort(key=lambda item: item["score"], reverse=True)

    return {
        "input": data.text,
        "user_profile": user_profile,
        "results": ranked[:6],
    }
