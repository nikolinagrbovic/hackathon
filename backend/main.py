from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PLACES_PATH = Path("places.json")


class UserInput(BaseModel):
    text: str


def load_json(path: Path):
    if not path.exists():
        print("Missing file:", path.resolve())
        return []

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_flight_url(city):
    return f"https://www.skyscanner.net/transport/flights/?query={city}"


SYNONYMS = {
    "calm": ["peaceful", "quiet", "slow", "silence", "relaxing", "soft", "still"],
    "quiet": ["peaceful", "calm", "silence", "slow", "hidden"],
    "relax": ["peaceful", "calm", "slow", "escape", "soft"],
    "romantic": ["love", "sunset", "golden", "dreamy", "soft", "intimate"],
    "sunset": ["golden", "evening", "romantic", "warm light", "soft light"],
    "sea": ["ocean", "water", "blue", "coast", "beach", "harbor", "breeze"],
    "ocean": ["sea", "water", "blue", "coast", "beach", "breeze"],
    "old": ["ancient", "history", "historic", "stone", "architecture", "heritage"],
    "history": ["ancient", "museum", "cultural", "ruins", "heritage", "architecture"],
    "museum": ["art", "culture", "history", "gallery", "cultural"],
    "art": ["museum", "gallery", "creative", "cultural", "colorful"],
    "food": ["market", "taste", "street food", "local food", "sensory"],
    "market": ["food", "local food", "noise", "colors", "lively"],
    "dark": ["mysterious", "gothic", "shadows", "hidden", "ancient"],
    "mystery": ["mysterious", "hidden", "secret", "dark", "strange"],
    "hidden": ["secret", "quiet", "mysterious", "narrow streets"],
    "colorful": ["playful", "vibrant", "fun", "joyful", "whimsical"],
    "fun": ["playful", "lively", "colorful", "joyful"],
    "adventure": ["adventurous", "explore", "climbing", "viewpoint", "discovery"],
    "explore": ["adventurous", "walking", "discovery", "hidden", "viewpoint"],
    "nature": ["green", "forest", "water", "fresh air", "peaceful", "outdoors"],
    "cold": ["fresh air", "winter", "crisp", "quiet"],
    "warm": ["sun", "summer", "heat", "golden", "outdoor"],
}


def extract_words(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def fast_user_profile(text):
    words = extract_words(text)

    tags = []
    expanded = []

    for word in words:
        if len(word) < 3:
            continue

        if word not in tags:
            tags.append(word)

        for synonym in SYNONYMS.get(word, []):
            if synonym not in expanded:
                expanded.append(synonym)

    full_text = text.lower()

    mood = "any"
    if any(w in full_text for w in ["romantic", "love", "sunset", "golden", "dream"]):
        mood = "romantic"
    elif any(w in full_text for w in ["calm", "quiet", "peace", "relax", "silence"]):
        mood = "peaceful"
    elif any(w in full_text for w in ["history", "museum", "old", "ancient", "art"]):
        mood = "cultural"
    elif any(w in full_text for w in ["dark", "mystery", "hidden", "strange"]):
        mood = "mysterious"
    elif any(w in full_text for w in ["fun", "colorful", "magic", "playful"]):
        mood = "playful"
    elif any(w in full_text for w in ["adventure", "explore", "climb", "hiking"]):
        mood = "adventurous"

    climate = "any"
    if any(w in full_text for w in ["warm", "sun", "summer", "hot", "beach"]):
        climate = "warm"
    elif any(w in full_text for w in ["cold", "winter", "snow", "crisp"]):
        climate = "cold"
    elif any(w in full_text for w in ["mild", "spring", "autumn"]):
        climate = "mild"

    pace = "any"
    if any(w in full_text for w in ["slow", "calm", "relax", "quiet"]):
        pace = "slow"
    elif any(w in full_text for w in ["energy", "energetic", "busy", "nightlife", "adrenaline"]):
        pace = "energetic"

    crowd = "any"
    if any(w in full_text for w in ["quiet", "alone", "hidden", "peaceful", "calm"]):
        crowd = "quiet"
    elif any(w in full_text for w in ["crowd", "busy", "people", "market", "nightlife"]):
        crowd = "lively"

    activities = []
    for word in ["walking", "hiking", "museum", "food", "market", "architecture", "history", "viewpoint", "nature"]:
        if word in full_text:
            activities.append(word)

    sensory = []
    for word in ["sun", "sea", "water", "wind", "light", "colors", "silence", "noise", "stone", "fresh air"]:
        if word in full_text:
            sensory.append(word)

    travel_style = []
    if mood != "any":
        travel_style.append(mood)

    return {
        "tags": tags[:15],
        "expanded_terms": expanded[:35],
        "mood": mood,
        "climate": climate,
        "pace": pace,
        "crowd": crowd,
        "activities": activities,
        "sensory": sensory,
        "travel_style": travel_style,
    }


def soft_match(a, b):
    a = str(a).lower().strip()
    b = str(b).lower().strip()

    if not a or not b:
        return False

    return a == b or a in b or b in a


def list_overlap_score(user_items, place_items, weight):
    score = 0
    matched = []

    for user_item in user_items or []:
        for place_item in place_items or []:
            if soft_match(user_item, place_item):
                score += weight
                matched.append(user_item)
                break

    return score, matched


def similarity_score(user_profile, place):
    score = 0
    matched = []
    conflicts = []

    all_user_tags = list(dict.fromkeys(
        user_profile.get("tags", []) +
        user_profile.get("expanded_terms", [])
    ))

    place_all_tags = list(dict.fromkeys(
        place.get("tags", []) +
        place.get("activities", []) +
        place.get("sensory", []) +
        place.get("travel_style", []) +
        place.get("landmarks", [])
    ))

    tag_score, tag_matches = list_overlap_score(all_user_tags, place_all_tags, 3)
    score += tag_score
    matched += tag_matches

    activity_score, activity_matches = list_overlap_score(
        user_profile.get("activities", []),
        place.get("activities", []),
        4,
    )
    score += activity_score
    matched += activity_matches

    sensory_score, sensory_matches = list_overlap_score(
        user_profile.get("sensory", []),
        place.get("sensory", []),
        3,
    )
    score += sensory_score
    matched += sensory_matches

    style_score, style_matches = list_overlap_score(
        user_profile.get("travel_style", []),
        place.get("travel_style", []),
        3,
    )
    score += style_score
    matched += style_matches

    if user_profile.get("mood") != "any":
        if user_profile.get("mood") == place.get("mood"):
            score += 10
            matched.append(user_profile["mood"])

    if user_profile.get("climate") != "any":
        if user_profile.get("climate") == place.get("climate"):
            score += 4
            matched.append(user_profile["climate"])
        else:
            score -= 1
            conflicts.append(place.get("climate", ""))

    if user_profile.get("pace") != "any":
        if user_profile.get("pace") == place.get("pace"):
            score += 5
            matched.append(user_profile["pace"])
        else:
            score -= 1
            conflicts.append(place.get("pace", ""))

    if user_profile.get("crowd") != "any":
        if user_profile.get("crowd") == place.get("crowd"):
            score += 5
            matched.append(user_profile["crowd"])
        else:
            score -= 1
            conflicts.append(place.get("crowd", ""))

    searchable_text = " ".join([
        place.get("story", ""),
        place.get("mood", ""),
        place.get("climate", ""),
        place.get("pace", ""),
        place.get("crowd", ""),
        " ".join(place.get("landmarks", [])),
        " ".join(place.get("tags", [])),
        " ".join(place.get("activities", [])),
        " ".join(place.get("sensory", [])),
        " ".join(place.get("travel_style", [])),
    ]).lower()

    for term in all_user_tags:
        if term and term in searchable_text and term not in matched:
            score += 1
            matched.append(term)

    matched = list(dict.fromkeys([m for m in matched if m]))
    conflicts = list(dict.fromkeys([c for c in conflicts if c]))

    score += len(matched) * 0.6
    score -= len(conflicts) * 0.2

    max_score = (
        len(all_user_tags) * 3 +
        len(user_profile.get("activities", [])) * 4 +
        len(user_profile.get("sensory", [])) * 3 +
        len(user_profile.get("travel_style", [])) * 3 +
        len(all_user_tags) * 1 +
        10
    )

    if user_profile.get("climate") != "any":
        max_score += 4
    if user_profile.get("pace") != "any":
        max_score += 5
    if user_profile.get("crowd") != "any":
        max_score += 5

    if max_score <= 0:
        percent = 45
    else:
        raw_percent = (score / max_score) * 100

        # podiže niske rezultate da izgledaju bolje za demo
        percent = round(35 + (raw_percent * 0.65))

    percent = int(max(35, min(96, percent)))

    return score, percent, matched[:10], conflicts[:5]


def build_match_reason(matched, conflicts, place):
    if matched:
        return f"Matched because of {', '.join(matched[:5])}."

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
    return {"message": "Backend is running fast without AI per request33"}


@app.post("/recommend")
def recommend(data: UserInput):
    user_text = data.text
    user_profile = fast_user_profile(user_text)
    places = load_json(PLACES_PATH)

    ranked = []

    for place in places:
        score, percent, matched, conflicts = similarity_score(user_profile, place)
        ranked.append(format_profile(place, score, percent, matched, conflicts))

    ranked.sort(key=lambda item: item["score"], reverse=True)

    return {
        "input": user_text,
        "user_profile": user_profile,
        "results": ranked[:6],
    }

COORDS_PATH = Path("coords.json")


def load_coords():
    if not COORDS_PATH.exists():
        return {}

    with open(COORDS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/journey/{city}")
def get_journey(city: str):
    places = load_json(PLACES_PATH)
    coords_data = load_coords()

    for place in places:
        if place["city"].lower().strip() == city.lower().strip():
            city_name = place["city"]

            coords_entry = {}

            for key, value in coords_data.items():
                if key.lower().strip() == city_name.lower().strip():
                    coords_entry = value
                    break

            city_coords = coords_entry.get("coordinates", [])
            landmark_coords = coords_entry.get("landmarks", {})

            print("JOURNEY CITY:", city_name)
            print("CITY COORDS:", city_coords)
            print("LANDMARK COORDS:", landmark_coords)

            return {
                **place,
                "coordinates": city_coords,
                "landmarkCoordinates": landmark_coords,
                "hotelUrl": f"https://www.skyscanner.net/hotels/search?entity_name={city_name}",
                "flightUrl": place.get("flightUrl") or build_flight_url(city_name),
            }

    return {"error": "Place not found"}


PLACES_PATH = Path("places.json")
@app.get("/stories")
def get_stories():
    return load_json(PLACES_PATH)
