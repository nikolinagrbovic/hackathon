from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

with open("processed_destinations.json", "r", encoding="utf-8") as f:
    destinations = json.load(f)

class UserInput(BaseModel):
    text: str

CRITERIA_WEIGHTS = {
    "moods": 0.30,
    "activities": 0.20,
    "environment": 0.20,
    "tags": 0.15,
    "climate": 0.05,
    "crowd": 0.05,
    "pace": 0.05
}

def call_llama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def extract_user_profile(user_text):
    prompt = f"""
You are a travel emotion extractor.

Analyze the user's travel desire and return ONLY valid JSON.

User text:
{user_text}

Return this exact JSON:
{{
  "tags": [],
  "moods": [],
  "activities": [],
  "environment": [],
  "climate": "",
  "crowd": "",
  "budget": "",
  "pace": ""
}}

Rules:
- Use lowercase words.
- tags are short general travel signals.
- moods are feelings like calm, romantic, inspired, peaceful, adventurous.
- activities are things the user wants to do.
- environment is the type of place: sea, mountain, city, nature, island, forest, old town.
- climate can be: warm, cold, mild, tropical, dry, unknown.
- crowd can be: low, medium, high, unknown.
- budget can be: low, medium, high, unknown.
- pace can be: slow, balanced, energetic, unknown.
- Return JSON only.
"""

    result = call_llama(prompt)

    start = result.find("{")
    end = result.rfind("}") + 1
    clean_json = result[start:end]

    return json.loads(clean_json)

def normalize(value):
    if not value:
        return []

    if isinstance(value, list):
        return [str(v).lower().strip() for v in value if str(v).strip()]

    return [str(value).lower().strip()]

def match_score(user_values, destination_values):
    user_set = set(normalize(user_values))
    destination_set = set(normalize(destination_values))

    if not user_set:
        return 0, []

    matched = user_set.intersection(destination_set)
    score = len(matched) / len(user_set)

    return score, list(matched)

def recommend_destinations(user_profile):
    results = []

    for destination in destinations:
        total_score = 0
        all_matched = []
        details = {}

        for criterion, weight in CRITERIA_WEIGHTS.items():
            user_values = user_profile.get(criterion, [])
            destination_values = destination.get(criterion, [])

            score, matched = match_score(user_values, destination_values)

            total_score += score * weight
            all_matched.extend(matched)

            details[criterion] = {
                "matched": matched,
                "score": round(score * 100)
            }

        reason = "Matched: " + ", ".join(all_matched[:8]) if all_matched else "Matched by emotional travel similarity."

        results.append({
            "name": destination.get("name"),
            "country": destination.get("country", "Unknown"),
            "match": round(total_score * 100),
            "reason": reason,
            "details": details
        })

    results = sorted(results, key=lambda x: x["match"], reverse=True)
    return results[:5]

@app.get("/")
def home():
    return {"message": "FeelTrip backend is running"}

@app.post("/recommend")
def recommend(data: UserInput):
    user_profile = extract_user_profile(data.text)
    recommendations = recommend_destinations(user_profile)

    return {
        "input": data.text,
        "user_profile": user_profile,
        "recommendations": recommendations
    }