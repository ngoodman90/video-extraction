import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# 1. Setup Client (Ensure GEMINI_API_KEY is in your environment)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_ID = "gemini-3-flash-preview"


def extract_segments(video_path, user_prompt):
    # 2. Upload the video to the File API
    print(f"Uploading {video_path}...")
    video_file = client.files.upload(file=f'../assets/{video_path}')

    # 3. Wait for the video to be processed (Required for inference)
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed.")

    print("\nVideo is active. Analyzing...")

    # 4. Define the prompt with recall-biased system instructions
    prompt = f"""You are a video analysis assistant. Your job is to find ALL segments in this video matching the user's query.

Query: "{user_prompt}"

Instructions:
- Err on the side of INCLUSION. A false positive is acceptable; a missed segment is not.
- Look for both direct and indirect depictions (e.g. facial expressions, body language, text overlays, before/after comparisons).
- Each distinct moment should be its own segment — do NOT merge nearby scenes.
- Use the tightest time window for each match.
- Scan the entire video thoroughly from start to finish.
- Rate your confidence for each match from 0.0 to 1.0.

Return ONLY valid JSON:
{{"matches": [{{"start_time": "MM:SS", "end_time": "MM:SS", "confidence": 0.0, "reasoning": "text"}}]}}
"""

    # 5. Generate Content
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[video_file, prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="medium"),
            response_mime_type="application/json"
        )
    )

    # 6. Parse and Return
    return json.loads(response.text)


# Example Usage
if __name__ == "__main__":
    try:
        results = extract_segments("back_pain_commercial.mp4", "Woman experiencing back pain")
        matches = results.get("matches", [])
        print(f"Found {len(matches)} segment(s):\n")
        for i, match in enumerate(matches, 1):
            print(f"  Segment {i}: {match['start_time']} - {match['end_time']} (confidence: {match['confidence']})")
            print(f"    Reasoning: {match['reasoning']}\n")
    except Exception as e:
        print(f"Error: {e}")