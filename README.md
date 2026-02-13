# video-extraction

Compare LLM-based (Gemini) and local (CLIP) approaches for finding temporal segments in videos matching a text query.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage

Place video files in the `assets/` directory, then run from `src/`:

```bash
cd src
python compare.py <video_filename> <query>
```

Example:

```bash
python compare.py back_pain_commercial.mp4 "Woman experiencing back pain"
```

This runs both analyzers and prints a comparison showing overlapping segments, method-specific segments, and a temporal IoU score. Results are also saved to a JSON file.

You can also run each analyzer independently:

```bash
python llm_analysis.py
python local_analysis.py
```