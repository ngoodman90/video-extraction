"""
Compare results from LLM (Gemini) and local (CLIP) video analysis.

Usage:
    python compare.py <video_filename> <query>
    python compare.py back_pain_commercial.mp4 "Woman experiencing back pain"
"""

import sys
import time
import json

import llm_analysis
import local_analysis


def parse_timestamp(ts: str) -> int:
    """Convert MM:SS to total seconds."""
    parts = ts.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def segments_overlap(a, b) -> bool:
    """Check if two segments overlap at all."""
    a_start = parse_timestamp(a["start_time"])
    a_end = parse_timestamp(a["end_time"])
    b_start = parse_timestamp(b["start_time"])
    b_end = parse_timestamp(b["end_time"])
    return a_start < b_end and b_start < a_end


def overlap_seconds(a, b) -> int:
    """Compute overlap duration in seconds between two segments."""
    a_start = parse_timestamp(a["start_time"])
    a_end = parse_timestamp(a["end_time"])
    b_start = parse_timestamp(b["start_time"])
    b_end = parse_timestamp(b["end_time"])
    overlap = min(a_end, b_end) - max(a_start, b_start)
    return max(0, overlap)


def duration(seg) -> int:
    return parse_timestamp(seg["end_time"]) - parse_timestamp(seg["start_time"])


def print_matches(label: str, matches: list):
    print(f"\n{'=' * 60}")
    print(f"  {label}: {len(matches)} segment(s)")
    print(f"{'=' * 60}")
    for i, m in enumerate(matches, 1):
        conf = m.get("confidence", "N/A")
        print(f"  {i}. {m['start_time']} - {m['end_time']}  (confidence: {conf})")
        print(f"     {m['reasoning']}")


def compare(llm_matches: list, local_matches: list):
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON")
    print(f"{'=' * 60}")

    # Match each LLM segment to its best-overlapping local segment (and vice versa)
    llm_matched = set()
    local_matched = set()
    pairs = []

    for i, lm in enumerate(llm_matches):
        best_j, best_overlap = None, 0
        for j, loc in enumerate(local_matches):
            ov = overlap_seconds(lm, loc)
            if ov > best_overlap:
                best_j, best_overlap = j, ov
        if best_j is not None and best_overlap > 0:
            pairs.append((i, best_j, best_overlap))
            llm_matched.add(i)
            local_matched.add(best_j)

    # Overlapping pairs
    if pairs:
        print(f"\n  Overlapping segments ({len(pairs)}):")
        for i, j, ov in pairs:
            lm, loc = llm_matches[i], local_matches[j]
            print(f"    LLM   {lm['start_time']}-{lm['end_time']}  |  "
                  f"Local {loc['start_time']}-{loc['end_time']}  |  "
                  f"overlap: {ov}s")

    # LLM-only
    llm_only = [m for i, m in enumerate(llm_matches) if i not in llm_matched]
    if llm_only:
        print(f"\n  LLM-only segments ({len(llm_only)}):")
        for m in llm_only:
            print(f"    {m['start_time']} - {m['end_time']}  ({m['reasoning']})")

    # Local-only
    local_only = [m for j, m in enumerate(local_matches) if j not in local_matched]
    if local_only:
        print(f"\n  Local-only segments ({len(local_only)}):")
        for m in local_only:
            print(f"    {m['start_time']} - {m['end_time']}  ({m['reasoning']})")

    # Summary stats
    total_llm = sum(duration(m) for m in llm_matches)
    total_local = sum(duration(m) for m in local_matches)
    total_overlap = sum(ov for _, _, ov in pairs)

    print(f"\n  Summary:")
    print(f"    LLM total coverage:   {total_llm}s across {len(llm_matches)} segments")
    print(f"    Local total coverage:  {total_local}s across {len(local_matches)} segments")
    print(f"    Overlapping coverage:  {total_overlap}s across {len(pairs)} pairs")
    if total_llm + total_local - total_overlap > 0:
        iou = total_overlap / (total_llm + total_local - total_overlap)
        print(f"    Temporal IoU:          {iou:.1%}")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <video_filename> <query>")
        print(f'Example: python {sys.argv[0]} back_pain_commercial.mp4 "Woman experiencing back pain"')
        sys.exit(1)

    video_path = sys.argv[1]
    user_prompt = " ".join(sys.argv[2:])

    print(f"Video: {video_path}")
    print(f"Query: \"{user_prompt}\"")

    # Run LLM analysis
    print(f"\n{'#' * 60}")
    print(f"  Running LLM analysis (Gemini)...")
    print(f"{'#' * 60}")
    t0 = time.time()
    llm_results = llm_analysis.extract_segments(video_path, user_prompt)
    llm_time = time.time() - t0
    llm_matches = llm_results.get("matches", [])
    print_matches("LLM (Gemini)", llm_matches)
    print(f"  Time: {llm_time:.1f}s")

    # Run local CLIP analysis
    print(f"\n{'#' * 60}")
    print(f"  Running local analysis (CLIP)...")
    print(f"{'#' * 60}")
    t0 = time.time()
    local_results = local_analysis.extract_segments(video_path, user_prompt)
    local_time = time.time() - t0
    local_matches = local_results.get("matches", [])
    print_matches("Local (CLIP)", local_matches)
    print(f"  Time: {local_time:.1f}s")

    # Compare
    compare(llm_matches, local_matches)

    # Save raw results
    output = {
        "video": video_path,
        "query": user_prompt,
        "llm": {"matches": llm_matches, "time_seconds": round(llm_time, 1)},
        "local": {"matches": local_matches, "time_seconds": round(local_time, 1)},
    }
    out_path = f"comparison_{video_path.replace('.', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Raw results saved to {out_path}")


if __name__ == "__main__":
    main()
