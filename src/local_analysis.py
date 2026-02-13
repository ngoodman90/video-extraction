"""
Temporal Video Grounding using CLIP.
Finds precise start/end timestamps for text queries in assets.

Dependencies:
pip install torch torchvision transformers opencv-python numpy pillow scipy
"""

import cv2
import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from scipy.ndimage import gaussian_filter1d

SYSTEM_PROMPT = """You are a video analysis system. Your job is to find ALL segments matching the user's query.
Err on the side of INCLUSION — a false positive is acceptable; a missed segment is not.
Look for both direct and indirect depictions (facial expressions, body language, text overlays, before/after comparisons).
Each distinct moment should be its own segment."""

QUERY_TEMPLATES = [
    "{}",
    "a photo of {}",
    "a video frame showing {}",
    "a scene depicting {}",
]


@dataclass
class TemporalSegment:
    """Represents a temporal segment with relevance score."""
    start_time: float
    end_time: float
    score: float
    start_frame: int
    end_frame: int


class TemporalVideoGrounder:
    """Grounds text queries to precise temporal segments in assets."""

    def __init__(
            self,
            model_name: str = "openai/clip-vit-large-patch14-336",
            device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def extract_frames(
            self,
            video_path: str,
            sample_rate: int = 1
    ) -> Tuple[List[np.ndarray], float, int]:
        """
        Extract frames from video at specified sample rate.

        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame (1 = all frames)

        Returns:
            Tuple of (frames, fps, total_frames)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_idx += 1

        cap.release()
        return frames, fps, total_frames

    def _encode_texts(self, queries: List[str]) -> torch.Tensor:
        """Encode multiple text queries and return normalized embeddings."""
        inputs = self.processor(text=queries, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def compute_similarity_scores(
            self,
            frames: List[np.ndarray],
            text_query: str,
            batch_size: int = 16
    ) -> np.ndarray:
        """
        Compute CLIP similarity scores between frames and text.

        Uses multiple query augmentations and averages scores for robustness.
        """
        queries = [t.format(text_query) for t in QUERY_TEMPLATES]
        text_embeds = self._encode_texts(queries)  # (num_queries, dim)

        scores = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            pil_images = [Image.fromarray(f) for f in batch_frames]

            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_embeds = self.model.get_image_features(**inputs)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                # (batch, num_queries) -> average across query augmentations
                sims = (image_embeds @ text_embeds.T).mean(dim=-1)

            scores.extend(sims.cpu().numpy())

        return np.array(scores)

    def smooth_scores(
            self,
            scores: np.ndarray,
            sigma: float = 2.0
    ) -> np.ndarray:
        """Apply Gaussian smoothing to reduce noise while preserving peaks."""
        return gaussian_filter1d(scores, sigma=sigma)

    def find_segments(
            self,
            scores: np.ndarray,
            fps: float,
            sample_rate: int,
            min_duration: float = 1.0,
            max_duration: float = 30.0,
            gap_tolerance: float = 1.0
    ) -> List[TemporalSegment]:
        """
        Find contiguous regions above a dynamic threshold.

        Args:
            scores: Frame similarity scores
            fps: Video frames per second
            sample_rate: Frame sampling rate used
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            gap_tolerance: Merge regions closer than this many seconds

        Returns:
            List of TemporalSegments sorted by score descending
        """
        smoothed = self.smooth_scores(scores)
        threshold = np.mean(smoothed) + 0.25 * np.std(smoothed)

        # Find contiguous regions above threshold
        above = smoothed >= threshold
        regions = []
        start = None
        for i, v in enumerate(above):
            if v and start is None:
                start = i
            elif not v and start is not None:
                regions.append((start, i))
                start = None
        if start is not None:
            regions.append((start, len(above)))

        if not regions:
            return []

        # Merge nearby regions within gap_tolerance
        gap_frames = int(gap_tolerance * fps / sample_rate)
        merged = [regions[0]]
        for s, e in regions[1:]:
            prev_s, prev_e = merged[-1]
            if s - prev_e <= gap_frames:
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))

        # Convert to TemporalSegments, filtering by duration
        segments = []
        for s_idx, e_idx in merged:
            start_frame = s_idx * sample_rate
            end_frame = e_idx * sample_rate
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time

            if duration < min_duration or duration > max_duration:
                continue

            avg_score = float(np.mean(smoothed[s_idx:e_idx]))
            segments.append(TemporalSegment(
                start_time=start_time,
                end_time=end_time,
                score=avg_score,
                start_frame=start_frame,
                end_frame=end_frame
            ))

        # Sort by score descending
        segments.sort(key=lambda seg: seg.score, reverse=True)
        return segments

    def refine_boundaries(
            self,
            segment: TemporalSegment,
            scores: np.ndarray,
            fps: float,
            sample_rate: int,
            threshold_percentile: float = 0.6
    ) -> TemporalSegment:
        """
        Refine segment boundaries by trimming low-score edges.

        Args:
            segment: Initial segment to refine
            scores: Frame similarity scores
            fps: Video frames per second
            sample_rate: Frame sampling rate used
            threshold_percentile: Score percentile for boundary trimming

        Returns:
            Refined TemporalSegment
        """
        start_idx = segment.start_frame // sample_rate
        end_idx = segment.end_frame // sample_rate

        segment_scores = scores[start_idx:end_idx]
        threshold = np.percentile(segment_scores, threshold_percentile * 100)

        # Find first frame above threshold
        above_threshold = segment_scores >= threshold
        if not any(above_threshold):
            return segment

        first_idx = np.argmax(above_threshold)
        last_idx = len(segment_scores) - np.argmax(above_threshold[::-1]) - 1

        # Update segment boundaries
        new_start_frame = (start_idx + first_idx) * sample_rate
        new_end_frame = (start_idx + last_idx + 1) * sample_rate

        return TemporalSegment(
            start_time=new_start_frame / fps,
            end_time=new_end_frame / fps,
            score=segment.score,
            start_frame=new_start_frame,
            end_frame=new_end_frame
        )

    def ground(
            self,
            video_path: str,
            text_query: str,
            sample_rate: int = 1,
            refine: bool = True,
            **kwargs
    ) -> List[TemporalSegment]:
        """
        Main grounding function: find temporal segments matching text query.

        Args:
            video_path: Path to video file
            text_query: Text description to locate
            sample_rate: Extract every Nth frame (higher = faster but less precise)
            refine: Whether to refine segment boundaries
            **kwargs: Additional arguments for find_segments

        Returns:
            List of TemporalSegments sorted by score descending
        """
        print(f"Processing video: {video_path}")
        print(f"Query: '{text_query}'")

        # Extract frames
        print("Extracting frames...")
        frames, fps, total_frames = self.extract_frames(video_path, sample_rate)
        print(f"Extracted {len(frames)} frames from {total_frames} total (fps={fps:.2f})")

        # Compute similarity scores
        print("Computing similarity scores...")
        scores = self.compute_similarity_scores(frames, text_query)
        print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

        # Find segments
        print("Finding temporal segments...")
        segments = self.find_segments(scores, fps, sample_rate, **kwargs)

        if not segments:
            print("No valid segments found above threshold")
            return []

        # Optionally refine boundaries
        if refine:
            print("Refining segment boundaries...")
            segments = [self.refine_boundaries(seg, scores, fps, sample_rate)
                        for seg in segments]

        print(f"\nFound {len(segments)} segment(s):")
        for i, seg in enumerate(segments, 1):
            duration = seg.end_time - seg.start_time
            print(f"  Segment {i}: {seg.start_time:.2f}s - {seg.end_time:.2f}s "
                  f"(duration={duration:.2f}s, score={seg.score:.3f})")

        return segments


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def extract_segments(video_path: str, user_prompt: str) -> dict:
    """
    Find temporal segments matching a user's query in a video.

    Args:
        video_path: Video filename (will be loaded from ../assets/)
        user_prompt: Text description of what to find

    Returns:
        Dict with "matches" list containing start_time, end_time, confidence, reasoning
    """
    full_path = f"../assets/{video_path}"

    grounder = TemporalVideoGrounder()
    segments = grounder.ground(
        full_path,
        user_prompt,
        sample_rate=1,
        min_duration=0.5,
        max_duration=30.0,
        gap_tolerance=2.0,
    )

    # Normalize scores to 0-1 confidence range
    if segments:
        max_score = max(seg.score for seg in segments)
        min_score = min(seg.score for seg in segments)
        score_range = max_score - min_score if max_score > min_score else 1.0

    matches = []
    for seg in segments:
        confidence = (seg.score - min_score) / score_range if score_range > 0 else 1.0
        confidence = round(max(0.0, min(1.0, confidence)), 2)
        matches.append({
            "start_time": format_timestamp(seg.start_time),
            "end_time": format_timestamp(seg.end_time),
            "confidence": confidence,
            "reasoning": f"CLIP similarity score {seg.score:.3f} for '{user_prompt}'"
        })

    return {"matches": matches}


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