"""
Video processing utilities for frame extraction.
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video frame extraction with multiple strategies."""
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video metadata."""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            return {
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': duration
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}
    
    @staticmethod
    def extract_frames_uniform(video_path: str, num_frames: int = 10) -> List[Image.Image]:
        """
        Extract frames uniformly distributed across the video.
        This ensures we sample the entire video duration.
        """
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                logger.warning("Could not get total frame count")
                return []
            
            # Calculate frame indices uniformly across video
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            logger.info(f"Extracting {num_frames} frames from {total_frames} total frames")
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                else:
                    logger.warning(f"Failed to read frame at index {idx}")
            
            cap.release()
            logger.info(f"Successfully extracted {len(frames)} frames")
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        
        return frames
    
    @staticmethod
    def extract_frames_keyframes(video_path: str, max_frames: int = 10) -> List[Image.Image]:
        """
        Extract key frames based on scene changes (more advanced).
        Falls back to uniform extraction if detection fails.
        """
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            prev_frame = None
            frame_count = 0
            threshold = 30.0  # Threshold for scene change detection
            
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Check for scene change
                if prev_frame is not None:
                    diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                      cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
                    mean_diff = np.mean(diff)
                    
                    if mean_diff > threshold:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame_rgb))
                        logger.debug(f"Key frame detected at frame {frame_count}")
                else:
                    # Always include first frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                
                prev_frame = frame.copy()
            
            cap.release()
            
            # Fallback to uniform if too few keyframes detected
            if len(frames) < max_frames // 2:
                logger.info("Too few keyframes, falling back to uniform extraction")
                return VideoProcessor.extract_frames_uniform(video_path, max_frames)
            
            logger.info(f"Extracted {len(frames)} key frames")
            
        except Exception as e:
            logger.error(f"Error in keyframe extraction: {e}")
            # Fallback to uniform
            return VideoProcessor.extract_frames_uniform(video_path, max_frames)
        
        return frames
    
    @staticmethod
    def extract_frames(video_path: str, num_frames: int = 10, 
                      strategy: str = 'uniform') -> List[Image.Image]:
        """
        Main entry point for frame extraction.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            strategy: 'uniform' or 'keyframes'
        """
        if strategy == 'keyframes':
            return VideoProcessor.extract_frames_keyframes(video_path, num_frames)
        else:
            return VideoProcessor.extract_frames_uniform(video_path, num_frames)
