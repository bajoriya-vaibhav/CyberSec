"""
Audio processing utilities for extraction from video.
"""
import os
import moviepy.editor as mp
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio extraction from video files."""
    
    @staticmethod
    def extract_audio(video_path: str, output_path: str = "temp_audio.wav") -> Optional[str]:
        """
        Extract audio track from video file.
        
        Args:
            video_path: Path to video file
            output_path: Where to save extracted audio
            
        Returns:
            Path to extracted audio file, or None if no audio track exists
        """
        try:
            logger.info(f"Extracting audio from: {video_path}")
            video = mp.VideoFileClip(video_path)
            
            if video.audio is None:
                logger.warning("No audio track found in video")
                video.close()
                return None
            
            # Extract audio
            video.audio.write_audiofile(output_path, verbose=False, logger=None)
            video.close()
            
            # Verify file was created and has size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Audio extracted successfully to: {output_path}")
                return output_path
            else:
                logger.warning("Audio file was not created properly")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None
    
    @staticmethod
    def cleanup_audio(audio_path: str) -> None:
        """Clean up temporary audio file."""
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.debug(f"Cleaned up audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Could not clean up audio file: {e}")
    
    @staticmethod
    def get_audio_info(audio_path: str) -> dict:
        """Get audio file metadata."""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'samples': len(y)
            }
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {}
