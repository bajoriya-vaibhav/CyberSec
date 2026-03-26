"""
Mock backend server for Deepfake Capture app testing.
Saves received video frames (JPEG) and audio segments (WAV) to disk.

Usage:
    python mock_server.py

Files saved to: ./received_data/
    frames/  -> frame_001.jpg, frame_002.jpg, ...
    audio/   -> audio_001.wav, audio_002.wav, ...
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import random
import time
import os
import re

# Create output directories
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "received_data")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

frame_counter = 0
audio_counter = 0


def parse_multipart(body, content_type):
    """Parse multipart/form-data and return dict of {field_name: bytes}."""
    parts = {}

    # Extract boundary from content-type
    match = re.search(r'boundary=(.+?)(?:;|$)', content_type)
    if not match:
        return parts

    boundary = match.group(1).strip()
    # Handle quoted boundaries
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    boundary_bytes = boundary.encode('utf-8')
    delimiter = b'--' + boundary_bytes

    # Split body by boundary
    segments = body.split(delimiter)

    for segment in segments:
        if not segment or segment == b'--\r\n' or segment == b'--':
            continue

        # Split headers from content
        header_end = segment.find(b'\r\n\r\n')
        if header_end == -1:
            continue

        header_section = segment[:header_end].decode('utf-8', errors='replace')
        content = segment[header_end + 4:]

        # Remove trailing \r\n
        if content.endswith(b'\r\n'):
            content = content[:-2]

        # Extract field name and filename
        name_match = re.search(r'name="(.+?)"', header_section)
        if name_match:
            field_name = name_match.group(1)
            parts[field_name] = content

    return parts


class PredictHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global frame_counter, audio_counter

        content_length = int(self.headers.get('Content-Length', 0))
        content_type = self.headers.get('Content-Type', '')
        body = self.rfile.read(content_length)

        print(f"\n{'='*60}")
        print(f"[{time.strftime('%H:%M:%S')}] POST {self.path}  ({content_length} bytes)")

        # Parse multipart data
        parts = parse_multipart(body, content_type)

        # Save video frame
        if 'video_frame' in parts:
            frame_counter += 1
            filename = f"frame_{frame_counter:04d}.jpg"
            filepath = os.path.join(FRAMES_DIR, filename)
            with open(filepath, 'wb') as f:
                f.write(parts['video_frame'])
            size_kb = len(parts['video_frame']) / 1024
            print(f"  [VIDEO] Saved {filename} ({size_kb:.1f} KB)")
        else:
            print(f"  [VIDEO] No video frame in request")

        # Save audio segment
        if 'audio_segment' in parts:
            audio_counter += 1
            filename = f"audio_{audio_counter:04d}.wav"
            filepath = os.path.join(AUDIO_DIR, filename)
            with open(filepath, 'wb') as f:
                f.write(parts['audio_segment'])
            size_kb = len(parts['audio_segment']) / 1024
            print(f"  [AUDIO] Saved {filename} ({size_kb:.1f} KB)")
        else:
            print(f"  [AUDIO] No audio segment in request")

        # Generate mock prediction
        is_fake = random.random() > 0.5
        prediction = "Fake" if is_fake else "Real"
        confidence = round(random.uniform(0.70, 0.99), 2)

        response = {"prediction": prediction, "confidence": confidence}
        print(f"  [REPLY] {prediction} (confidence: {confidence})")
        print(f"  Totals: {frame_counter} frames, {audio_counter} audio segments saved")

        response_bytes = json.dumps(response).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Deepfake Detection Mock Server is running')

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    port = 7860
    server = HTTPServer(('0.0.0.0', port), PredictHandler)
    print(f"Mock Deepfake Detection Server")
    print(f"  Listening on: http://0.0.0.0:{port}")
    print(f"  Emulator URL: http://10.0.2.2:{port}")
    print(f"  Saving to:    {OUTPUT_DIR}")
    print(f"\n  Waiting for data...\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nServer stopped. Total saved: {frame_counter} frames, {audio_counter} audio segments")
