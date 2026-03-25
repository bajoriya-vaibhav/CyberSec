"""
Test script to verify Gemini API integration works correctly.
"""
from google import genai
from google.genai import types
from PIL import Image
import io

# Initialize client
client = genai.Client(api_key="AIzaSyCSz7HeKZI3YVKqortMu_MRGdXbPY06nS4")

# Create a simple test image (100x100 red square)
test_image = Image.new('RGB', (100, 100), color='red')

# Convert to bytes
buf = io.BytesIO()
test_image.save(buf, format="PNG")
image_bytes = buf.getvalue()

print("Testing Gemini API with image...")

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(text="What color is this image? Reply with just the color name."),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=image_bytes
                        )
                    )
                ]
            )
        ]
    )
    
    print(f"✅ SUCCESS! Gemini API is working!")
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
