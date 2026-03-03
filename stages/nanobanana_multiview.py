"""
Nano Banana Pro (Gemini 3) Multiview Generator.
Strictly configured to use gemini-3-pro-image-preview only.
"""

import os
import time
from io import BytesIO
from PIL import Image
import concurrent.futures

# Make sure google-genai and tenacity are installed
try:
    from google import genai
    from google.genai.types import GenerateContentConfig, Modality, HttpOptions
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    raise ImportError("pip install google-genai tenacity")

def generate_single_view(client, input_img, view_name: str, azimuth: int, elevation: int, category: str = "jewelry", model="gemini-3-pro-image-preview"):
    """
    Call ONLY Gemini 3 Pro Image (Nano Banana Pro) to generate a single view.
    Uses exponential backoff for 429 errors.
    """
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(Exception), # Standard for Vertex AI exceptions
        reraise=True
    )
    def call_api():
        prompt = (
            f"You are a professional 3D product photographer. Generate a high-resolution, photorealistic "
            f"{view_name} view of the SAME {category} shown in the reference image. "
            f"CRITICAL: Rotate the object or the camera so that we see it from an angle of "
            f"AZIMUTH={azimuth} degrees and ELEVATION={elevation} degrees relative to the FRONT view. "
            f"Keep the materials (gold, diamonds), textures, and lighting IDENTICAL to the source. "
            f"The object must be perfectly centered on a pure white background (#FFFFFF) with NO shadows or floor. "
            f"Generate ONLY the image of the object from this new 3D perspective."
        )
        
        print(f"   [NanoBanana] Generating {view_name} (Az:{azimuth}, El:{elevation}) using {model} (v1beta1)...")
        
        response = client.models.generate_content(
            model=model,
            contents=[
                input_img,
                prompt
            ],
            config=GenerateContentConfig(
                response_modalities=[Modality.IMAGE],
                temperature=0.5,
            ),
        )
        
        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                img = Image.open(BytesIO(part.inline_data.data)).convert("RGB")
                # Resize each view to 320x320 as expected by InstantMesh's grid
                img = img.resize((320, 320), Image.LANCZOS)
                return img
                
        raise ValueError(f"No image returned for view {view_name}")

    return call_api()

def generate_multiview_grid(input_image_path: str, output_image_path: str, category: str = "jewelry", project_id: str = None, location: str = "us-central1"):
    """
    Generates 6 views and stitches them into a 640x960 grid.
    Strictly uses Gemini 3 Pro Image with v1beta1.
    """
    
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    
    # Force v1beta1 via HttpOptions
    client = genai.Client(
        http_options=HttpOptions(api_version='v1beta1')
    )
    
    # Load input image
    input_img = Image.open(input_image_path).convert("RGB")
    
    # Exactly Zero123++ canonical poses
    views_spec = [
        {"name": "front-right high", "az": 30,  "el": 20},
        {"name": "right low",        "az": 90,  "el": -10},
        {"name": "back-right high",  "az": 150, "el": 20},
        {"name": "back-left low",    "az": 210, "el": -10},
        {"name": "left high",        "az": 270, "el": 20},
        {"name": "front-left low",   "az": 330, "el": -10}
    ]
    
    print(f"   ℹ Querying Nano Banana Pro (Gemini 3) for 6 views in parallel...")
    start_time = time.time()
    generated_views = [None] * 6
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_idx = {
            executor.submit(generate_single_view, client, input_img, spec["name"], spec["az"], spec["el"], category): idx
            for idx, spec in enumerate(views_spec)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                generated_views[idx] = future.result()
            except Exception as e:
                print(f"   ⚠ Nano Banana failed for {views_spec[idx]['name']}: {e}")
                # Use a white placeholder on failure to avoid crashing the rest of the pipe
                generated_views[idx] = Image.new('RGB', (320, 320), (255, 255, 255))
                
    print(f"   ✓ Generated 6 views in {time.time() - start_time:.2f}s")
    
    # Stitch into 2x3 grid
    grid_img = Image.new('RGB', (640, 960), (255, 255, 255))
    for idx, img in enumerate(generated_views):
        row = idx // 2
        col = idx % 2
        grid_img.paste(img, (col * 320, row * 320))
        
    grid_img.save(output_image_path, "PNG")
    print(f"   ✓ Grid saved to {output_image_path}")
    return output_image_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        generate_multiview_grid(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python nanobanana_multiview.py input.jpg output.png")
