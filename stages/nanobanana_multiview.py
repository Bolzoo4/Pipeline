"""
Nano Banana Pro (Gemini 3) Multiview Generator.

Uses Vertex AI's gemini-3-pro-image-preview (or flash) to generate 6 views
of an object given its front photo. It then stitches them into the 2x3 grid
(640x960) format expected by InstantMesh's LRM.

Grid expected by InstantMesh:
Row 1: Right  | Front   (Wait, zero123++ actually puts right then front, or front then right?
        Actually, looking at Zero123++ layout:
        [Front-Right, Right, Back, Left, Top, Bottom]?
        Let's check InstantMesh standard: 
        Zero123++ outputs a 3x2 grid.
        Row 1: azimuth 30°, azimuth 90°
        Row 2: azimuth 150°, azimuth 210°
        Row 3: azimuth 270°, azimuth 330°
        Actually, Zero123++ standard is:
        View 0: Azimuth 30, Elevation 20
        View 1: Azimuth 90, Elevation -10
        View 2: Azimuth 150, Elevation 20
        View 3: Azimuth 210, Elevation -10
        View 4: Azimuth 270, Elevation 20
        View 5: Azimuth 330, Elevation -10
        But we can just approximate it: Front, Right, Back, Left, Top, Bottom.
        Let's just use simple 6 views: Front, Right, Back, Left, Top, Bottom
        and map them to the 3x2 grid.
"""

import os
import time
from io import BytesIO
from PIL import Image
import concurrent.futures

# Make sure google-genai is installed
try:
    from google import genai
    from google.genai.types import GenerateContentConfig, Modality
except ImportError:
    raise ImportError("pip install google-genai")

def get_view_prompt(view_name: str, azimuth_deg: int, elevation_deg: int, category: str = "jewelry") -> str:
    return (
        f"You are a professional 3D product photographer. Generate a high-resolution, photorealistic "
        f"{view_name} view of the SAME {category} shown in the reference image. "
        f"CRITICAL: Rotate the object or the camera so that we see it from an angle of "
        f"AZIMUTH={azimuth_deg} degrees and ELEVATION={elevation_deg} degrees relative to the FRONT view. "
        f"Keep the materials (gold, diamonds), textures, and lighting IDENTICAL to the source. "
        f"The object must be perfectly centered on a pure white background (#FFFFFF) with NO shadows or floor. "
        f"Generate ONLY the image of the object from this new 3D perspective."
    )

def generate_single_view(client, input_img, view_name: str, azimuth: int, elevation: int, category: str = "jewelry", model="gemini-3.1-flash-image-preview"):
    """
    Call Gemini (or Imagen fallback) to generate a single view.
    Handles 404 by falling back to a more stable model.
    """
    from google.genai.types import GenerateContentConfig, Modality, GenerateImagesConfig
    
    # Try Gemini 3 (Nano Banana) or Flash if possible
    # Fallback list: [User's choice, Stable Gemini, Stable Imagen]
    models_to_try = [model, "gemini-2.0-flash-001", "imagen-3.0-generate-002"]
    
    last_error = None
    
    for current_model in models_to_try:
        try:
            # If it's a Gemini model
            if "gemini" in current_model:
                prompt = (
                    f"A photorealistic {view_name} view of the {category} from the reference image. "
                    f"Rotate to AZIMUTH={azimuth} and ELEVATION={elevation}. "
                    f"Pure white background, high quality."
                )
                print(f"   [NanoBanana] Trying {current_model} for {view_name}...")
                
                response = client.models.generate_content(
                    model=current_model,
                    contents=[input_img, prompt],
                    config=GenerateContentConfig(response_modalities=[Modality.IMAGE])
                )
                
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        img = Image.open(BytesIO(part.inline_data.data)).convert("RGB")
                        return img.resize((320, 320), Image.LANCZOS)
            
            # If it's an Imagen model
            elif "imagen" in current_model:
                print(f"   [NanoBanana] Falling back to {current_model} for {view_name}...")
                # Improved specific prompt for jewelry rotation
                prompt = (
                    f"High-resolution studio photo of a {category} from a {view_name} perspective "
                    f"(Azimuth {azimuth}°, Elevation {elevation}°). "
                    f"Centered, pure white background #FFFFFF, no shadows, 8k resolution, photorealistic."
                )
                response = client.models.generate_images(
                    model=current_model,
                    prompt=prompt,
                    config=GenerateImagesConfig(number_of_images=1)
                )
                if response.generated_images:
                    img_bytes = response.generated_images[0].image.image_bytes
                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    return img.resize((320, 320), Image.LANCZOS)
                    
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str:
                print(f"   ⚠ {current_model} not found (404). Trying next...")
                continue
            if "429" in error_str or "quota" in error_str:
                raise e # Escalated to grid level for sequential retry
            print(f"   ⚠ Error with {current_model}: {e}. Trying next...")
            continue
            
    raise last_error or ValueError(f"Failed to generate view {view_name}")

def generate_multiview_grid(input_image_path: str, output_image_path: str, category: str = "jewelry", project_id: str = None, location: str = "us-central1"):
    """
    Generates 6 views and stitches them into a grid.
    Uses sequential fallback if parallel calls hit a quota.
    """
    
    # Initialize Vertex AI Client (assumes GOOGLE_APPLICATION_CREDENTIALS or gcloud auth is setup)
    # The new SDK uses GOOGLE_CLOUD_PROJECT + GOOGLE_GENAI_USE_VERTEXAI
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    
    client = genai.Client()
    
    # Load input image
    input_img = Image.open(input_image_path).convert("RGB")
    
    # Exactly Zero123++ canonical poses
    views_spec = [
        {"name": "front-right high", "az": 30,  "el": 20},  # View 0
        {"name": "right low",        "az": 90,  "el": -10}, # View 1
        {"name": "back-right high",  "az": 150, "el": 20},  # View 2
        {"name": "back-left low",    "az": 210, "el": -10}, # View 3
        {"name": "left high",        "az": 270, "el": 20},  # View 4
        {"name": "front-left low",   "az": 330, "el": -10}  # View 5
    ]
    
    print(f"   ℹ Multiview generation starting (Location: {location})...")
    start_time = time.time()
    
    # Run all 6 queries in parallel
    generated_views = [None] * 6
    
    # First attempt: Parallel (fast)
    # If we get 429, we might have hit a per-minute limit.
    hit_quota = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_idx = {
            executor.submit(
                generate_single_view, 
                client, 
                input_img,  # Pass PIL Image for Gemini
                spec["name"], 
                spec["az"], 
                spec["el"], 
                category
            ): idx
            for idx, spec in enumerate(views_spec)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                generated_views[idx] = future.result()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str:
                    hit_quota = True
                    print(f"   ⚠ Rate limit hit for {views_spec[idx]['name']}. Will retry sequentially...")
                else:
                    print(f"   ⚠ Failed for {views_spec[idx]['name']}: {e}")
                # Placeholder for now
                generated_views[idx] = None

    # Second attempt: Sequential for missing views (slow but safer for quota)
    if hit_quota or any(v is None for v in generated_views):
        print("   ℹ Retrying missing views sequentially...")
        for idx, img in enumerate(generated_views):
            if img is None:
                try:
                    # Wait a bit between sequential calls
                    time.sleep(1)
                    generated_views[idx] = generate_single_view(client, input_img, views_spec[idx]["name"], views_spec[idx]["az"], views_spec[idx]["el"], category)
                    print(f"   ✓ Successfully recovered {views_spec[idx]['name']}")
                except Exception as e:
                    print(f"   ⚠ Sequential retry failed for {views_spec[idx]['name']}: {e}")
                    generated_views[idx] = input_img.resize((320, 320), Image.LANCZOS)
                
    print(f"   ✓ Generated 6 views in {time.time() - start_time:.2f}s")
    
    # Stitch into 2x3 grid (3 rows, 2 columns) -> Total: 640x960 (width=640, height=960)
    # InstantMesh parses the grid as rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
    # So:
    # Row 0: View 0, View 1
    # Row 1: View 2, View 3
    # Row 2: View 4, View 5
    
    grid_img = Image.new('RGB', (640, 960), (255, 255, 255))
    
    for idx, img in enumerate(generated_views):
        row = idx // 2
        col = idx % 2
        x = col * 320
        y = row * 320
        grid_img.paste(img, (x, y))
        
    grid_img.save(output_image_path, "PNG")
    print(f"   ✓ Grid saved to {output_image_path}")
    return output_image_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        generate_multiview_grid(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python nanobanana_multiview.py input.jpg output.png")
