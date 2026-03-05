"""
Nano Banana Multiview Generator (Gemini 2.5 Flash Image).
Generates 4 orthographic views (Front, Right, Back, Left)
and stitches them into a 2x2 grid for Unique3D.
"""

import os
import time
from io import BytesIO
from PIL import Image
import concurrent.futures

try:
    from google import genai
    from google.genai.types import GenerateContentConfig, Modality
except ImportError:
    raise ImportError("pip install google-genai")


def generate_single_view(client, input_img, view_name: str, azimuth: int, elevation: int, category: str = "jewelry", model="gemini-2.5-flash-image"):
    """
    Call Gemini 2.5 Flash Image to generate a single view.
    """
    prompt = (
        f"You are a professional 3D product photographer. Generate a high-resolution, photorealistic "
        f"{view_name} view of the SAME {category} shown in the reference image. "
        f"CRITICAL: Rotate the object or the camera so that we see it from an angle of "
        f"AZIMUTH={azimuth} degrees and ELEVATION={elevation} degrees relative to the FRONT view. "
        f"Keep the materials (gold, diamonds), textures, and lighting IDENTICAL to the source. "
        f"The object must be perfectly centered on a pure white background (#FFFFFF) with NO shadows or floor. "
        f"Generate ONLY the image of the object from this new 3D perspective."
    )
    
    print(f"   [NanoBanana] Generating {view_name} (Az:{azimuth}, El:{elevation}) using {model}...")
    
    response = client.models.generate_content(
        model=model,
        contents=[input_img, prompt],
        config=GenerateContentConfig(
            response_modalities=[Modality.IMAGE],
            temperature=0.5,
        ),
    )
    
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            img = Image.open(BytesIO(part.inline_data.data)).convert("RGB")
            return img
            
    raise ValueError(f"No image returned for view {view_name}")


def generate_multiview_grid(input_image_path: str, output_grid_path: str, category: str = "jewelry"):
    """
    Generates 4 orthographic views (Front, Right, Back, Left) using gemini-2.5-flash-image
    and stitches them into a 2x2 grid (1024x1024) for Unique3D ISOMER reconstruction.
    """
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    client = genai.Client()
    
    input_img = Image.open(input_image_path).convert("RGB")
    
    views = [
        {"name": "front", "azimuth": 0, "elevation": 0},
        {"name": "right", "azimuth": 90, "elevation": 0},
        {"name": "back", "azimuth": 180, "elevation": 0},
        {"name": "left", "azimuth": 270, "elevation": 0},
    ]

    print(f"   ℹ Querying Nano Banana for {len(views)} views...")
    start_time = time.time()
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(views)) as executor:
        futures = {
            executor.submit(
                generate_single_view, 
                client, input_img, v["name"], v["azimuth"], v["elevation"], category
            ): v for v in views
        }
        
        for future in concurrent.futures.as_completed(futures):
            view_info = futures[future]
            try:
                img = future.result()
                results[view_info["name"]] = img
            except Exception as e:
                print(f"   ⚠ Failed for {view_info['name']}: {e}")

    # Sequential retry for 429s
    failed_views = [v for v in views if v["name"] not in results]
    if failed_views:
        print(f"   ℹ Retrying {len(failed_views)} failed views sequentially (10s delay)...")
        for v in failed_views:
            time.sleep(10)
            try:
                img = generate_single_view(client, input_img, v["name"], v["azimuth"], v["elevation"], category)
                results[v["name"]] = img
                print(f"   ✓ {v['name']} recovered")
            except Exception as e:
                print(f"   ❌ {v['name']} retry failed: {e}")

    if len(results) < len(views):
        raise RuntimeError(f"Could only generate {len(results)}/{len(views)} views.")

    print(f"   ✓ Generated all views in {time.time() - start_time:.2f}s")

    # Stitch into 2x2 grid (1024x1024 total, each cell 512x512)
    grid = Image.new('RGB', (1024, 1024), (255, 255, 255))
    order = ["front", "right", "back", "left"]
    for i, name in enumerate(order):
        img = results[name].resize((512, 512), Image.LANCZOS)
        x = (i % 2) * 512
        y = (i // 2) * 512
        grid.paste(img, (x, y))
    
    grid.save(output_grid_path)
    print(f"   ✓ 4-view 2x2 grid saved: {output_grid_path}")
    return output_grid_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        generate_multiview_grid(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python nanobanana_multiview.py input.jpg output.png")
