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

def get_view_prompt(view_name: str, category: str = "jewelry") -> str:
    return (
        f"Generate a photorealistic {view_name} view of this exact {category}. "
        "The object must be isolated on an absolute pure white background (#FFFFFF). "
        "Keep the exact same lighting, material, color, and scale. "
        "Do NOT add any shadows on the floor. Center the object perfectly."
    )

def generate_single_view(client, image_part, view_name: str, category: str = "jewelry", model="gemini-3.1-flash-image-preview"):
    """Call Gemini to generate a single view from the reference image."""
    prompt = get_view_prompt(view_name, category)
    
    # We pass the original image + the prompt instructing it to rotate/change view
    # Note: Using gemini-3.1-flash-image-preview as it's typically faster for 6 parallel queries
    print(f"   [NanoBanana] Generating {view_name} view...")
    response = client.models.generate_content(
        model=model,
        contents=[
            image_part,
            prompt
        ],
        config=GenerateContentConfig(
            response_modalities=[Modality.IMAGE],
            temperature=0.0,
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

def generate_multiview_grid(input_image_path: str, output_image_path: str, category: str = "jewelry", project_id: str = None, location: str = "us-central1"):
    """Generates 6 views and stitches them into a 640x960 grid."""
    
    # Initialize Vertex AI Client (assumes GOOGLE_APPLICATION_CREDENTIALS or gcloud auth is setup)
    # The new SDK uses GOOGLE_CLOUD_PROJECT + GOOGLE_GENAI_USE_VERTEXAI
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    
    client = genai.Client()
    
    # Load input image
    input_img = Image.open(input_image_path).convert("RGB")
    
    views_needed = [
        "front-right 30 degree", # View 1 (approx Zero123++ spec)
        "right side",            # View 2
        "back-right 150 degree", # View 3
        "back-left 210 degree",  # View 4
        "left side",             # View 5
        "front-left 330 degree"  # View 6
    ]
    
    print(f"   ℹ Querying Nano Banana Pro (Gemini 3) for 6 views in parallel...")
    start_time = time.time()
    
    # Run all 6 queries in parallel
    generated_views = [None] * 6
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_idx = {
            executor.submit(generate_single_view, client, input_img, view_name, category): idx
            for idx, view_name in enumerate(views_needed)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                img = future.result()
                generated_views[idx] = img
            except Exception as e:
                print(f"   ⚠ Nano Banana failed for view {views_needed[idx]}: {e}")
                # Fallback: just use input image if generation fails
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
