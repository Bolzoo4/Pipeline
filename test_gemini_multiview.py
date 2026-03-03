"""
Local Test Script: Gemini 3.1 Multiview Generator (Nano Banana Pro)

This script generates 6 views of a jewelry item and stitches them into a 
640x960 grid for InstantMesh LRM.

Requirements:
    pip install google-genai Pillow

Environment Variables:
    export GOOGLE_CLOUD_PROJECT="virtual-try-on-488619"
    export GOOGLE_CLOUD_LOCATION="europe-west1"
    export GOOGLE_GENAI_USE_VERTEXAI="True"
    
Usage:
    python test_gemini_multiview.py input_ring.jpg output_grid.png
"""

import os
import sys
from pathlib import Path

# Add stages to path so we can reuse the logic
sys.path.append(str(Path(__file__).parent))

from stages.nanobanana_multiview import generate_multiview_grid

def run_test():
    if len(sys.argv) < 3:
        print("❌ Usage: python test_gemini_multiview.py <input_image> <output_grid_name>")
        print("Example: python test_gemini_multiview.py ring.jpg grid.png")
        return

    input_img = sys.argv[1]
    output_grid = sys.argv[2]
    
    # Check environment
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("⚠ WARNING: GOOGLE_CLOUD_PROJECT is not set.")
        print("Fix: export GOOGLE_CLOUD_PROJECT=\"virtual-try-on-488619\"")
        
    print(f"🚀 Starting Multiview Generation for: {input_img}")
    
    try:
        generate_multiview_grid(
            input_image_path=input_img,
            output_image_path=output_grid,
            category="jewelry",
            project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", "virtual-try-on-488619"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west1")
        )
        print(f"✅ SUCCESS! Grid saved to: {output_grid}")
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    run_test()
