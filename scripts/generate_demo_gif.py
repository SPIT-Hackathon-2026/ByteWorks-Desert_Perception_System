import os
from PIL import Image

def generate_gif(source_dir, output_path, patterns, duration=300):
    files = []
    for f in sorted(os.listdir(source_dir)):
        if any(f.startswith(p) for p in patterns) and f.endswith(".png"):
            files.append(os.path.join(source_dir, f))
    
    if not files:
        print("No matching files found.")
        return

    print(f"Found {len(files)} images for GIF.")
    images = [Image.open(f) for f in files]
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved to {output_path}")

if __name__ == "__main__":
    SOURCE = "umixformer_pipeline/predictions/comparisons/"
    OUTPUT = "frontend/public/demo.gif"
    PATTERNS = ["ww", "mt", "cc"]
    
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    generate_gif(SOURCE, OUTPUT, PATTERNS)
