from PIL import Image
import sys

def merge_images(img1_path, img2_path, output_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Get dimensions
    w1, h1 = img1.size
    w2, h2 = img2.size
    
    # Combine side by side
    combined = Image.new('RGB', (w1 + w2 + 10, max(h1, h2)), (200, 200, 200))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (w1 + 10, 0))
    
    combined.save(output_path)
    print(f"Saved merged image to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 merge.py img1 img2 output")
    else:
        merge_images(sys.argv[1], sys.argv[2], sys.argv[3])
