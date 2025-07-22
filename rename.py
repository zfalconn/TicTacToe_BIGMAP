import os
from pathlib import Path

def rename_images_by_time(folder_path):
    folder = Path(folder_path)
    images = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']],
        key=lambda f: f.stat().st_mtime
    )

    for i, img_path in enumerate(images, start=1):
        new_name = f"{i}{img_path.suffix.lower()}"
        new_path = folder / new_name
        print(f"Renaming: {img_path.name} → {new_name}")
        img_path.rename(new_path)

# Example usage
rename_images_by_time("test_imgs")
