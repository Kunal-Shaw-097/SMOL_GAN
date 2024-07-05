import os
import shutil

output_dir = "LSUN/"

os.makedirs(output_dir, exist_ok=True)
i = 0
j = set()
for paren, mid, files in os.walk("archive/"):
    for file in files :
        full_path = os.path.join(paren, file)

        if full_path.endswith(".jpg"):
            j.add(full_path)
            i += 1
            file_path = full_path.split("/")[-1]
            save_path = os.path.join(output_dir, file_path)
            shutil.copy(full_path, save_path)

print(f"Done! {i} images")
print(len(j))
