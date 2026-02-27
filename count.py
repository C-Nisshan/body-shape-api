import os

# Path to your main directory
# base_dir = "woman-body-shape-522"

'''
# Function to count files in each subdirectory
def count_images_in_dir(directory):
    summary = {}
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            # Count only files (you can filter by image extensions if needed)
            count = len([f for f in os.listdir(subdir_path) 
                         if os.path.isfile(os.path.join(subdir_path, f))])
            summary[subdir] = count
    return summary

# Iterate through main directories: train, valid, test
for split in ["train", "valid", "test"]:
    split_path = os.path.join(base_dir, split)
    if os.path.exists(split_path):
        counts = count_images_in_dir(split_path)
        print(f"\nDirectory: {split} (Total subdirs: {len(counts)})")
        for category, num_images in counts.items():
            print(f"  {category}: {num_images} images")
'''

'''
base_dir = "v2_woman-body-shape-dataset"

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

for root, dirs, files in os.walk(base_dir):
    
    image_count = sum(
        1 for file in files 
        if file.lower().endswith(image_extensions)
    )
    
    if image_count > 0:
        # Get relative path from base directory
        relative_path = os.path.relpath(root, base_dir)
        
        # Calculate depth level
        level = relative_path.count(os.sep)
        indent = "    " * level
        
        print(f"{indent}{relative_path} --> {image_count}")
'''

# Get absolute path of current directory
base_dir = os.path.abspath(".")

# Extract only the base folder name
base_name = os.path.basename(base_dir)

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

for root, dirs, files in os.walk(base_dir):
    
    image_count = sum(
        1 for file in files 
        if file.lower().endswith(image_extensions)
    )
    
    if image_count > 0:
        relative_path = os.path.relpath(root, base_dir)
        
        # If it's the root itself
        if relative_path == ".":
            display_path = base_name
        else:
            display_path = os.path.join(base_name, relative_path)
        
        level = relative_path.count(os.sep) if relative_path != "." else 0
        indent = "    " * level
        
        print(f"{indent}{display_path} --> {image_count}")