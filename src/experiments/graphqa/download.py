from huggingface_hub import snapshot_download
import os
import json
import glob

base_dir = "./src/experiments/graphqa/hf_dataset"

print("Downloading raw JSON files and folder structure...")

# This will download the entire dataset repository as-is
local_path = snapshot_download(
    repo_id="baharef/GraphQA", 
    repo_type="dataset",
    local_dir=base_dir  # It will create this folder in your current working directory
)

print(f"Done! All your raw JSON files are now saved in: {local_path}")


# Find every single .json file inside that folder and its subfolders
json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)

print(f"Found {len(json_files)} JSON files. Formatting them now...")

for file_path in json_files:
    try:
        # Open and read the messy, single-line JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Overwrite the exact same file, but this time with indent=4 for beautiful formatting
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {file_path}. Skipping.")

print("Done! All your JSON files are now beautifully indented.")