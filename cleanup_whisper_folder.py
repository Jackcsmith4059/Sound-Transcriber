import os

# Folder where your Whisper model files are stored
whisper_folder = os.path.expanduser("/Users/jack/Desktop/Sound Transcriptor/whisper")

# Files to KEEP
files_to_keep = {
    "pytorch_model.bin",
    "config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "normalizer.json",
    "README.md",  # Optional: Keep the readme, very small
    "generation_config.json",  # Optional: Advanced settings, very small
}

# Go through each file
for filename in os.listdir(whisper_folder):
    file_path = os.path.join(whisper_folder, filename)
    
    if os.path.isfile(file_path) and filename not in files_to_keep:
        print(f"Deleting unnecessary file: {filename}")
        os.remove(file_path)

print("✅ Cleanup complete! Whisper folder is now light and clean.")