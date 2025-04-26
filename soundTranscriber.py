import os
import sys
import whisper
import ffmpeg
import tqdm
from datetime import datetime

# === SETTINGS ===
input_folder = "input_audio"    # Folder where you place audio files
output_folder = "output_texts"  # Folder where transcriptions will be saved
model_folder = "whisper"        # Local whisper folder now
rename_output_files = True      # If True, renames output files with detected language and timestamp

# === CHECK IF MODEL EXISTS LOCALLY ===
model_path = os.path.join(model_folder, "pytorch_model.bin")
if not os.path.exists(model_path):
    print(f"ERROR: Model file 'pytorch_model.bin' not found in {model_folder}/")
    print("Please ensure that the model is downloaded correctly.")
    sys.exit(1)

# === LOAD WHISPER MODEL ===
print(f"Loading Whisper model from '{model_folder}'...")
model = whisper.load_model("medium")

# === CREATE INPUT/OUTPUT FOLDERS IF NEEDED ===
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# === START TRANSCRIBING FILES ===
audio_extensions = (".mp3", ".wav", ".m4a", ".flac", ".ogg")

audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(audio_extensions)]

if not audio_files:
    print(f"No audio files found in '{input_folder}' folder.")
    sys.exit(0)

print(f"Found {len(audio_files)} audio files. Starting transcription...")

for filename in tqdm.tqdm(audio_files, desc="Transcribing"):
    input_path = os.path.join(input_folder, filename)

    try:
        # Transcribe audio
        result = model.transcribe(input_path)
        text = result["text"].strip()
        language = result.get("language", "unknown")

        # Determine output filename
        base_filename = os.path.splitext(filename)[0]
        if rename_output_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_filename}_{language}_{timestamp}.txt"
        else:
            output_filename = f"{base_filename}.txt"

        output_path = os.path.join(output_folder, output_filename)

        # Save transcription
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Transcribed: {filename} (Detected language: {language})")

    except Exception as e:
        # Handle errors gracefully
        error_log_path = os.path.join(output_folder, "error_log.txt")
        with open(error_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"Failed to transcribe {filename} at {datetime.now()}:\n{str(e)}\n\n")
        print(f"Error processing {filename}. Logged the error.")

print("\nAll done! Check your 'output_texts' folder.")