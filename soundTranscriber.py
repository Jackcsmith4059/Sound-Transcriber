import os
import sys
import whisper
import ffmpeg
import tqdm
from datetime import datetime
import argparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import platform
import subprocess
import time
from plyer import notification

# === ARGUMENTS FOR CLI ===
parser = argparse.ArgumentParser(description='Transcribe audio files using Whisper.')
parser.add_argument('--input_folder', type=str, default='input_audio', help='Folder containing audio files.')
parser.add_argument('--output_folder', type=str, default='output_texts', help='Folder to save transcriptions.')
parser.add_argument('--model', type=str, default='large', help='Model to use for transcription.')
parser.add_argument('--delete_after', action='store_true', help='Delete processed audio files after transcription.')
parser.add_argument('--email_on_complete', action='store_true', help='Send an email when transcription is complete.')
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
delete_after_transcription = args.delete_after
email_on_complete = args.email_on_complete

model_folder = "whisper"        # Local whisper folder now
marker_folder = ".transcribed"
os.makedirs(marker_folder, exist_ok=True)
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)

# === CHECK IF MODEL EXISTS LOCALLY ===
model_path = os.path.join(model_folder, "pytorch_model.bin")
if not os.path.exists(model_path):
    print(f"ERROR: Model file 'pytorch_model.bin' not found in {model_folder}/")
    print("Please ensure that the model is downloaded correctly.")
    sys.exit(1)

# === LOAD WHISPER MODEL ===
print(f"Loading Whisper model '{args.model}' from '{model_folder}'...")
model = whisper.load_model(args.model)

# === CREATE INPUT/OUTPUT FOLDERS IF NEEDED ===
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

def send_email_notification():
    sender_email = "youremail@example.com"
    receiver_email = "receiveremail@example.com"
    password = "yourpassword"  # Use a secure method to store the password
    subject = "Whisper Transcription Complete"
    body = "The transcription process has completed successfully."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Email notification sent.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def preprocess_audio(input_path):
    output_path = input_path.rsplit(".", 1)[0] + "_processed.wav"
    ffmpeg.input(input_path).output(output_path, ar=16000).run(quiet=True, overwrite_output=True)  # Change sample rate to 16kHz
    return output_path

# === START TRANSCRIBING FILES ===
audio_extensions = (".mp3", ".wav", ".m4a", ".flac", ".ogg")

audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(audio_extensions)]

if not audio_files:
    print(f"No audio files found in '{input_folder}' folder.")
    sys.exit(0)

print(f"Found {len(audio_files)} audio files. Starting transcription...")

# === CLEANUP MARKERS THAT HAVE NO AUDIO FILE ===
for marker_filename in os.listdir(marker_folder):
    if marker_filename.endswith(".done"):
        base_filename = os.path.splitext(marker_filename)[0]
        corresponding_audio_exists = any(
            os.path.exists(os.path.join(input_folder, base_filename + ext))
            for ext in audio_extensions
        )
        if not corresponding_audio_exists:
            marker_path = os.path.join(marker_folder, marker_filename)
            os.remove(marker_path)
            print(f"Deleted orphaned marker: {marker_filename}")

try:
    for filename in tqdm.tqdm(audio_files, desc="Transcribing"):
        input_path = os.path.join(input_folder, filename)
        base_filename = os.path.splitext(filename)[0]

        # Marker file to know if already processed
        marker_file = os.path.join(marker_folder, base_filename + ".done")

        # Check if already transcribed
        if os.path.exists(marker_file):
            print(f"Skipping {filename} (already transcribed)")
            continue

        try:
            # Preprocess audio to improve transcription quality
            processed_path = preprocess_audio(input_path)

            start_time = time.time()
            result = model.transcribe(processed_path)
            end_time = time.time()
            processing_time = end_time - start_time

            text = result["text"].strip()
            language = result.get("language", "unknown")

            print(f"Transcribed {filename} in {processing_time:.2f} seconds")

            # Determine output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_filename}_{language}_{timestamp}.txt"
            output_path = os.path.join(output_folder, output_filename)

            # Save transcription
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Create a marker file to remember it's done
            with open(marker_file, "w") as m:
                m.write("done")

            print(f"Transcribed: {filename} (Detected language: {language})")

            if delete_after_transcription:
                os.remove(input_path)
                print(f"Deleted {filename} after transcription.")
            # Remove processed intermediate file
            if os.path.exists(processed_path):
                os.remove(processed_path)

        except Exception as e:
            # Handle errors gracefully
            # Create a specific error log per file
            error_log_filename = base_filename + "_error.txt"
            error_log_path = os.path.join(log_folder, error_log_filename)
            with open(error_log_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"Failed to transcribe {filename} at {datetime.now()}:\n{str(e)}\n")
            print(f"Error processing {filename}. Logged error to {error_log_filename}.")

except KeyboardInterrupt:
    print("\nProcess interrupted. Saving progress and exiting...")
    sys.exit(0)

# === OPTIONAL: DELETE LOGS FOLDER IF NO ERROR LOGS EXIST ===
if not any(fname.endswith(".txt") for fname in os.listdir(log_folder)):
    os.rmdir(log_folder)
    print("No errors occurred. Cleaned up empty 'logs' folder.")

# === OPTIONAL: AUTO-OPEN OUTPUT FOLDER ===
output_folder_path = os.path.abspath(output_folder)
try:
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", output_folder_path])
    elif platform.system() == "Windows":  # Windows
        subprocess.run(["start", output_folder_path], shell=True)
    elif platform.system() == "Linux":  # Linux
        subprocess.run(["xdg-open", output_folder_path])
except Exception as e:
    print(f"Could not open output folder automatically: {e}")

# === OPTIONAL: NOTIFICATION AFTER TRANSCRIPTION ===
if email_on_complete:
    send_email_notification()

notification.notify(
    title="Transcription Complete",
    message="The transcription process has finished.",
    timeout=10  # in seconds
)

print("\nAll done! Check your output folder.")