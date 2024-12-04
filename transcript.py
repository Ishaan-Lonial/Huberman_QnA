import whisper
import os
import pandas as pd

# Load Whisper model
model = whisper.load_model("tiny")
# List of video files
video_files = [
    'vid00001.mp4',
    'vid00002.mp4',
    'vid00003.mp4',
    'vid00004.mp4',
    'vid00005.mp4',
    'vid00006.mp4',
    'vid00007.mp4',
    'vid00008.mp4',
    'vid00009.mp4',
    'vid00010.mp4',
    'vid00011.mp4',
    'vid00012.mp4',
    'vid00013.mp4',
    'vid00014.mp4',
    'vid00015.mp4',
    'vid00016.mp4',
    'vid00017.mp4'
]

# Function to convert video to audio
def convert_to_audio(video_file, output_dir="audio_files"):
    """Convert video to audio using ffmpeg."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    audio_file = os.path.join(output_dir, os.path.basename(video_file).rsplit('.', 1)[0] + ".mp3")
    os.system(f"ffmpeg -i '{video_file}' -q:a 0 -map a '{audio_file}' -y")
    return audio_file

# List to hold video titles and transcripts
data = []

# Iterate over video files and generate transcripts
for video_file in video_files:
    try:
        # Convert video to audio
        audio_file = convert_to_audio(video_file)

        # Transcribe using Whisper
        print(f"Transcribing {audio_file}...")
        result = model.transcribe(audio_file)

        # Extract video title (filename without extension)
        video_title = video_file.rsplit('/', 1)[-1].rsplit('.', 1)[0]

        # Add video title and transcript to data list
        data.append({"videotitle": video_title, "transcripts": result['text']})

    except Exception as e:
        print(f"Error processing {video_file}: {e}")

# Create a DataFrame from the data list
df = pd.DataFrame(data)

# Save DataFrame to CSV
output_csv = "video_transcripts.csv"
df.to_csv(output_csv, index=False)

print(f"Transcripts saved to {output_csv}")