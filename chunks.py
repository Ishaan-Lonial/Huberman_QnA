import pandas as pd

# Function to split a transcript into chunks of a specified size
def split_transcript(transcript, chunk_size=500):
    return [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]

# Load the original CSV file
input_csv = "video_transcripts.csv"
df = pd.read_csv(input_csv)

# List to hold video titles and transcript chunks
data = []

# Iterate over the rows in the DataFrame
for index, row in df.iterrows():
    video_title = row['videotitle']
    transcript = row['transcripts']
    
    # Split transcript into chunks
    transcript_chunks = split_transcript(transcript)
    
    # Add video title and each transcript chunk to data list
    for chunk in transcript_chunks:
        data.append({"videotitle": video_title, "transcripts": chunk})

# Create a DataFrame from the data list
df_chunks = pd.DataFrame(data)

# Save DataFrame to a new CSV file
output_csv = "video_transcripts_chunks.csv"
df_chunks.to_csv(output_csv, index=False)

print(f"Transcripts with chunks saved to {output_csv}")
