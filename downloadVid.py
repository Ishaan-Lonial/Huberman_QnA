import yt_dlp
import os

def download_playlist_yt_dlp(playlist_url, download_path):
    try:
        # Ensure download_path exists
        if not os.path.exists(download_path):
            os.makedirs(download_path)

        # yt-dlp options with custom filename template and restricted quality
        ydl_opts = {
            'outtmpl': os.path.join(download_path, 'vid%(autonumber)s.%(ext)s'),  # Save as vid1.mp4, vid2.mp4, etc.
            'format': 'bestvideo[height<=480]+bestaudio/best',  # Restrict video to 480p and merge with best audio
            'merge_output_format': 'mp4',
        }

        # Use yt-dlp to download the playlist
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([playlist_url])

        print(f"Playlist downloaded to {download_path}")
    except Exception as e:
        print(f"Error downloading playlist: {e}")

# Replace with your playlist URL
playlist_url = 'https://www.youtube.com/watch?v=LYYyQcAJZfk&list=PLPNW_gerXa4O24l7ZHoJbMC2xOO7SpS7K'

# Replace with your download path
download_path = '/Users/lonial/PycharmProjects/256_Project/.venv/Archive'

# Download the playlist
download_playlist_yt_dlp(playlist_url, download_path)
