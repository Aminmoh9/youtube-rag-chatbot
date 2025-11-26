"""
Video/audio downloader using yt-dlp.
Downloads audio files from lesson videos and updates CSV with file paths.
"""
import yt_dlp
import pandas as pd
from pathlib import Path


def download_videos_from_csv(csv_path='data/video_links.csv', output_dir='data/videos'):
    """
    Download audio from lesson videos listed in CSV and update paths.
    
    Args:
        csv_path (str): Path to CSV file with video links
        output_dir (str): Directory to save audio files
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Filter only lesson videos
    lesson_videos = df[df['topic_tag'] != 'Non-Lesson'].copy()
    
    if len(lesson_videos) == 0:
        print("No lesson videos found to download!")
        return
    
    print(f"\nFiltered: {len(lesson_videos)} lesson videos (excluded {len(df) - len(lesson_videos)} non-lesson videos)")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download audio only - keep original format (no conversion needed for Whisper)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'quiet': False,
        'no_warnings': False,
        'sleep_interval': 5,  # Wait 5 seconds between downloads to avoid bot detection
        'max_sleep_interval': 15,
    }
    
    print(f"\n{'=' * 70}")
    print(f"DOWNLOADING {len(lesson_videos)} LESSON VIDEOS")
    print(f"{'=' * 70}\n")
    
    successful = 0
    failed = 0
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for idx, row in lesson_videos.iterrows():
            video_id = row['video_id']
            
            try:
                print(f"\n[{successful + failed + 1}/{len(lesson_videos)}] Downloading: {row['title'][:60]}...")
                print(f"Video ID: {video_id}")
                
                # Download
                info = ydl.extract_info(row['url'], download=True)
                
                # Find the downloaded file
                audio_files = list(Path(output_dir).glob(f"{video_id}.*"))
                
                if audio_files:
                    audio_path = str(audio_files[0])
                    # Update the dataframe with the file path
                    df.loc[df['video_id'] == video_id, 'local_audio_path'] = audio_path
                    print(f"✓ Successfully downloaded to: {audio_path}")
                    successful += 1
                else:
                    print(f"✗ File not found after download")
                    failed += 1
                    
            except Exception as e:
                print(f"✗ Error downloading {video_id}: {str(e)}")
                failed += 1
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'=' * 70}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {len(lesson_videos)}")
    print(f"\nFiles saved to: {output_dir}/")
    print(f"CSV updated: {csv_path}")


def download_single_video(video_url, output_dir='data/videos'):
    """
    Download a single video's audio.
    
    Args:
        video_url (str): YouTube video URL
        output_dir (str): Directory to save audio file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download audio only - keep original format (no conversion needed for Whisper)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Downloading: {video_url}")
        ydl.download([video_url])
        print(f"✓ Download complete")


if __name__ == "__main__":
    print("=" * 70)
    print("YOUTUBE VIDEO DOWNLOADER")
    print("=" * 70)
    
    # Download videos from CSV and update paths
    download_videos_from_csv()