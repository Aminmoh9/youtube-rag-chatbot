"""
Speech-to-text transcription using OpenAI Whisper.
Transcribes audio files and updates CSV with transcript paths.
"""
import whisper
import pandas as pd
from pathlib import Path


def transcribe_audio_file(audio_path, model_size='base'):
    """
    Transcribe a single audio file using Whisper.
    
    Args:
        audio_path (str): Path to audio file
        model_size (str): Whisper model size (tiny, base, small, medium, large)
    
    Returns:
        str: Transcribed text
    """
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing: {audio_path}")
    result = model.transcribe(str(audio_path))
    
    return result['text']


def transcribe_all_videos(csv_path='data/video_links.csv',
                          transcripts_dir='data/transcripts',
                          model_size='base'):
    """
    Transcribe all downloaded lesson videos and update CSV.
    
    Args:
        csv_path (str): Path to video metadata CSV
        transcripts_dir (str): Directory to save transcripts
        model_size (str): Whisper model size (tiny/base/small/medium/large)
    """
    # Create transcripts directory
    Path(transcripts_dir).mkdir(parents=True, exist_ok=True)
    
    # Load video metadata
    df = pd.read_csv(csv_path)
    
    # Ensure local_transcript_path column exists
    if 'local_transcript_path' not in df.columns:
        df['local_transcript_path'] = None
    
    # Filter lesson videos that have audio downloaded
    to_transcribe = df[
        (df['topic_tag'] != 'Non-Lesson') & 
        (df['local_audio_path'].notna())
    ].copy()
    
    if len(to_transcribe) == 0:
        print("No audio files found to transcribe!")
        print("Make sure to download videos first using video_downloader.py")
        return
    
    # Load Whisper model once
    print(f"\n{'=' * 70}")
    print(f"LOADING WHISPER MODEL: {model_size}")
    print(f"{'=' * 70}\n")
    model = whisper.load_model(model_size)
    
    print(f"\n{'=' * 70}")
    print(f"TRANSCRIBING {len(to_transcribe)} VIDEOS")
    print(f"{'=' * 70}\n")
    
    successful = 0
    skipped = 0
    failed = 0
    
    # Process each video
    for idx, row in to_transcribe.iterrows():
        video_id = row['video_id']
        audio_path = row['local_audio_path']
        transcript_path = Path(transcripts_dir) / f"{video_id}.txt"
        
        # Skip if already transcribed
        if transcript_path.exists():
            print(f"⊘ Skipping {video_id} (already transcribed)")
            # Update CSV even if skipped
            df.loc[df['video_id'] == video_id, 'local_transcript_path'] = str(transcript_path)
            skipped += 1
            continue
        
        # Check if audio file exists
        if not Path(audio_path).exists():
            print(f"✗ Audio file not found: {audio_path}")
            failed += 1
            continue
        
        try:
            # Transcribe
            print(f"\n[{successful + skipped + failed + 1}/{len(to_transcribe)}] Transcribing: {row['title'][:60]}...")
            # Convert to absolute path to avoid path issues
            abs_audio_path = str(Path(audio_path).absolute())
            result = model.transcribe(abs_audio_path)
            transcript = result['text']
            
            # Save transcript
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            # Update CSV with transcript path
            df.loc[df['video_id'] == video_id, 'local_transcript_path'] = str(transcript_path)
            
            print(f"✓ Saved to: {transcript_path}")
            successful += 1
            
        except Exception as e:
            print(f"✗ Error transcribing {video_id}: {str(e)}")
            failed += 1
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'=' * 70}")
    print(f"TRANSCRIPTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"✓ Successful: {successful}")
    print(f"⊘ Skipped (already done): {skipped}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {len(to_transcribe)}")
    print(f"\nTranscripts saved to: {transcripts_dir}/")
    print(f"CSV updated: {csv_path}")


if __name__ == "__main__":
    # Transcribe all videos with base model
    # Options: tiny, base, small, medium, large
    # Note: larger models are more accurate but slower
    transcribe_all_videos(model_size='base')
