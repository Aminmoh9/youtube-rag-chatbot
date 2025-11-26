"""
Transcript chunking with chapter-based and time-based fallback strategies.
Implements overlap for time-based chunking to preserve context.
"""
import json
import ast
import pandas as pd
from pathlib import Path


def chunk_by_chapters(transcript, video_id, chapters, metadata):
    """
    Chunk transcript based on video chapters.
    Each chapter becomes one chunk.
    
    Args:
        transcript (str): Full transcript text
        video_id (str): Video ID
        chapters (list): List of chapter dicts with 'title', 'start', 'end'
        metadata (dict): Video metadata (title, channel, etc.)
    
    Returns:
        list: List of chunk dictionaries
    """
    chunks = []
    
    # Estimate characters per second (rough average for speech)
    chars_per_second = len(transcript) / metadata.get('duration_sec', 1)
    
    for i, chapter in enumerate(chapters):
        start_sec = chapter['start']
        
        # Determine end time:
        # 1. Use chapter's 'end' if provided
        # 2. Use next chapter's start time
        # 3. Fall back to video duration or estimate
        if 'end' in chapter and chapter['end'] is not None:
            end_sec = chapter['end']
        elif i + 1 < len(chapters):
            end_sec = chapters[i + 1]['start']
        else:
            end_sec = metadata.get('duration_sec', start_sec + 300)
        
        # Estimate character positions
        start_char = int(start_sec * chars_per_second)
        end_char = int(end_sec * chars_per_second)
        
        chunk_text = transcript[start_char:end_char].strip()
        
        if chunk_text:
            chunks.append({
                'video_id': video_id,
                'chunk_id': f"{video_id}_ch{i}",
                'chunk_type': 'chapter',
                'title': metadata['title'],
                'chapter_title': chapter['title'],
                'start_time': start_sec,
                'end_time': end_sec,
                'text': chunk_text,
                'topic': metadata.get('topic_tag', 'General'),
                'channel': metadata.get('channel', ''),
                'url': metadata.get('url', '')
            })
    
    return chunks


def chunk_by_time_with_overlap(transcript, video_id, metadata, 
                                chunk_duration_sec=300, 
                                overlap_sec=60):
    """
    Chunk transcript by time intervals with overlap.
    
    Args:
        transcript (str): Full transcript text
        video_id (str): Video ID
        metadata (dict): Video metadata
        chunk_duration_sec (int): Duration of each chunk in seconds (default: 5 minutes)
        overlap_sec (int): Overlap between chunks in seconds (default: 1 minute)
    
    Returns:
        list: List of chunk dictionaries
    
    Example:
        Chunk 1: [0:00 - 5:00]
        Chunk 2: [4:00 - 9:00]  (1 min overlap)
        Chunk 3: [8:00 - 13:00] (1 min overlap)
    """
    chunks = []
    
    duration_sec = metadata.get('duration_sec', 0)
    chars_per_second = len(transcript) / max(duration_sec, 1)
    
    current_start = 0
    chunk_index = 0
    
    while current_start < duration_sec:
        # Calculate time boundaries
        chunk_end = min(current_start + chunk_duration_sec, duration_sec)
        
        # Convert to character positions
        start_char = int(current_start * chars_per_second)
        end_char = int(chunk_end * chars_per_second)
        
        chunk_text = transcript[start_char:end_char].strip()
        
        if chunk_text:
            chunks.append({
                'video_id': video_id,
                'chunk_id': f"{video_id}_t{chunk_index}",
                'chunk_type': 'time',
                'title': metadata['title'],
                'chapter_title': None,
                'start_time': current_start,
                'end_time': chunk_end,
                'text': chunk_text,
                'topic': metadata.get('topic_tag', 'General'),
                'channel': metadata.get('channel', ''),
                'url': metadata.get('url', '')
            })
            
            chunk_index += 1
        
        # Move to next chunk with overlap
        # If we'd go past the end, break
        next_start = current_start + chunk_duration_sec - overlap_sec
        if next_start >= duration_sec:
            break
        current_start = next_start
    
    return chunks


def chunk_transcript(transcript_path, video_metadata, 
                     chunk_duration_sec=300, 
                     overlap_sec=60):
    """
    Main chunking function that chooses strategy based on available data.
    
    Strategy:
    1. If video has chapters → use chapter-based chunking
    2. Otherwise → use time-based chunking with overlap
    
    Args:
        transcript_path (str): Path to transcript file
        video_metadata (dict): Video metadata including chapters
        chunk_duration_sec (int): Chunk duration for time-based (default: 5 min)
        overlap_sec (int): Overlap for time-based (default: 1 min)
    
    Returns:
        list: List of chunk dictionaries
    """
    # Read transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = f.read()
    
    video_id = video_metadata['video_id']
    
    # Parse chapters if available
    chapters = None
    # Handle chapters (could be None, NaN, string, or list)
    chapters_raw = video_metadata.get('chapters')
    if pd.notna(chapters_raw) and chapters_raw and chapters_raw != '[]':
        try:
            # Try JSON first (double quotes)
            chapters = json.loads(chapters_raw) if isinstance(chapters_raw, str) else chapters_raw
        except (json.JSONDecodeError, ValueError):
            try:
                # Fall back to ast.literal_eval (handles single quotes)
                chapters = ast.literal_eval(chapters_raw) if isinstance(chapters_raw, str) else chapters_raw
            except:
                chapters = None
    else:
        chapters = None
    
    # Choose chunking strategy
    if chapters and len(chapters) > 1:
        print(f"  → CHAPTER-based: {len(chapters)} chapters")
        return chunk_by_chapters(transcript, video_id, chapters, video_metadata)
    else:
        print(f"  → TIME-based with {overlap_sec}s overlap")
        return chunk_by_time_with_overlap(transcript, video_id, video_metadata, 
                                         chunk_duration_sec, overlap_sec)


def chunk_all_transcripts(transcripts_dir='data/transcripts',
                          csv_path='data/video_links.csv',
                          output_path='data/chunks.json',
                          chunk_duration_sec=300,
                          overlap_sec=60):
    """
    Process all transcripts and create chunks.
    
    Args:
        transcripts_dir (str): Directory with transcript files
        csv_path (str): Path to video metadata CSV
        output_path (str): Where to save chunks JSON
        chunk_duration_sec (int): Chunk duration for time-based (default: 5 min)
        overlap_sec (int): Overlap for time-based (default: 1 min)
    
    Returns:
        list: All chunks
    """
    df = pd.read_csv(csv_path)
    
    # Filter only lesson videos with transcripts
    df = df[
        (df['topic_tag'] != 'Non-Lesson') &
        (df['local_transcript_path'].notna())
    ]
    
    if len(df) == 0:
        print("No transcripts found to chunk!")
        print("Make sure to transcribe videos first using speech_to_text.py")
        return []
    
    all_chunks = []
    
    print("=" * 70)
    print("CHUNKING TRANSCRIPTS")
    print("=" * 70)
    print(f"Strategy: Chapter-based (when available) or Time-based with {overlap_sec}s overlap")
    print(f"Time-based chunk size: {chunk_duration_sec}s ({chunk_duration_sec//60} minutes)\n")
    
    for idx, row in df.iterrows():
        video_id = row['video_id']
        transcript_path = Path(row['local_transcript_path'])
        
        if not transcript_path.exists():
            print(f"⊘ Skipping {video_id} (transcript not found)")
            continue
        
        # Prepare metadata
        metadata = {
            'video_id': video_id,
            'title': row['title'],
            'channel': row['channel'],
            'url': row['url'],
            'topic_tag': row['topic_tag'],
            'duration_sec': row.get('duration_sec', 0),
            'chapters': row.get('chapters')
        }
        
        # Chunk the transcript
        print(f"\n[{idx+1}/{len(df)}] {row['topic_tag']}: {row['title'][:50]}...")
        chunks = chunk_transcript(transcript_path, metadata, 
                                 chunk_duration_sec, overlap_sec)
        all_chunks.extend(chunks)
        
        print(f"  → Created {len(chunks)} chunks")
    
    # Save all chunks
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    # Show statistics
    chapter_chunks = [c for c in all_chunks if c['chunk_type'] == 'chapter']
    time_chunks = [c for c in all_chunks if c['chunk_type'] == 'time']
    
    print(f"\n{'=' * 70}")
    print(f"CHUNKING SUMMARY")
    print(f"{'=' * 70}")
    print(f"Chapter-based chunks: {len(chapter_chunks)}")
    print(f"Time-based chunks: {len(time_chunks)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"\n✓ Saved to: {output_path}")
    
    return all_chunks


if __name__ == "__main__":
    # Create chunks with 5-minute duration and 1-minute overlap
    chunks = chunk_all_transcripts(
        chunk_duration_sec=300,  # 5 minutes
        overlap_sec=60           # 1 minute overlap
    )
