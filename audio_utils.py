from pydub import AudioSegment
from pydub.silence import split_on_silence
from moviepy.editor import VideoFileClip

def split_audio_silence(path, min_silence_len = 800, silence_thresh=-45):
    chunks = []
    sound_file = AudioSegment.from_file(path)
    audio_chunks = split_on_silence(sound_file, min_silence_len, silence_thresh)
    
    for index, chunk in enumerate(audio_chunks):
        out_file = "silence_chunks/chunk{0}.wav".format(index)
        chunk.export(out_file, format="wav")
        chunks += [out_file]

    return chunks

def split_audio_fixed(audio_file_path, chunk_length=30):
    audio = AudioSegment.from_file(audio_file_path)
    index = 0

    chunk_length_ms = chunk_length * 1000
    chunk_start = 0
    chunk_end = chunk_length_ms
    chunks = []

    while chunk_start < len(audio):
        out_file = "fixed_chunks/chunk{0}.wav".format(index)
        chunk = audio[chunk_start:chunk_end]
        print("exporting", out_file)
        chunk.export(out_file, format="wav")
        chunks += [out_file]

        chunk_start = chunk_end
        chunk_end += chunk_length_ms
        index += 1

    return chunks

def get_video_duration(mp4_file):
    video_clip = VideoFileClip(mp4_file)
    duration = video_clip.duration
    video_clip.close()
    return duration