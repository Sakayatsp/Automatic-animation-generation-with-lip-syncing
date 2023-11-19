import whisper
import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence

def split_audio_silence(path, min_silence_len = 800, silence_thresh=-45):
    chunks = []
    sound_file = AudioSegment.from_file(path)
    audio_chunks = split_on_silence(sound_file, min_silence_len, silence_thresh)
    
    for index, chunk in enumerate(audio_chunks):
        out_file = "silence_chunks/chunk{0}.wav".format(index)
        print("exporting", out_file)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = whisper.load_model("base")

# Divide the audio into chunks
# segments = split_audio_silence("media_audio/test_split_audio.m4a")
segments = split_audio_fixed("media_audio\\audio1.wav")
print(segments)

for idx, audio_segment in enumerate(segments, start=1):
    print(f"Processing segment {idx}/{len(segments)}")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_segment)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Ensure that the mel spectrogram is in float32 format
    mel = mel.to(torch.float32)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)

    # Print the recognized text
    print(f"Result for segment {idx}: {result.text}")
    print()