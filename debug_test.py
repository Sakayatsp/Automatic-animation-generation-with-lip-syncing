import whisper

model = whisper.load_model("tiny")
result = model.transcribe("fixed_chunks\chunk0.wav")
print(result["text"])