import cv2
from moviepy.editor import VideoFileClip
from nltk.tokenize import sent_tokenize
import numpy as np
import whisper
import torch
import text_utils as txt
from audio_utils import split_audio_fixed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = whisper.load_model("base")

# Chargement de la vidéo
video_path = "media_video/audio2.mp4"
clip = VideoFileClip(video_path)
fps = clip.fps
duree = clip.duration
nb_frames = duree * fps

# Chargement de l'audio depuis la vidéo
audio = clip.audio

# Enregistrement de l'audio extrait
audio_output_path = "output_audio/audio1.wav"
audio.write_audiofile(audio_output_path)

# Divide the audio into chunks
segments = split_audio_fixed(audio_output_path)
words = []

for idx, audio_segment in enumerate(segments, start=1):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_segment)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Ensure that the mel spectrogram is in float32 format
    mel = mel.to(torch.float32)

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)

    text = txt.convert_to_lowercase(result.text)
    sentences = txt.split_into_sentences(text)
    for sentence in sentences:
        words += txt.split_into_words(sentence)

# A ce stade, words est la liste des mots prononcés dans la vidéo originale. La ponctuation compte comme des mots.

# Pour la création de l'output

resolution = (1080, 1080)
codec = cv2.VideoWriter_fourcc(*'avc1')
video_writer = cv2.VideoWriter('output_video/latest_animation_mouth.mp4', codec, fps, resolution)

# Import des images
frames = []
for i in range(1,9):
    filename = 'frames/lipsync/{}.png'.format(i)
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_array = np.array(gray_image)
    frames += [gray_array]

list_of_sounds = []
for word in words:
    list_of_sounds += [txt.extract_sounds(word)]

remains = nb_frames // len(list_of_sounds) # Nombre de frames où une seule image reste à l'écran
last = nb_frames % len(list_of_sounds)

for sounds in list_of_sounds:
    frame = txt.insert_frame(sounds)
    for i in range(remains):
        video_writer.write(frame)
for i in range(last):
    video_writer.write('frames/lipsync/neutre.png')

#releasing the VideoCapture object
video_writer.release()
cv2.destroyAllWindows()