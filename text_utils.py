import pyphen
import nltk
import cv2
import numpy as np
nltk.download('punkt')
import re

from nltk.tokenize import sent_tokenize

def split_into_sentences(paragraph):
    # Use NLTK's sentence tokenizer to split the paragraph into sentences.
    sentences = sent_tokenize(paragraph)

    return sentences

def split_into_words(sentence):
    # Use regular expression to split the sentence into words, including punctuation marks.
    words = re.findall(r'\w+|[^\w\s]', sentence)
    return words

dic = pyphen.Pyphen(lang='fr_FR')

def count_syllables(string):
    res = 0
    for syllable in dic.iterate(string):
        res += 1
    return res

def extract_sounds(word):
    voyelles = 'aàeéèêoôuùiïy'
    n = len(word)
    res = ''

    if n == 1: #traite les signes de ponctuation
        return word

    if word[0] == 'c':
        if word[1] == 'h':
            res += '€'
        elif word[1] in 'aàâoöuù':
            res += 'k'
        elif word[1] in 'eéèëiïy':
            res += 's'
    elif word[0] == 'y':
        if word[1] in voyelles:
            res += 'µ'
        else:
            res += 'y'
    elif word[0] == 'g':
        if   word[1] == 'n':
            res += 'n'
        else:
            res += 'g'
    elif word[0] == 's':
        if   word[1] == 'h':
            res += '$'
        else:
            res += 's'
    else:
        res += word[0]

    i = 2
    while i < n:
        letter = word[i]
        if letter == 'a':
            if i + 1 < n and word[i + 1] == 'i':
                res += 'è'
                i += 2
            elif i + 1 < n and word[i + 1] == 'u':
                res += 'o'
                i += 2
            else:
                res += 'a'
                i += 1

        elif letter == 'e':
            if i + 2 < n and word[i + 1] == 'a' and word[i + 2] == 'u':
                res += 'o'
                i += 3
            elif i + 1 < n and word[i + 1] == 'a':
                res += 'a'
                i += 2
            elif i + 1 < n and word[i + 1] == 'e':
                res += 'e'
                i += 2
            elif i + 1 < n and word[i + 1] == 'i':
                res += 'è'
                i += 2
            elif i + 1 < n and word[i + 1] == 'y':
                res += 'è'
                i += 2
            else:
                res += 'e'
                i += 1

        elif letter == 'i':
            if i + 1 < n and word[i + 1] == 'e':
                res += 'e'
                i += 2
            else:
                res  += 'i'
                i += 1

        elif letter == 'o':
            if i + 1 < n and word[i + 1] in 'iu':
                res += 'o'
                i += 2
            else:
                res += 'o'
                i += 1
        else:
            i += 1

    if n > 0 and word[n - 1] in voyelles:
        if n > 1 and word[n - 2] not in voyelles:
            if word[n - 2] == 'h':
                res = res[:-1] + '€'
            elif word[n - 2] == 'c':
                res = res[:-1] + 's'
            elif word[n - 2] == 's':
                res = res[:-1] + 'z'
            elif word[n - 2] == 'g':
                res = res[:-1] + 'j'
            elif word[n - 2] == 'l' and word[n - 3] == 'l':
                res = res[:-1] + 'µ'
            else:
                res = res[:-1] + word[n-2]
        else: #c'est une voyelle avant le e
            res = res[:-1]
    return res

def nb_unique_frames(sentence):
    n = 0
    for word in sentence:
        n += len(word)
    return n

def convert_to_lowercase(sentence):
    return sentence.lower()

# Import des images
frames = []
for i in range(1,9):
    filename = 'frames/lipsync/{}.png'.format(i)
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_array = np.array(gray_image)
    frames += [gray_array]

def insert_frame(sounds):
    for letter in sounds:
        if letter in 'aéyi':
            return frames[0]
        elif letter in 'bmp':
            return frames[1]
        elif letter in 'djnsçxz$€':
            return frames[2]
        elif letter in 'fv':
            return frames[3]
        elif letter in 'ltknqµ':
            return frames[4]
        elif letter in 'ouwe':
            return frames[5]
        elif letter in 'règ':
            return frames[6]
        else:
            return frames[7]
        
