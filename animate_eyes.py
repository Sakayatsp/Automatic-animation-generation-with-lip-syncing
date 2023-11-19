import cv2
import dlib
import numpy as np

# Pour l'input

BLINK_RATIO_THRESHOLD = 4.3
no_blinks = 0
cap = cv2.VideoCapture("media_video/blink4.mp4")
cv2.namedWindow('Video Playback')

# Pour la création de l'output

resolution = (1080, 1080)
fps = cap.get(cv2.CAP_PROP_FPS)
codec = cv2.VideoWriter_fourcc(*'avc1')
video_writer = cv2.VideoWriter('output_video/latest_animation.mp4', codec, fps, resolution)
cv2.namedWindow('Video Writing')

# Import des images
frames = []
for i in range(9):
    filename = 'frames/frame{}.png'.format(i)
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_array = np.array(gray_image)
    frames += [gray_array]
buffer = 0

# Fonctions

def midpoint(point1 ,point2):
    return np.array(((point1.x + point2.x)/2, (point1.y + point2.y)/2))

def get_EAR(eye_points, facial_landmarks):
    
    # Définitions des points d'intérêts
    corner_left  = np.array((facial_landmarks.part(eye_points[0]).x, 
                    facial_landmarks.part(eye_points[0]).y))
    corner_right = np.array((facial_landmarks.part(eye_points[3]).x, 
                    facial_landmarks.part(eye_points[3]).y))
    
    center_top    = midpoint(facial_landmarks.part(eye_points[1]), 
                             facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), 
                             facial_landmarks.part(eye_points[4]))

    # Calcul des distances
    horizontal_length = np.linalg.norm(corner_left - corner_right)
    vertical_length = np.linalg.norm(center_top - center_bottom)

    EAR = horizontal_length / vertical_length

    return EAR

# Détecteur du visage

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Points d'intérêts pour notre étude
left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

while True:
    # Enregistrement de la frame
    retval, frame_read = cap.read()

    # Cas d'erreur
    if not retval:
        print("Can't receive frame (stream end?). Exiting ...")
        break 

    # Conversion en niveaux de gris
    frame_read = cv2.cvtColor(frame_read, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces,_,_ = detector.run(image = frame_read, upsample_num_times = 0, 
                       adjust_threshold = 0.0)

    for face in faces:
        
        landmarks = predictor(frame_read, face)

        # Dessin des landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame_read, (x, y), 2, (0, 255, 0), -1)

        # Calculs
        left_eye_ratio  = get_EAR(left_eye_landmarks, landmarks)
        right_eye_ratio = get_EAR(right_eye_landmarks, landmarks)
        blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2
        print(blink_ratio)

        if blink_ratio > BLINK_RATIO_THRESHOLD: #IMPORTANT: Les clignements ne peuvent pas être interrompus!
            # Clignement détecté!
            cv2.putText(frame_read,"BLINKING",(10,50), cv2.FONT_HERSHEY_SIMPLEX,
                        2,(255,255,255),2,cv2.LINE_AA)
            if buffer == 8: #s'il n'y a pas d'animation déjà en cours, on met le buffer à 0
                buffer = 0
        else:
            if buffer < 8: #on envoie les frames animées une par une en incrémentant le buffer jusqu'à 8
                buffer += 1
        video_writer.write(frames[buffer])


    cv2.imshow('Video Playback', frame_read)
    cv2.imshow('Video Writing', frames[buffer])

    key = cv2.waitKey(1)
    if key == 27:
        break

#releasing the VideoCapture object
cap.release()
video_writer.release()
cv2.destroyAllWindows()