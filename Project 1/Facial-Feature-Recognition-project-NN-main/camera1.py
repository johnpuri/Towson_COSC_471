import numpy as np
import sys
import cv2
from filter import apply_filter
from model import FaceKeypointsCaptureModel

#initializes a VideoCapture object to capture live video from the default camera (webcam).
# It then retrieves the frame rate of the video and assigns it to the variable fps.
# The code also loads a pre-trained cascade classifier for detecting frontal faces in images
# and creates an empty dictionary named labels.

rgb = cv2.VideoCapture(0)
# length = int(rgb.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(rgb.get(cv2.CAP_PROP_FPS))
print(fps)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels = {}

#The __get_data__() function reads a frame from the rgb video capture object,
# converts it to grayscale, detects faces in the grayscale image using a pre-trained classifier,
# and returns a tuple containing the detected faces as bounding boxes, the original color frame,
# and the grayscale frame. The function is used to process frames in a video stream.

def __get_data__():

    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)

    return faces, fr, gray

#The start_app() function initializes some variables and opens a video file for writing.
# It sets the codec to VP8 and the frame rate to 15 frames per second.
# The function does not use the original frame rate of the video capture object.
# The output video file is written to the path specified by PATH.
def start_app(cnn):
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    width = int(rgb.get(3))  # float
    height = int(rgb.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    PATH = 'something1.webm'
    # output = cv2.VideoWriter(PATH, fourcc, fps, (width, height))
    output = cv2.VideoWriter(PATH, fourcc, 15, (width, height))

#This code loops indefinitely and processes each frame of the video capture object.
    # For each frame, it calls the __get_data__() function to detect faces in the image,
    # applies a trained CNN model to predict facial landmarks,
    # and applies a filter to the frame based on the predicted landmarks.
    # The filtered frame is then written to the output video file and displayed in a window.

    # for sm in range(1,length-1):
    while True:
        ix += 1
        faces, fr, gray_fr = __get_data__()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]

            roi = cv2.resize(fc, (96, 96))
            pred, pred_dict = cnn.predict_points(roi[np.newaxis, :, :, np.newaxis])
            pred, pred_dict = cnn.scale_prediction((x, fc.shape[1] + x), (y, fc.shape[0] + y))

            fr = apply_filter(fr, pred_dict)
        # sys.stdout.write(f"writing...{int((sm/length)*100)+1}%\n")
        # sys.stdout.flush()
        output.write(fr)
        cv2.imshow("", fr)
        if cv2.waitKey(1) == 27:
            break
    # cv2.imshow('Filter', fr)
    rgb.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = FaceKeypointsCaptureModel("face_model.json", "face_model.h5")
    start_app(model)