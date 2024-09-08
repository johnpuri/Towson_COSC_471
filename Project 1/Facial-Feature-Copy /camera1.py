import numpy as np
import cv2
from filter import apply_filter
from model import FaceKeypointsCaptureModel


def get_faces(frame, face_cascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    return faces, gray_frame


def start_app(cnn):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)
    # length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    print(fps)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    output_path = 'something.webm'
    output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    skip_frame = 10
    ix = 0

    while True:
        ix += 1
        ret, frame = video_capture.read()
        if ret == False:
            break

        faces, gray_frame = get_faces(frame, face_cascade)

        for (x, y, w, h) in faces:
            fc = gray_frame[y:y + h, x:x + w]
            roi = cv2.resize(fc, (96, 96))
            pred, pred_dict = cnn.predict_points(roi[np.newaxis, :, :, np.newaxis])
            pred, pred_dict = cnn.scale_prediction((x, fc.shape[1] + x), (y, fc.shape[0] + y))
            frame = apply_filter(frame, pred_dict)

        # sys.stdout.write(f"writing...{int((sm/length)*100)+1}%\n")
        # sys.stdout.flush()
        output.write(frame)
        cv2.imshow("", frame)
        if cv2.waitKey(1) == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = FaceKeypointsCaptureModel("face_model.json", "face_model.h5")
    start_app(model)