from keras.models import model_from_json
import numpy as np


class FaceKeypointsCaptureModel(object):
    # Define the names of the facial keypoints in the order they will be predicted
    COLUMNS = ['left_eye_center_x', 'left_eye_center_y',
               'right_eye_center_x', 'right_eye_center_y',
               'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
               'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
               'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
               'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
               'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
               'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
               'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
               'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
               'nose_tip_x', 'nose_tip_y',
               'mouth_left_corner_x', 'mouth_left_corner_y',
               'mouth_right_corner_x', 'mouth_right_corner_y',
               'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
               'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']

    def __init__(self, model_json_file, model_weights_file):
        # Load the model architecture from a JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # Load the model weights from a file
        self.loaded_model.load_weights(model_weights_file)

        # Print a message confirming that the model was loaded successfully
        print("Model loaded from disk")

        # Print a summary of the model architecture
        self.loaded_model.summary()

    def predict_points(self, img):
        # Generate predictions for the input image and apply the modulus operator to ensure that predicted keypoints are within the bounds of the 96x96 input image
        self.preds = self.loaded_model.predict(img) % 96

        # Create a dictionary mapping each facial keypoint name to its predicted (x, y) coordinates
        self.pred_dict = dict([(point, val) for point, val in zip(FaceKeypointsCaptureModel.COLUMNS, self.preds[0])])

        # Return the predicted keypoints as a NumPy array and the dictionary mapping keypoint names to coordinates
        return self.preds, self.pred_dict

    def scale_prediction(self, out_range_x=(-1, 1), out_range_y=(-1, 1)):
        # Define the range of the input image coordinates (0 to 96)
        range_ = [0, 96]

        # Normalize the predicted keypoints to the range [0, 1]
        self.preds = ((self.preds - range_[0]) / (range_[1] - range_[0]))

        # Scale the normalized keypoints to the specified output range using the out_range_x and out_range_y arguments
        self.preds[:, range(0, 30, 2)] = (
                    (self.preds[:, range(0, 30, 2)] * (out_range_x[1] - out_range_x[0])) + out_range_x[0])
        self.preds[:, range(1, 30, 2)] = (
                    (self.preds[:, range(1, 30, 2)] * (out_range_y[1] - out_range_y[0])) + out_range_y[0])

        # Update the dictionary mapping each facial keypoint name to its scaled (x, y) coordinates
        self.pred_dict = dict([(point, val) for point, val in zip(FaceKeypointsCaptureModel.COLUMNS, self.preds[0])])

        # Return the scaled keypoints as a NumPy array and the dictionary mapping keypoint names to coordinates
        return self.preds, self.pred_dict


if __name__ == '__main__':
    # Load the pre-trained model
    model = FaceKeypointsCaptureModel("face_model.json", "face_model.h5")

    # Load an input image and resize it to 96x96 pixels (the input size of the model)
    import matplotlib.pyplot as plt
    import cv2

    img = cv2.cvtColor(cv2.imread('dataset/trial1.jpg'), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img, (96, 96))

    # Convert the image to a 4D tensor with shape (1, 96, 96, 1) to match the input shape of the model
    img1 = img1[np.newaxis, :, :, np.newaxis]

    # Print the shape of the input tensor
    print(img1.shape)

    # Predict the keypoints of the input image using the loaded model
    pts, pts_dict = model.predict_points(img1)

    # Scale the predicted keypoints to the range (0, 200)
    pts1, pred_dict1 = model.scale_prediction((0, 200))

    # Display the input image with the scaled keypoints overlaid
    plt.figure(0)
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', interpolation=None)
    plt.scatter(pts1[range(0, 30, 2)], pts1[range(1, 30, 2)], marker='x')

    # Display the resized input image with the predicted keypoints overlaid
    plt.subplot(1, 2, 2)
    plt.imshow(img1[0, :, :, 0], cmap='gray', interpolation=None)
    plt.scatter(pts[0, range(0, 30, 2)], pts[0, range(1, 30, 2)], marker='x')
    plt.show()