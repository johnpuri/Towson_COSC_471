
## Dependencies
1. OpenCV
2. Matplotlib
3. Numpy
4. Keras
5. Tensorflow

## Face Detection 

This Python script uses OpenCV to detect faces in a video stream from the default camera and extracts facial keypoints using a pre-trained model. The filtered video is then written to a file and displayed in a window.

## Requirements
We used all open source libraries: 
1. opencv-python: for image and video processing
2. numpy: for numerical operations
3. filter: a custom module for applying filters to video frames
4. model: a custom module defining a pre-trained model for facial keypoint extraction

## Usage

To run the script that reads a video file, execute the start_app(model) function in the script, where model is an instance of the FaceKeypointsCaptureModel class defined in the model module. The script reads the input video file 'efg.MOV', and writes the filtered video to a file called 'something.webm' in the same directory as the script. The script will display a progress message indicating the percentage of frames processed so far. To stop the script, press the ESC key.

To run the script that captures video from the default camera, simply execute the main() function in the script. The script will start capturing video from the default camera and displaying the filtered video in a window. To stop the script, press the ESC key. The resulting video file will be saved as something1.webm in the same directory as the script.
