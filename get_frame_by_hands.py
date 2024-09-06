import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import cv2
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
from argparse import ArgumentParser
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class FrameExtractor:
    def __init__(self, video_path, output_dir, gaussian_sigma=5, prominence=0.8, csv_file='selected_valleys.csv'):
        self._folder_init(video_path, output_dir)
        self._mediapipe_init()
        self._visualization_init()

        self.cap = cv2.VideoCapture(self.video_path)
        self.num_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        '''
        Hardcoded possible handedness is a bad practice because we do not know how the Google API may change in the future.
        But because how hard it is to actually query category_module.Category and get all possible handedness, we will just hardcode it.
        '''
        self.all_landmark_pos = {"Right": np.zeros((self.num_frame, 2)), "Left": np.zeros((self.num_frame, 2))}

        # signal processing parameters
        self.gaussian_sigma = gaussian_sigma
        self.prominence = prominence

        self.csv_file = csv_file
 
    def extract_frames(self):
        '''
        Extract the frames from the video.
        '''
        self.analyze_video()

        self.all_possible_handedness = set(self.all_landmark_pos.keys()) # all possible handedness detected in the video

        self.all_speeds = {}

        for handedness in self.all_possible_handedness:
            print(f"Calculating the speed curve of the {handedness} hand.")
            self.all_speeds[handedness] = self.get_speed(self.all_landmark_pos[handedness])
            print(f"Plotting the speed curve of the {handedness} hand.")
            self.plot_speed(self.all_speeds[handedness], handedness)
            print(f"Making the video of the {handedness} hand.")
            self.make_video(handedness)
        
        print("Deciding which hand to focus on.")
        self.handedness = self.decide_handedness()
        print(f"The decided handedness is {self.handedness}.")

        print(f"Processing the speed curve of {self.handedness} hand.")
        smoothed_curve = self.process_speed_curve()

        print(f"Getting the peaks and valleys of the speed curve of {self.handedness} hand.")
        peaks, valleys = self.get_peaks_valleys(smoothed_curve)

        # Filter valleys based on the index difference
        selected_valleys = []
        for i in range(len(valleys)):
            if i == 0 or (valleys[i] - selected_valleys[-1]) >= 15:
                selected_valleys.append(valleys[i])

        print(f"Plotting and making videos with smoothed {self.handedness} hand speed curve.")
        self.plot_speed(speeds=smoothed_curve, 
                        handedness=f'Smoothed {self.handedness}',
                        selected_frame=selected_valleys)
        
        self.make_video(f'Smoothed {self.handedness}')

        first_frame = self.get_frame(0)
        cv2.imwrite(f'{str(self.selected_folder)}/{0}.jpg', first_frame)
        for valley in selected_valleys:
            frame = self.get_frame(valley)
            cv2.imwrite(f'{str(self.selected_folder)}/{valley}.jpg', frame)
        print(f"The selected valley frames are: {selected_valleys}")

        for valley in valleys:
            frame = self.get_frame(valley)
            cv2.imwrite(f'{str(self.all_valleys_folder)}/{valley}.jpg', frame)
        print(f"All valley frames are: {valleys}")

        # Save selected valleys to a unified CSV
        self.save_selected_valleys_to_csv(selected_valleys)

        self.cap.release()
        print("All done!")

    def save_selected_valleys_to_csv(self, selected_valleys):
        '''
        Save the selected valleys to a unified CSV file with the video name as the identifier.
        This ensures all videos append to the same CSV file.
        '''
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.video_name, selected_valleys])
        print(f"Selected valleys for {self.video_name} saved to {self.csv_file}")

    def analyze_video(self):
        '''
        Analyze the video frame by frame.
        '''

        frame_counter = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print(f"End of video at frame {frame_counter}")
                break

            # convert the BGR image to RGB because of OpenCV uses BGR while MediaPipe uses RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with self.HandLandmarker.create_from_options(self.mp_hand_options) as landmarker:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) # convert the image to MediaPipe format
                mp_timestamp = int(round(time.time()*1000)) # get the current timestamp in milliseconds

                results = landmarker.detect_for_video(mp_image, mp_timestamp) # detect the hands in the frame

                # Draw the landmarks on the image and get the average x and y coordinates of the hand landmarks in this frame.
                annotated_image, avg_landmark_x, avg_landmark_y, all_handedness = self.process_frame_results(rgb_image=frame, 
                                                                                                       detection_result=results)

                if len(all_handedness) == 0:
                    # no hands detected in this frame
                    self.all_landmark_pos['Right'][frame_counter] = None
                    self.all_landmark_pos['Left'][frame_counter] = None
                else:
                    for handedness in all_handedness:
                        # collet both left and right hand landmarks
                        self.all_landmark_pos[handedness][frame_counter] = (avg_landmark_x[handedness], avg_landmark_y[handedness])

            # save the image to a folder, not as a video
            cv2.imwrite(f'{str(self.hand_images_folder)}/{frame_counter}.jpg', annotated_image)
            
            
            frame_counter += 1

        '''
        Replace the 0 in the landmark_pos with np.nan
        When one hand is detected, the other hand will have 0 as the x and y coordinates.
        '''
        for handedness in self.all_landmark_pos.keys():
            self.all_landmark_pos[handedness] = np.where(self.all_landmark_pos[handedness] == 0, np.nan, self.all_landmark_pos[handedness])
    
    def process_frame_results(self, rgb_image, detection_result):
        '''
        Annotate the image with the hand landmarks and handedness.
        Also process the results to get the average x and y coordinates of the hand landmarks in this frame.

        Mostly copied from the MediaPipe example. Kinda messy to be honest.

        Input:
        rgb_image: np.array. The image to be annotated.
        detection_result: mp.tasks.vision.HandLandmarkerResult. The detection result.

        Return:
        annotated_image: np.array. The annotated image.
        avg_landmark_x: dict. The average x coordinate of the hand landmarks in this frame. Both left and right hand.
        avg_landmark_y: dict. The average y coordinate of the hand landmarks in this frame. Both left and right hand.
        all_handedness: set. All handedness detected in this frame.
        '''

        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        

        avg_landmark_x = {} # average x coordinate of the hand landmarks in this frame
        avg_landmark_y = {} # average y coordinate of the hand landmarks in this frame
        all_handedness = set() # all handedness detected in this frame
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList() # type: ignore
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks # type: ignore
            ])
            solutions.drawing_utils.draw_landmarks( # type: ignore
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS, # type: ignore
            solutions.drawing_styles.get_default_hand_landmarks_style(), # type: ignore
            solutions.drawing_styles.get_default_hand_connections_style()) # type: ignore

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self.MARGIN

            avg_landmark_x[handedness[0].category_name] = np.average(x_coordinates) * width
            avg_landmark_y[handedness[0].category_name] = np.average(y_coordinates) * height
            all_handedness.add(handedness[0].category_name)

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        self.FONT_SIZE, self.HANDEDNESS_TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image, avg_landmark_x, avg_landmark_y, all_handedness
    
    def decide_handedness(self):
        '''
        Decide which hand is the one we are interested in.

        Return:
        str. The decided handedness. Either "Right" or "Left".
        '''

        if len(self.all_possible_handedness) == 1:
            # only one hand detected in the video
            return list(self.all_possible_handedness)[0]
        if len(self.all_possible_handedness) == 0:
            # no hand detected in the video
            return None
        
        scoreboard = {}
        for handedness in self.all_possible_handedness:
            scoreboard[handedness] = 0
        
        # count which hand has the most non-zero landmarks
        max_nonzero_num = 0
        max_nonzero_handedness = None

        # count which hand has the most speed range
        max_speed_range = 0
        max_speed_range_handedness = None

        # count which hand has the most x and y range
        max_range = 0
        max_range_handedness = None

        for handedness in self.all_possible_handedness:
            # decide which hand has the most non-zero landmarks
            non_zero_count = np.count_nonzero(~np.isnan(self.all_landmark_pos[handedness]))
            if non_zero_count > max_nonzero_num:
                max_nonzero_num = non_zero_count
                max_nonzero_handedness = handedness
            
            # decide which hand has the most speed range
            speed_max = np.nanmax(self.all_speeds[handedness])
            speed_min = np.nanmin(self.all_speeds[handedness])
            speed_range = speed_max - speed_min
            if speed_range > max_speed_range:
                max_speed_range = speed_range
                max_speed_range_handedness = handedness
            
            pos_max = np.nanmax(self.all_landmark_pos[handedness])
            pos_min = np.nanmin(self.all_landmark_pos[handedness])
            pos_range = pos_max - pos_min
            if pos_range > max_range:
                max_range = pos_range
                max_range_handedness = handedness
        
        scoreboard[max_nonzero_handedness] += 1
        scoreboard[max_speed_range_handedness] += 1
        scoreboard[max_range_handedness] += 1

        # return the hand with the most votes
        return max(scoreboard, key=scoreboard.get) # type: ignore
    
    def process_speed_curve(self):
        '''
        Process the speed curve of the hand so we can find peaks and valley more robustly.

        1. Linearly interpolate the nan values in the speed curve.
        2. Use Gaussian filter to smooth the speed curve.

        Return:
        np.array. The smoothed speed curve.
        '''

        # linear interpolation
        y = self.all_speeds[self.handedness]
        nans, x_nans = np.isnan(y), lambda z: z.nonzero()[0]
        y_interpolated = y.copy()
        y_interpolated[nans] = np.interp(x_nans(nans), x_nans(~nans), y[~nans])

        # Gaussian filter smoothing 
        y_smoothed = gaussian_filter(y_interpolated, sigma=self.gaussian_sigma)

        return y_smoothed
    
    def get_peaks_valleys(self, smoothed_curve):
        '''
        Get the peaks and valleys of the speed curve.

        Input: np.array. The smoothed speed curve.

        Return:
        peaks: np.array. The indices of the peaks.
        valleys: np.array. The indices of the valleys.
        '''
        peaks, _ = find_peaks(smoothed_curve)
        valleys, _ = find_peaks(-smoothed_curve)
        x = np.arange(len(smoothed_curve))

        plt.figure()
        plt.plot(x, smoothed_curve, label='Smoothed Data');
        plt.plot(x[peaks], smoothed_curve[peaks], 'rx', label='Peaks');
        plt.plot(x[valleys], smoothed_curve[valleys], 'go', label='Valleys');
        plt.title(f'Smoothed {self.handedness} Hand Speed Peaks and Valleys')
        plt.savefig(f'{str(self.base_folder)}/{self.handedness}_speed_smoothed.jpg')
        plt.close()

        return peaks, valleys

    def get_frame(self, frame_number):
        '''
        Get the frame of the video at the specified frame number.

        Input:
        frame_number: int. The frame number.

        Return:
        np.array. The frame of the video.
        '''
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if not ret:
            print(f"Something went wrong when reading frame {frame_number}.")
            return np.zeros((self.VIDEO_HEIGHT, self.VIDEO_WIDTH, 3))
        return frame
    
    def get_speed(self, coordinates):
        '''
        Get the speed curve of the hand.
        '''
        length = len(coordinates)
        speed = np.zeros(length-1)
        for i in range(length-1):
            if coordinates[i] is not None and coordinates[i+1] is not None:
                speed[i] = np.linalg.norm(coordinates[i] - coordinates[i+1])
            else:
                speed[i] = None
        return speed
    
    def plot_speed(self, speeds, handedness, selected_frame=None):
        '''
        Plot the speed curve of the hand. Overlay the current hand speed of i-th frame on the entire speed curve.

        Input:
        speeds: np.array. The speed curve of the hand.
        handedness: str. The handedness of the hand. Either "Right" or "Left".
        '''
        for i in range(self.num_frame-1):
            '''
            For every frame, we are plotting the entire speed curve again and again for the purpose of making a video at the end.
            Again, can be better if I am more fluent in matplotlib.
            '''
            plt.figure();
            plt.plot(speeds, label=f'{handedness} Hand Speed Distribution');
            if selected_frame is not None:
                plt.scatter(selected_frame, speeds[selected_frame], color='blue', label=f'Selected Frame {selected_frame}');
            plt.scatter(i, speeds[i], color='red', label=f'Current {handedness} Hand');
            plt.legend();
            plt.savefig(f'{str(self.plot_folder)}/{i}_{handedness}.jpg')
            plt.close()

    def make_video(self, handedness):
        '''
        Make a video of the hand images and the speed plot combined.

        Input:
        handedness: str. The handedness of the hand. Either "Right" or "Left".
        '''
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        out = cv2.VideoWriter(f'{str(self.base_folder)}/{handedness}_combined.mp4', fourcc, 30.0, (self.VIDEO_WIDTH*2, self.VIDEO_HEIGHT))
        for i in range(self.num_frame-1):
            frame = cv2.imread(f'{self.hand_images_folder}/{i+1}.jpg')
            frame = cv2.resize(frame, (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
            plot = cv2.imread(f'{str(self.plot_folder)}/{i}_{handedness}.jpg')
            combined = cv2.hconcat([frame, plot])

            out.write(combined)
        out.release()

    def _mediapipe_init(self):
        '''
        Initialize the MediaPipe HandLandmarker.
        '''

        BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a hand landmarker instance with the video mode:
        self.mp_hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2,
            )
    
    def _visualization_init(self):
        '''
        Initialize the visualization parameters.
        '''
        self.MARGIN = 10  # pixels
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
        self.VIDEO_WIDTH = 640
        self.VIDEO_HEIGHT = 480
    
    def _folder_init(self, video_path, output_dir):
        '''
        Initialize all folder-related stuff.
        '''
        output_dir = Path(output_dir)
        self.video_path = video_path
        self.video_name = Path(video_path).stem

        # This folder will hold all output files related to this SPECIFIC video.
        self.base_folder = output_dir / self.video_name

        # This folder will hold all annotated hand images extracted from the video.
        self.hand_images_folder = self.base_folder / 'hand_images'
        os.makedirs(self.hand_images_folder, exist_ok=True)

        # This folder will hold all matplotlib plots related to the hand landmarks.
        self.plot_folder = self.base_folder / 'plots'
        os.makedirs(self.plot_folder, exist_ok=True)

        # This folder will hold the selected frames extracted from the video.
        self.selected_folder = self.base_folder / 'selected_frames'
        os.makedirs(self.selected_folder, exist_ok=True)

        # This folder will hold ALL valled frames extracted from the video.
        self.all_valleys_folder = self.base_folder / 'all_valleys'
        os.makedirs(self.all_valleys_folder, exist_ok=True)


def main(args):
    args_parsed = args.parse_args()
    video_path = args_parsed.video_path
    output_dir = args_parsed.output_dir
    gaussian_sigma = args_parsed.gaussian_sigma
    prominence = args_parsed.prominence

    frame_extractor = FrameExtractor(video_path, output_dir, gaussian_sigma, prominence)
    frame_extractor.extract_frames()

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--video_path', type=str, help='The path to the video file.')
    args.add_argument('--output_dir', type=str, help='The path to the output directory.', default='./output')
    args.add_argument('--gaussian_sigma', type=int, help='The sigma value for the Gaussian filter.', default=5)
    args.add_argument('--prominence', type=float, help='The prominence value for the find_peaks function.', default=0.8)
    main(args)