import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval in terms of number of frames
    frame_interval_frames = int(fps * frame_interval)

    # Counter to keep track of frames
    frame_count = 0

    # Read the first frame
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Write the frame to file if it's time to do so
        if frame_count % frame_interval_frames == 0:
            frame_filename = f"{output_folder}/frame_{frame_count // frame_interval_frames}.jpg"
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return video_length

