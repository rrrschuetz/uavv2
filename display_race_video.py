import cv2

def display_video(video_filename):
    # Open the video file
    video_capture = cv2.VideoCapture(video_filename)

    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_filename}")
        return

    while True:
        # Read the next frame from the video
        ret, frame = video_capture.read()

        if not ret:
            print("End of video reached.")
            break

        # Display the frame
        cv2.imshow("Video Frame", frame)

        # Wait for a key event
        key = cv2.waitKey(0)

        if key == ord('q'):
            # Quit if 'q' is pressed
            break
        elif key == ord('n'):
            # Next frame if 'n' is pressed
            continue
        elif key == ord('p'):
            # Previous frame if 'p' is pressed
            # Note: This is tricky with videos, so consider adding a more advanced implementation if needed
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, video_capture.get(cv2.CAP_PROP_POS_FRAMES) - 2))

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_video("output_video.avi")  # Path to your video file
