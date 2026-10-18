import cv2

import FaceDetector

if __name__ == "__main__":
    detector = FaceDetector.FaceDetector()
    detector.start()  # background thread begins capturing
    try:
        # Main thread: display only (cv2.imshow must stay in the main thread)
        while True:
            frame, ear, features = detector.get_latest()
            if frame is not None:
                if features is not None:
                    print(f"pitch_dev={features.pitch_from_baseline:6.1f}  closed={features.closed_fraction:.2f}  "
                          f"band={features.band_power_ratio:.2f}  freq={features.dominant_freq_hz:.1f}")
                text: str = f"EAR: {ear:.2f}" if ear is not None else "NO FACE"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("face test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
                break
    finally:
        detector.stop()  # stop the thread and release the camera
        cv2.destroyAllWindows()