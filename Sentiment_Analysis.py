import cv2
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    try:
        # Analyze emotion
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )

        if isinstance(result, list):
            dominant_emotion = result[0].get("dominant_emotion", "Unknown")
        else:
            dominant_emotion = result.get("dominant_emotion", "Unknown")

        cv2.putText(
            frame,
            f"Emotion: {dominant_emotion}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    except Exception as e:
        cv2.putText(
            frame,
            "No emotion detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        print("Error:", e)

    # Display frame
    cv2.imshow("Real-Time Emotion Detection System", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()