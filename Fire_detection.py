from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")  # Path to the best-trained model

def detect_fire_image(image_path):
    results = model(image_path, save=True)  # Perform detection and save output
    print("Fire detection completed. Check the saved results.")

def detect_fire_video(video_path, output_path="output.mp4"):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for r in results:
            frame = r.plot()  # Draw bounding boxes on frame
        out.write(frame)
        cv2.imshow("Fire Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Fire detection video saved at: {output_path}")

