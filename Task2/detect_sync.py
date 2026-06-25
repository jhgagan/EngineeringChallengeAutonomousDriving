from ultralytics import YOLO
import cv2

model=YOLO("yolov8n.pt")

IMG="sync_camera.jpg"

results=model(

IMG,

conf=0.10,

imgsz=1280,

verbose=True

)

r=results[0]

out=r.plot()

cv2.imwrite(
"sync_detect.jpg",
out
)

print()

for b in r.boxes:

    cls=int(
        b.cls[0]
    )

    print(
        model.names[cls]
    )

print()

print(
"saved"
)
