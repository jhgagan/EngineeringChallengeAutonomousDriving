from ultralytics import YOLO
import cv2
import os

model=YOLO(
"yolov8m.pt"
)

BASE=r"C:\Data\Raw1\Raw1_all"

cams=[

"camera_fl",

"camera_cl",

"camera_cr",

"camera_fr"

]

OUT=r"C:\Data\task3_1_multi_tracking"

os.makedirs(
OUT,
exist_ok=True
)

files=[]

for cam in cams:

    folder=os.path.join(
        BASE,
        cam
    )

    files.append(

        sorted([

            f

            for f in os.listdir(
                folder
            )

            if f.endswith(
                ".jpg"
            )

        ])

    )

N=min(
len(x)
for x in files
)

trail=[]

last_camera=None

for i in range(N):

    best=None
    best_area=0

    for k,cam in enumerate(cams):

        img=files[k][i]

        path=os.path.join(
            BASE,
            cam,
            img
        )

        frame=cv2.imread(
            path
        )

        H,W=frame.shape[:2]

        r=model(

            frame,

            classes=[2],

            conf=0.22,

            verbose=False

        )[0]

        for box in r.boxes:

            x1,y1,x2,y2=map(
                int,
                box.xyxy[0]
            )

            if y2>H-120:
                continue

            if x1>W-350:
                continue

            area=(x2-x1)*(y2-y1)

            if area<1200:
                continue

            if area>best_area:

                best_area=area

                best=(

                    frame.copy(),

                    x1,

                    y1,

                    x2,

                    y2,

                    cam

                )

    if best:

        frame,x1,y1,x2,y2,cam=best

        # RESET TRAIL ON CAMERA SWITCH

        if cam!=last_camera:

            trail=[]

            last_camera=cam

        cx=(x1+x2)//2
        cy=(y1+y2)//2

        trail.append(
            (
                cx,
                cy
            )
        )

        if len(trail)>80:

            trail.pop(
                0
            )

        cv2.rectangle(

            frame,

            (x1,y1),

            (x2,y2),

            (0,255,0),

            4

        )

        cv2.putText(

            frame,

            "CAR ID 1",

            (x1,y1-10),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0,255,0),

            3

        )

        for t in range(
            1,
            len(trail)
        ):

            cv2.line(

                frame,

                trail[t-1],

                trail[t],

                (0,0,255),

                3

            )

        cv2.putText(

            frame,

            cam,

            (40,70),

            cv2.FONT_HERSHEY_SIMPLEX,

            2,

            (255,0,0),

            4

        )

        cv2.imwrite(

            os.path.join(

                OUT,

                f"{i:04d}.jpg"

            ),

            frame

        )

    if i%100==0:

        print(
            i,
            "/",
            N
        )

print()

print(
"DONE"
)