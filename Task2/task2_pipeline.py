import sqlite3
import numpy as np
import pandas as pd

# ==========================
# CONFIG
# ==========================

DB="/mnt/d/EngineeringChallenge/Data/2025-10-08_09-35_sensors_raw_1.db3"

CAM_TS=[]

with open(
"car_frames.txt"
) as f:

    for line in f:

        ts=int(
            line.strip().split(",")[1]
        )

        CAM_TS.append(
            ts
        )

CAM_TS=CAM_TS[:20]

# ==========================
# CONNECT
# ==========================

conn=sqlite3.connect(DB)

# ==========================
# TOPICS
# ==========================

lidar_topic=conn.execute(
"""
SELECT id
FROM topics
WHERE name='/sensor/lidar_front/points'
"""
).fetchone()[0]

radar_topic=conn.execute(
"""
SELECT id
FROM topics
WHERE name='/sensor/radar_front/points'
"""
).fetchone()[0]

# ==========================
# TIMESTAMPS
# ==========================

lidar_all=conn.execute(
"""
SELECT timestamp
FROM messages
WHERE topic_id=?
""",
(
lidar_topic,
)
).fetchall()

lidar_all=[
x[0]
for x in lidar_all
]

radar_all=conn.execute(
"""
SELECT timestamp
FROM messages
WHERE topic_id=?
""",
(
radar_topic,
)
).fetchall()

radar_all=[
x[0]
for x in radar_all
]

# ==========================
# MAIN LOOP
# ==========================

rows=[]

for cam in CAM_TS:

    lidar=min(

        lidar_all,

        key=lambda x:
        abs(
            x-cam
        )

    )

    radar=min(

        radar_all,

        key=lambda x:
        abs(
            x-cam
        )

    )

    distance=-1

    raw=conn.execute(
    """
    SELECT data

    FROM messages

    WHERE topic_id=?
    AND timestamp=?

    LIMIT 1
    """,
    (
        lidar_topic,
        lidar
    )
    ).fetchone()

    if raw:

        raw=raw[0]

        payload=raw[244:]

        usable=(

            len(payload)//32

        )*32

        payload=payload[
            :usable
        ]

        dtype=np.dtype([

            ("x","<f4"),

            ("y","<f4"),

            ("z","<f4"),

            ("skip","V20")

        ])

        pts=np.frombuffer(

            payload,

            dtype=dtype

        )

        x=pts["x"]
        y=pts["y"]

        mask=(

            x>0

        )&(

            x<40

        )&(

            abs(y)<8

        )

        x=x[mask]
        y=y[mask]

        if len(x):

            d=np.sqrt(

                x*x+

                y*y

            )

            # remove ego

            d=d[
                d>3
            ]

            if len(d):

                distance=round(

                    float(

                        np.percentile(
                            d,
                            20
                        )

                    ),

                    2

                )

    rows.append([

        cam,

        lidar,

        radar,

        distance

    ])

conn.close()

# ==========================
# DATAFRAME
# ==========================

df=pd.DataFrame(

rows,

columns=[

"camera",

"lidar",

"radar",

"distance_m"

]

)

print()

print(df)

print()

df.to_csv(

"task2_measurements.csv",

index=False

)

print()

print(
"saved task2_measurements.csv"
)
