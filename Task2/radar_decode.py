import sqlite3
import numpy as np
import pandas as pd

DB="/mnt/d/EngineeringChallenge/Data/2025-10-08_09-35_sensors_raw_1.db3"

TARGET=1759902480866707757

conn=sqlite3.connect(DB)

topic_id=conn.execute(
"""
SELECT id
FROM topics
WHERE name='/sensor/radar_front/points'
"""
).fetchone()[0]

raw=conn.execute(
"""
SELECT data
FROM messages
WHERE topic_id=?
AND timestamp=?
LIMIT 1
""",
(
topic_id,
TARGET
)
).fetchone()[0]

conn.close()

# ==================
# PARSE
# ==================

OFFSET=220
POINT_STEP=32

payload=raw[OFFSET:]

usable=(

len(payload)//POINT_STEP

)*POINT_STEP

payload=payload[
:usable
]

dtype=np.dtype([

("x","<f4"),

("y","<f4"),

("z","<f4"),

("velocity","<f4"),

("snr","<f4"),

("rcs","<f4"),

("confidence","<f4"),

("vint","<f4")

])

pts=np.frombuffer(
payload,
dtype=dtype
)

df=pd.DataFrame({

"x":pts["x"],

"y":pts["y"],

"velocity":pts["velocity"],

"confidence":pts["confidence"]

})

# ==================
# FILTER
# ==================

df=df[

(df.x>5)

&

(df.x<15)

]

df=df[

abs(
df.y
)<3

]

df=df[

df.confidence>10

]

df=df[

abs(
df.velocity
)>5

]

df["distance"]=np.sqrt(

df.x**2+

df.y**2

)

df=df.sort_values(
"distance"
)

target=df.head(1)

# ==================
# OUTPUT
# ==================

print()

print(
"TARGET"
)

print()

print(
target
)

print()

if len(target):

    print(
        "distance:",
        round(
            float(
                target.distance.iloc[0]
            ),
            2
        )
    )

    print(
        "velocity:",
        round(
            float(
                target.velocity.iloc[0]
            ),
            2
        )
    )

target.to_csv(
"radar_target.csv",
index=False
)

print()

print(
"saved radar_target.csv"
)
