import sqlite3
import numpy as np
import matplotlib.pyplot as plt

DB="Data/2025-10-08_09-35_sensors_raw_0.db3"

conn=sqlite3.connect(DB)

topic="/sensor/lidar_front/points"

topic_id=conn.execute(
"SELECT id FROM topics WHERE name=?",
(topic,)
).fetchone()[0]

raw=conn.execute(
"""
SELECT data
FROM messages
WHERE topic_id=?
LIMIT 1
"""
,
(topic_id,)
).fetchone()[0]

conn.close()

POINT_STEP=32
OFFSET=272

payload=raw[OFFSET:]

usable=(
len(payload)//POINT_STEP
)*POINT_STEP

payload=payload[:usable]

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
z=pts["z"]

np.save(
"lidar_points.npy",
np.column_stack(
[x,y,z]
)
)

mask=(

x>0
)&(

x<50
)&(

abs(y)<25
)&(

z>-2
)&(

z<3

)

plt.figure(
figsize=(10,10)
)

plt.scatter(
x[mask],
y[mask],
s=0.1
)

plt.axis(
"equal"
)

plt.savefig(
"lidar.png"
)

print("done")