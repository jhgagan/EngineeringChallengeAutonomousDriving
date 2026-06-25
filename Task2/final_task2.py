import pandas as pd

# ====================

lidar=pd.read_csv(
"task2_kf.csv"
)

radar=pd.read_csv(
"radar_target.csv"
)

# ====================

distance=round(

lidar.kf_distance.iloc[-1],

2

)

velocity=round(

radar.velocity.iloc[0],

2

)

heading=0

confidence=round(

radar.confidence.iloc[0],

1

)

out=pd.DataFrame({

"position_m":[
distance
],

"velocity_m_s":[
velocity
],

"heading_deg":[
heading
],

"confidence":[
confidence
]

})

print()

print(
"FINAL TRACK"
)

print()

print(
out
)

print()

out.to_csv(

"task2_submission.csv",

index=False

)

print(
"saved task2_submission.csv"
)
