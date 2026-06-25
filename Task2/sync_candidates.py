import sqlite3

DB="/mnt/d/EngineeringChallenge/Data/2025-10-08_09-35_sensors_raw_1.db3"

CAM_TS=[

1759902480865569779,
1759902483352654898,
1759902485864660743,
1759902488429397486

]

topics={

"lidar":
"/sensor/lidar_front/points",

"radar":
"/sensor/radar_front/points"

}

conn=sqlite3.connect(DB)

data={}

for k,v in topics.items():

    topic_id=conn.execute(
    """
    SELECT id
    FROM topics
    WHERE name=?
    """,
    (v,)
    ).fetchone()[0]

    rows=conn.execute(
    """
    SELECT timestamp
    FROM messages
    WHERE topic_id=?
    """,
    (topic_id,)
    ).fetchall()

    data[k]=[
        x[0]
        for x in rows
    ]

print()

for cam in CAM_TS:

    lidar=min(
        data["lidar"],
        key=lambda x:
        abs(
            x-cam
        )
    )

    radar=min(
        data["radar"],
        key=lambda x:
        abs(
            x-cam
        )
    )

    print()

    print(
        "camera:",
        cam
    )

    print(
        "lidar :",
        lidar
    )

    print(
        "radar :",
        radar
    )

conn.close()
