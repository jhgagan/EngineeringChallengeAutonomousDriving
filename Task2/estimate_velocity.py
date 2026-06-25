import pandas as pd

df=pd.read_csv(
"task2_measurements.csv"
)

t=df["camera"].values

d=df["distance_m"].values

v=[]

for i in range(
len(d)
):

    if i==0:

        v.append(
            0
        )

    else:

        dt=(

        t[i]

        -

        t[i-1]

        )/1e9

        vel=(

        d[i]

        -

        d[i-1]

        )/dt

        v.append(

            round(
                vel,
                2
            )

        )

df[
"velocity_m_s"
]=v

print()

print(df)

print()

df.to_csv(

"task2_state.csv",

index=False

)

print(
"saved task2_state.csv"
)
