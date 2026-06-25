import pandas as pd
import numpy as np

df=pd.read_csv(
"task2_state.csv"
)

z=df[
"distance_m"
].values

t=df[
"camera"
].values

x=np.array([

z[0],

0

])

P=np.eye(
2
)

Q=np.array([

[0.01,0],
[0,0.01]

])

R=0.1

out=[]

for i in range(
len(z)
):

    if i==0:

        out.append(
            x.copy()
        )

        continue

    dt=(

    t[i]

    -

    t[i-1]

    )/1e9

    A=np.array([

    [1,dt],

    [0,1]

    ])

    # predict

    x=A@x

    P=A@P@A.T+Q

    # update

    H=np.array([
    [1,0]
    ])

    y=np.array([
    z[i]
    ])-(H@x)

    S=H@P@H.T+R

    K=P@H.T@np.linalg.inv(
        S
    )

    x=x+(
        K@y
    )

    P=(
        np.eye(2)
        -
        K@H
    )@P

    out.append(
        x.copy()
    )

dist=[]
vel=[]

for s in out:

    dist.append(
        round(
            s[0],
            3
        )
    )

    vel.append(
        round(
            s[1],
            3
        )
    )

df[
"kf_distance"
]=dist

df[
"kf_velocity"
]=vel

print()

print(df)

print()

df.to_csv(

"task2_kf.csv",

index=False

)

print(
"saved task2_kf.csv"
)
