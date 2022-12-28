import pickle as pkl
import numpy as np

filenames=["./data/pkl/ais-processed-log-2021-0" +str(i)+".pkl" if i<10 else "./data/pkl/ais-processed-log-2021-" +str(i)+".pkl" for i in range(1, 13)]
data=[]
for x in filenames:
    with open(x, "rb") as f:
        data+=pkl.load(f)
total_seq_len=0
for i in data:
    total_seq_len+=len(i["traj"])
    i["traj"]=np.array(i["traj"])
curr_seq_len=0
splits=[]
for i in range(len(data)):
    curr_seq_len+=len(data[i]["traj"])
    if curr_seq_len>=total_seq_len*0.7:
        splits.append(i)
        remaining_len=total_seq_len-curr_seq_len
        curr_seq_len=0
        for j in range(i+1, len(data)):
            curr_seq_len+=len(data[j]["traj"])
            if curr_seq_len>=0.5*remaining_len:
                splits.append(j)
                break
        break
sets=[{"train":data[:splits[0]+1]}, {"valid":data[splits[0]+1:splits[1]+1]}, {"test":data[splits[1]+1:]}]
for i in sets:
    with open(filenames[0].replace("-01", "_"+list(i.keys())[0]).replace("data/pkl", "data/datasets"), "wb") as f:
        pkl.dump(i[list(i.keys())[0]], f)