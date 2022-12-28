import pickle as pkl

with open("./data/datasets/By year/ais-processed-log-2021_train.pkl", "rb") as f:
    data=pkl.load(f)
with open("./ct_dma_train.pkl", "rb") as f:
    data3=pkl.load(f)
lst, lst3=[],[]

for i in data:
    lst.append(len(i["traj"]))
for i in data3:
    lst3.append(len(i["traj"]))

print(max(lst), max(lst3))
print(len(data), len(data3))