import pickle as pkl
import numpy as np

# filenames=["./data/pkl/ais-processed-log-2021-0" +str(i)+".pkl" for i in range(1, 3)]
class Splitter:
    def __init__(self):
        pass
    def forward(self, filenames):
        for x in range(0, len(filenames)):
            file1=filenames[x]
            with open(file1, "rb") as f:
                data=pkl.load(f)
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
                with open(filenames[x].replace(".pkl", "_"+str(list(i.keys())[0])+".pkl").replace("data/pkl", "data/datasets/monthly"), "wb") as f:
                    pkl.dump(i[list(i.keys())[0]], f)

        data=[]
        # By 2 months
        for x in range(0, len(filenames), 2):
            file1=filenames[x]
            file2=filenames[x+1]
            with open(file1, "rb") as f:
                data1=pkl.load(f)
            with open(file2, "rb") as f:
                data2=pkl.load(f)
            data=data1+data2
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
                with open(filenames[x].replace(".pkl", "-"+"0"+ str(x+2)+"_"+str(list(i.keys())[0])+".pkl" if x<8 else "-"+ str(x+2)+"_"+str(list(i.keys())[0])+".pkl").replace("data/pkl", "data/datasets/bimonthly"), "wb") as f:
                    pkl.dump(i[list(i.keys())[0]], f)


        # Half a year(6 months)
        for x in range(0, len(filenames), 6):
            data=[]
            files=[filenames[x+m] for m in range(6)]
            for i in files:
                with open(i, "rb") as f:
                    print(x)
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
                with open(filenames[x].replace(".pkl", "-"+"0"+ str(x+6)+"_"+str(list(i.keys())[0])+".pkl" if x<8 else "-"+ str(x+6)+"_"+str(list(i.keys())[0])+".pkl").replace("data/pkl", "data/datasets/half_yearly"), "wb") as f:
                    pkl.dump(i[list(i.keys())[0]], f)
            if len(filenames)!=12:
                break
        # By year
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
            with open(filenames[0].replace("-01", "_"+list(i.keys())[0]).replace("data/pkl", "data/datasets/yearly"), "wb") as f:
                pkl.dump(i[list(i.keys())[0]], f)
