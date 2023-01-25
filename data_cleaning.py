import pickle as pkl
import datetime as dt
import pandas as pd
from math import *
from haversine import *
import vptree
import shapefile
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pytz
import splitter
import movingpandas as mpd
import geopandas as gpd
from fiona.crs import from_epsg
from shapely.geometry import Point

from data_processing import Processor
start=dt.datetime.now()
p=Processor()
some_value=1

# res=p.json_to_pkl("./ais-processed-log-2019-03.json", "./ais-processed-log-2019-03.pkl")
# datas=p.process(["./data/SG/2021/ais-processed-log-2021-0"+str(i)+".json" for i in range(1,3)], multiple_files=True)
# with open("./ais-processed-log-2019-03.pkl", "rb") as f:
#     data=pkl.load(f)
datas=[]
datas=["./data/SG/202"+str(some_value)+"/ais-processed-log-202"+str(some_value)+"-0" +str(i)+".pkl" if i<10 else "./data/SG/202"+str(some_value)+"/ais-processed-log-202"+str(some_value)+"-" +str(i)+".pkl" for i in range(1, 13)]
# datas=p.process(["./data/SG/202"+str(some_value)+"/ais-processed-log-202"+str(some_value)+"-0" +str(i)+".json" for i in range(1, 7)], multiple_files=True)
aistype5data=[]
for filename in datas:
    with open(filename, "rb") as f:
        data=pkl.load(f)
    aistype5data = []
    for i in data:
        if i["type"]==5:
            aistype5data.append(i)

    req_mmsis=[]
    for i in aistype5data:
        if i["shiptype"]<90 and i["shiptype"]>69 and i["mmsi"] not in req_mmsis:
            req_mmsis.append(i["mmsi"])

    req_data=[]
    for i in data:
        if i["mmsi"] in req_mmsis and i["type"]!=5:
            req_data.append(i)
    req_data_sorted=sorted(req_data, key=lambda x: x["mmsi"])

    # print(req_data_sorted[0:100])

    sorted_data=[]
    mmsi_list=[]
    prev_mmsi=0
    val=-1
    for i in req_data_sorted:
        curr_mmsi=i["mmsi"]
        if curr_mmsi!=prev_mmsi:
            val+=1
            mmsi_list.append(i["mmsi"])
            # sorted_data.append({"mmsi": i["mmsi"], "traj": [[i["lat"], i["lon"], i["speed"], i["course"]]]})#rearrange the parameters as needed
            temp_dict = {key: val for key, val in i.items() if key != "mmsi"}
            time=temp_dict["time"]
            temp_dict["time"]=pd.to_datetime(time)
            sorted_data.append({"mmsi":i["mmsi"], "data": [temp_dict]})
        else:
            # sorted_data[val]["traj"].append([i["lat"], i["lon"], i["speed"], i["course"]])
            temp_dict = {key: val for key, val in i.items() if key != "mmsi"}
            time=temp_dict["time"]
            temp_dict["time"]=pd.to_datetime(time)
            sorted_data[val]["data"].append(temp_dict)
        prev_mmsi=curr_mmsi
    #Remove infeasible speed and pos messages(SOG>40 or the message is on land, or COG is 'null', or the lat/lon coords are out of the circle defined)
    for i in sorted_data:
        rem=[]
        for j in range(len(i["data"])):
            lat1, lon1=i["data"][j]["lat"], i["data"][j]["lon"]
            if abs(lat1)>90 or abs(lon1)>180:
                rem.append(j)
            elif i["data"][j]["speed"] is None:
                rem.append(j)
            elif i["data"][j]["speed"]>30:
                rem.append(j)
            elif i["data"][j]["course"] is None:
                rem.append(j)
            elif haversine((lat1, lon1), (1.11161, 103.63893), unit=Unit.KILOMETERS)>30: #Circle is centered at about 1.11161, 103.63893
                rem.append(j)
        for j in rem[::-1]:
            i["data"].pop(j)

    #Remove moored(status 5) or at-anchor(status 1) vessels
    rem=[]
    for i in range(len(sorted_data)):
        for j in sorted_data[i]["data"]:
            try:
                if j["status"]==1 or j["status"]==5:
                    rem.append(i)
                    break
            except KeyError:
                pass
    for i in rem[::-1]:
        sorted_data.pop(i)
    #Remove AIS observations whose distance to coastline is smaller than 2km
    s=shapefile.Reader("./ne_50m_coastline/ne_50m_coastline.shp")

    def geoddist(p1, p2):
      # p1, p2 are of format (lon, lat)
      return haversine((p1[1], p1[0]), (p2[1], p2[0]))

    points=[]
    for i in s.shapeRecords():
        for j in i.shape.__geo_interface__["coordinates"]:
            if len(j)>2:
                for k in j:
                    points.append(k)
            else:
                points.append(j)
    coast = vptree.VPTree(points, geoddist)
    for i in sorted_data:
        rem=[]
        for j in range(len(i["data"])):
            lat, lon=i["data"][j]["lat"], i["data"][j]["lon"]
            c = coast.get_nearest_neighbor((lon, lat))
            if c[0]<2:
                rem.append(j)
        for j in rem[::-1]:
            i["data"].pop(j)

    #Remove AIS tracks whose length is smaller than 20 or those last less than 4h
    rem=[]
    for i in range(len(sorted_data)):
        if len(sorted_data[i]["data"])<20:
            rem.append(i)
        elif (sorted_data[i]["data"][-1]["time"]-sorted_data[i]["data"][0]["time"]).total_seconds()<14400:
            rem.append(i)
    for i in rem[::-1]:
        sorted_data.pop(i)

    #Remove abnormal messages. An AIS message is considered as abnormal if the empirical speed
    #(calculated bydividing the distance travelled by the corresponding interval between the two consecutive messages) is unrealistic, in this case >40 knots
    for i in sorted_data:
        rem=[]
        for j in range(len(i["data"])-1):
            lat1, lon1, lat2, lon2=i["data"][j]["lat"], i["data"][j]["lon"], i["data"][j+1]["lat"], i["data"][j+1]["lon"]
            haversine_dist=haversine((lat1, lon1), (lat2, lon2), unit=Unit.NAUTICAL_MILES)
            try:
                empirical_speed=(haversine_dist/(i["data"][j+1]["time"]-i["data"][j]["time"]).total_seconds()/3600)
                if empirical_speed>40:
                    rem.append(j+1)#Delete the second message, change if required
            except ZeroDivisionError:
                rem.append(j) #Delete first message if timestamps are the same (coinflip said so)
                pass
        for j in rem[::-1]:
            i["data"].pop(j)

    #Down-sample AIS trajectory data with a sampling rate of 5-minute.
    # for i in sorted_data:
    #     rem=[]
    #     curr_timestamp=i["data"][0]["time"]
    #     for j in range(1, len(i["data"])):
    #         if (i["data"][j]["time"] - curr_timestamp).total_seconds()<300:
    #             rem.append(j)
    #         else:
    #             curr_timestamp=i["data"][j]["time"]
    #     for j in rem[::-1]:
    #         i["data"].pop(j)
    for i in sorted_data:
        df = pd.DataFrame(
            [{'geometry': Point(i["data"][j]["lat"], i["data"][j]["lon"]), 't': i["data"][j]["time"]} for j in
             range(len(i["data"]))]).set_index('t')
        gdf = gpd.GeoDataFrame(df, crs=from_epsg(31256))
        traj = mpd.Trajectory(gdf, 1)
        traj.add_speed(False, 'sog')
        traj.add_direction(False, 'cog')
        new = []
        mmsi = i["mmsi"]
        start_time = i["data"][0]["time"]
        new.append(i["data"][0])
        # times = [i["data"][j]["time"] for j in range(len(i["data"]))]
        curr_time = start_time
        ten_min_interval_counter = 1
        i_step = 0
        for message_1 in i["data"]:
            skip = False
            if (message_1["time"] - curr_time).total_seconds() < 600:
                skip = True
            elif (message_1["time"] - curr_time).total_seconds() == 600:
                new.append(message_1)
                ten_min_interval_counter += 1
                curr_time = start_time + dt.timedelta(seconds=600)
            elif (message_1["time"] - curr_time).total_seconds() > 600:
                lat, lon = traj.get_position_at(i["data"][0]["time"] + dt.timedelta(
                    seconds=600 * ten_min_interval_counter)).x, traj.get_position_at(
                    i["data"][0]["time"] + dt.timedelta(seconds=600 * ten_min_interval_counter)).y
                t1 = i["data"][i_step - 1]["time"]
                t2 = message_1["time"]
                p1 = traj.get_position_at(t1)  # Empirical speed between the two original points(for which the new point is between)
                p2 = traj.get_position_at(t2)  # Empirical speed between the two original points(for which the new point is between)
                sog = haversine((p1.x, p1.y), (p2.x, p2.y), unit=Unit.NAUTICAL_MILES) * 3600 / (
                        t2 - t1).total_seconds()  # Empirical speed between the two original points(for which the new point is between)
                cog = traj.get_segment_between(t1, t2).get_direction()
                new.append({"mmsi:": mmsi, "speed": sog, "lon": lon, "lat": lat, "course": cog,
                            "time": i["data"][0]["time"] + dt.timedelta(seconds=600 * ten_min_interval_counter)})
                ten_min_interval_counter += 1
                curr_time = start_time + dt.timedelta(seconds=600)
            i_step += 1
        i["data"] = new

    #Split long trajectories to shorter ones with the maximum sequence length of T + L(split trajectories into ones that are of length [3+(3~17)]hrs. Given T+1 points, predict pos L timesteps later).
    for i in range(len(sorted_data)):
        num=1
        if len(sorted_data[i]["data"])>120:
            num=int(len(sorted_data[i]["data"])/120)
            temp=[]
            for j in range(num-1):
                temp.append(sorted_data[i]["data"][120*num:120*(num+1)])
            temp.append(sorted_data[i]["data"][120*(num-1):-30])
            temp.append(sorted_data[i]["data"][-30:])
            sorted_data[i]["data"]=temp
        else:
            sorted_data[i]["data"]=[sorted_data[i]["data"]]

    #Convert the data into a readable format
    scalers=[MinMaxScaler() for i in range(4)]
    values=[[],[],[],[]]#lat lon sog cog
    for x in sorted_data:
        for i in x["data"]:
            for j in i:
                values[0].append([j["lat"]])
                values[1].append([j["lon"]])
                values[2].append([j["speed"]])
                values[3].append([j["course"]])

    new_values=[[],[],[],[]]

    #Standardize values for lat, lon, sog and cog
    for i in range(len(values)):
        scalers[i]=scalers[i].fit(values[i])
        new_values[i]=scalers[i].transform(values[i])
    temp=0
    sorted_data_final=[]
    for x in sorted_data:
        sorted_data_final.append({"mmsi": x["mmsi"], "traj": []})
        for i in x["data"]:
            for j in i:
                timestamp=(j["time"]-dt.datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
                sorted_data_final[-1]["traj"].append([new_values[0][temp][0], new_values[1][temp][0], new_values[2][temp][0], new_values[3][temp][0], timestamp, x["mmsi"]])
                temp+=1

    #Rewrite the data into pkl file
    # with open("./data/pkl/"+filename.split('/')[-1][:-4]+"pkl", "wb") as f:
    #     pkl.dump(sorted_data_final, f)

# filenames_2020=["./data/pkl/ais-processed-log-2020-0" +str(i)+".pkl" if i<10 else "./data/pkl/ais-processed-log-2020-" +str(i)+".pkl" for i in range(1, 13)]
# filenames_2021=["./data/pkl/ais-processed-log-2021-0" +str(i)+".pkl" if i<10 else "./data/pkl/ais-processed-log-2021-" +str(i)+".pkl" for i in range(1, 13)]
# splitter=splitter.Splitter()
# Splitter(filenames_2020)
# splitter.forward(filenames_2021)
#     file1="./data/pkl/"+filename.split('/')[-1][:-4]+"pkl"
#     with open(file1, "rb") as f:
#         data=pkl.load(f)
    data=sorted_data_final
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
        with open("./data/datasets/monthly/"+filename.split('/')[-1][:-4]+"_"+str(list(i.keys())[0])+".pkl", "wb") as f:
            pkl.dump(i[list(i.keys())[0]], f)

end=dt.datetime.now()
time_taken=end-start
print("Time taken:", time_taken)