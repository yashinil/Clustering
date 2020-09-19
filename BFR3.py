#command to run the task
#python BFR3.py hw6_clustering.txt 10 output.txt
#python BFR3.py input.txt 2 output.txt
# {"pyspark.mllib": "pyspark.rdd.RDD", "pyspark.ml": "pyspark.sql.DataFrame"}
from sklearn.metrics.cluster import normalized_mutual_info_score
from pyspark import SparkContext, SparkConf
import pyspark
import time
import sys
from pyspark.mllib.clustering import KMeans, KMeansModel
from sklearn.cluster import KMeans
import numpy as np
from numpy import array
from math import sqrt

#timer start
start_time=time.time()

#creating a spark context
conf = pyspark.SparkConf().setMaster("local[*]").setAppName("bfr").setAll([('spark.executor.memory', '8g'), ('spark.executor.cores', '3'), ('spark.cores.max', '3'), ('spark.driver.memory','8g')])
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
sc = SparkContext(conf=conf)

#creating a spark context
#sc = SparkContext('local[*]','first')
sc.setLogLevel('ERROR')

#take command line inputs
input_path = sys.argv[1]
input_clusters = int(sys.argv[2])
output_path = sys.argv[3]

#initalizing Discard set (DS), Compression set (CS), Retained set (RS), final clustering results list and intermediate results list
discard_set=list()
compression_set=list()
retained_set=set()
clustering_results=list()
intermediate_results=list()

#reading the dataset into an RDD
originalRDD=sc.textFile(input_path).map(lambda line: line.split(",")).map(lambda line: [float(item) for item in line]).persist(pyspark.StorageLevel.DISK_ONLY)

input_length=originalRDD.count()
parts=[input_length//5]*5

if input_length%5!=0:
    for i in range(input_length%5):
        parts[i]+=1

# sample 20% data in the first part
first_part = originalRDD.takeSample(False,parts[0],10)
# print(first_part[:10])
first_part = sc.parallelize(first_part).repartition(5).persist(pyspark.StorageLevel.DISK_ONLY)

#adding first sample data in data_used and reoving those from originalRDD
data_used = first_part.map(lambda line: int(line[0])).collect()
data_used = set(data_used)
originalRDD = originalRDD.filter(lambda line: True if int(line[0]) not in data_used else False)

# Trains a k-means model.
# make clusters model with 5*number of clusters
# predicting results for every point in the first part
train_data = first_part.map(lambda line: array(line[2:]))
train_data = np.array(train_data.collect())
kmeans = KMeans(n_clusters=input_clusters*5, random_state=0).fit(train_data)
print(kmeans.labels_)
results = first_part.map(lambda line: (kmeans.predict([line[2:]]),[int(line[0])])).map(lambda line: (line[0].tolist()[0],line[1])).reduceByKey(lambda a,b: a+b).persist(pyspark.StorageLevel.DISK_ONLY)

# seperating the clusters with only one point and adding them to the retained set
RetainedSetRDD = results.filter( lambda line: True if len(line[1]) == 1 else False ).map(lambda line: line[1][0])
retained_set.update(set(RetainedSetRDD.collect()))
# print(retained_set)

# Running K means on the candidates for Discard Set
remaining = results.filter( lambda line: True if len(line[1]) > 1 else False ).flatMap(lambda line: line[1])
remaining = set(remaining.collect())
# print(remaining)

#get data for the points in retained points and in remaining points from the first part
retained_points = first_part.filter( lambda line: True if line[0] in retained_set else False ).persist(pyspark.StorageLevel.DISK_ONLY)
remaining_points = first_part.filter( lambda line: True if line[0] in remaining else False ).persist(pyspark.StorageLevel.DISK_ONLY)

# calculating for discard set
# make clusters model with number of clusters equal to input_clusters
# predicting data and getting the new results
train_data = remaining_points.map(lambda line: array(line[2:]))
train_data = np.array(train_data.collect())
kmeans = KMeans(n_clusters=input_clusters, random_state=0).fit(train_data)
print(kmeans.labels_)
results = remaining_points.map(lambda line: (kmeans.predict([line[2:]]),line[0],line[2:])).map(lambda line: (line[0].tolist()[0],line[1],line[2])).persist(pyspark.StorageLevel.DISK_ONLY)
    
clustering_results = results.map(lambda line: (line[1],line[0])).collect()
# print(clustering_results)

results = results.map(lambda line: (line[0],[line[2]])).reduceByKey(lambda a,b: a+b)

for x in results.collect():
    d=len(x[1][0])
    summation=[0]*d
    sq_summation=[0]*d
    for element in x[1]:
        for i in range(len(element)):
            summation[i]+=element[i]
            sq_summation[i]+=pow(element[i],2)

    discard_set.append([ x[0], len(x[1]), summation, sq_summation ])
# print(discard_set)

# Running K means on the points in retained set
# make clusters model with number of clusters equal to input_clusters*5
# predicting data and getting the new results
if retained_points.count()>=5*input_clusters:
    train_data = retained_points.map(lambda line: array(line[2:]))
    train_data = np.array(train_data.collect())
    kmeans = KMeans(n_clusters=input_clusters*5, random_state=0).fit(train_data)
    print(kmeans.labels_)   
    results = retained_points.map(lambda line: (kmeans.predict([line[2:]]),[line])).map(lambda line: (line[0].tolist()[0],line[1])).reduceByKey(lambda a,b: a+b)
    for x in results.take(10):
        print(x)

    # creating compression set and retained set
    compressionSetRDD = results.filter(lambda line: True if len(line[1]) > 1 else False).map(lambda line: line[1])

    for x in compressionSetRDD.collect():
        d=len(x[0][2:])
        summation=[0]*d
        sq_summation=[0]*d
        cs_points=list()
        for element in x:
            for i in range(len(element[2:])):
                summation[i]+=element[i+2]
                sq_summation[i]+=pow(element[i+2],2)
            cs_points.append(element[0])

        compression_set.append([ len(x), summation, sq_summation, cs_points ])
    # print(compression_set)

    cs_points=0
    for x in compression_set:
        cs_points+=len(x[3])
    # print(cs_points)

    RetainedSetRDD = results.filter( lambda line: True if len(line[1]) == 1 else False ).map(lambda line: line[1][0])
    # for x in RetainedSetRDD.collect():
    #     print(x)
else:
    RetainedSetRDD = retained_points
    retained_count = RetainedSetRDD.count()
    cs_points = 0

intermediate_results.append((len(clustering_results),len(compression_set),cs_points,retained_count))
print(intermediate_results)

def get_ds_mahalanobis(point):
    dimensions = point[2:]
    distance = 99
    best_fit = [ point[0], -1, distance ]
    diff = 2*sqrt(len(dimensions))
    
    for tpl in discard_set:
        md = 0
        for i in range(len(dimensions)):
            sigma = tpl[3][i]/tpl[1] - pow(tpl[2][i]/tpl[1],2)
            sigma = sqrt(sigma)
            ci = tpl[2][i]/tpl[1]
            md += pow((dimensions[i] - ci)/sigma,2)
        md = sqrt(md)
        if md < distance and md < diff:
            best_fit[1] = tpl[0]
            best_fit[2] = md
            distance = md

    if best_fit[1] != -1:
        return (point,best_fit[1])
    return point

def get_cs_mahalanobis(point):
    dimensions = point[2:]
    distance = 99
    best_fit = [ point[0], -1, distance ]
    diff = 2*sqrt(len(dimensions))
    
    for tpl in compression_set:
        md = 0
        for i in range(len(dimensions)):
            sigma = tpl[2][i]/tpl[0] - pow(tpl[1][i]/tpl[0],2)
            sigma = sqrt(sigma)
            ci = tpl[1][i]/tpl[0]
            md += pow((dimensions[i] - ci)/sigma,2)
        md = sqrt(md)
        if md < distance and md < diff:
            best_fit[1] = tpl[3][0]
            best_fit[2] = md
            distance = md

    if best_fit[1] != -1:
        return (point,best_fit[1])
    return point

iteration=1
while originalRDD.isEmpty() != True:
    # sample 20% data
    section = originalRDD.takeSample(False,parts[iteration],10)
    section = sc.parallelize(section).repartition(5).persist(pyspark.StorageLevel.DISK_ONLY)
    
    #adding sample data in data_used and removing those from originalRDD
    data_used_temp = section.map(lambda line: int(line[0])).collect()
    data_used_temp = set(data_used_temp)
    data_used.update(data_used_temp)
    originalRDD = originalRDD.filter(lambda line: True if int(line[0]) not in data_used else False)
    
    #compare each of the new points to DS using mahalanobis Distance
    mahalanobisRDD = section.map(get_ds_mahalanobis).persist(pyspark.StorageLevel.DISK_ONLY)
    temp_ds_results = mahalanobisRDD.filter(lambda line: True if len(line) == 2 else False).collect()

    for x in temp_ds_results:
        for tpl in discard_set:
            if tpl[0] == x[1]:
                tpl[1]+=1
                for i in range(len(tpl[2])):
                    tpl[2][i]+=x[0][i+2]
                    tpl[3][i]+=pow(x[0][i+2],2)
        clustering_results.append((x[0][0],x[1]))

    # print(discard_set)
    # print(clustering_results)

    mahalanobisRDD = mahalanobisRDD.filter(lambda line: True if len(line) != 2 else False).map(get_cs_mahalanobis).persist(pyspark.StorageLevel.DISK_ONLY)
    temp_cs_results = mahalanobisRDD.filter(lambda line: True if len(line) == 2 else False).collect()

    for x in temp_cs_results:
        for tpl in compression_set:
            if tpl[3][0] == x[1]:
                tpl[0]+=1
                for i in range(len(tpl[1])):
                    tpl[1][i]+=x[0][i+2]
                    tpl[2][i]+=pow(x[0][i+2],2)
                tpl[3].append(x[0][0])

    # print(compression_set)

    mahalanobisRDD = mahalanobisRDD.filter(lambda line: True if len(line) != 2 else False)
    RetainedSetRDD = RetainedSetRDD.union(mahalanobisRDD).repartition(5).persist(pyspark.StorageLevel.DISK_ONLY)

    # Running K means on the points in retained set\
    # make clusters model with number of clusters equal to input_clusters*5
    # predicting data and getting the new results
    if RetainedSetRDD.count() >= 5*input_clusters:
        train_data = RetainedSetRDD.repartition(5).map(lambda line: array(line[2:]))
        train_data = np.array(train_data.collect())
        kmeans = KMeans(n_clusters=input_clusters*5, random_state=0).fit(train_data)
        print(kmeans.labels_)
        results = RetainedSetRDD.map(lambda line: (kmeans.predict([line[2:]]),[line])).map(lambda line: (line[0].tolist()[0],line[1])).reduceByKey(lambda a,b: a+b)

        # creating compression set and retained set
        compressionSetRDD = results.filter(lambda line: True if len(line[1]) > 1 else False).map(lambda line: line[1])
        for x in compressionSetRDD.collect():
            d=len(x[0][2:])
            summation=[0]*d
            sq_summation=[0]*d
            cs_points=list()
            for element in x:
                for i in range(len(element[2:])):
                    summation[i]+=element[i+2]
                    sq_summation[i]+=pow(element[i+2],2)
                cs_points.append(element[0])
       
            distance = 99
            best_fit = [ -1, distance ]
            diff = 2*sqrt(d)
    
            for tpl in compression_set:
                md = 0
                for i in range(d):
                    sigma = tpl[2][i]/tpl[0] - pow(tpl[1][i]/tpl[0],2)
                    sigma = sqrt(sigma)
                    ci = tpl[1][i]/tpl[0]
                    md += pow((summation[i]/len(cs_points) - ci)/sigma,2)
                md = sqrt(md)
                if md < distance and md < diff:
                    best_fit[0] = tpl[3][0]
                    best_fit[1] = md
                    distance = md

            if best_fit[0] != -1:
                for tpl in compression_set:
                    if tpl[3][0] == best_fit[0]:
                        tpl[0]+=len(cs_points)
                        for i in range(len(tpl[1])):
                            tpl[1][i]+=summation[i]
                            tpl[2][i]+=sq_summation[i]
                        tpl[3]+=cs_points
            else:    
                compression_set.append([ len(x), summation, sq_summation, cs_points ])
        # print(compression_set)

        RetainedSetRDD = results.filter( lambda line: True if len(line[1]) == 1 else False ).map(lambda line: line[1][0])
    
    retained_count = RetainedSetRDD.count()
    
    if originalRDD.isEmpty() == True:
        cs_in_ds=list()
        for tpl in compression_set:
            distance = 99
            best_fit = [ -1, distance ]
            diff = 2*sqrt(d)
            
            for tpl1 in discard_set:
                md = 0
                for i in range(len(tpl1[2])):
                    sigma = tpl1[3][i]/tpl1[1] - pow(tpl1[2][i]/tpl1[1],2)
                    sigma = sqrt(sigma)
                    ci = tpl1[2][i]/tpl1[1]
                    md += pow((tpl[1][i]/tpl[0] - ci)/sigma,2)
                md = sqrt(md)
                if md < distance and md < diff:
                    best_fit[0] = tpl1[0]
                    best_fit[1] = md
                    distance = md

            if best_fit[0] != -1:
                for tpl1 in discard_set:
                    if tpl1[0] == best_fit[0]:
                        tpl1[1]+=len(tpl[3])
                        for i in range(len(tpl1[2])):
                            tpl1[2][i]+=tpl[1][i]
                            tpl1[3][i]+=tpl[2][i]
                        for p in tpl[3]:
                            clustering_results.append((p,tpl1[0]))
                cs_in_ds.append(tpl)

        for tpl in cs_in_ds:
            if tpl in compression_set:
                compression_set.remove(tpl)
        # print(cs_in_ds)
    # print(compression_set)
        cs_points=0
        for x in compression_set:
            cs_points+=len(x[3])
        # print(cs_points)

        intermediate_results.append((len(clustering_results),len(compression_set),cs_points,retained_count))
        print(intermediate_results)

        for tpl in compression_set:
            for x in tpl[3]:
                clustering_results.append((x,-1))

        for p in RetainedSetRDD.collect():
            clustering_results.append((p[0],-1))
    else:
        cs_points=0
        for x in compression_set:
            cs_points+=len(x[3])
        # print(cs_points)

        iteration+=1

        intermediate_results.append((len(clustering_results),len(compression_set),cs_points,retained_count))
        print(intermediate_results)
    print(len(clustering_results))
clustering_results.sort()
# print(len(clustering_results))

output_file = open(output_path,'w')
output_file.write("The intermediate results:\n")
for i in range(len(intermediate_results)):
    output_file.write("Round "+str(i+1)+": "+str(intermediate_results[i][0])+","+str(intermediate_results[i][1])+","+str(intermediate_results[i][2])+","+str(intermediate_results[i][3])+"\n")

output_file.write("\nThe clustering results:\n")
for res in clustering_results[:-1]:
    output_file.write(str(int(res[0]))+","+str(res[1])+"\n")
output_file.write(str(int(clustering_results[-1][0]))+","+str(clustering_results[-1][1]))
output_file.close()
    
print("Duration: %s" % (time.time() - start_time))

my_ans = [item[1] for item in clustering_results]
# print(my_ans)

original_ans = sc.textFile(input_path).map(lambda line: line.split(",")[:2]).map(lambda line: tuple([float(item) for item in line])).collect()
original_ans.sort()
original_ans = [int(item[1]) for item in original_ans]
# print(len(original_ans))

print( normalized_mutual_info_score( original_ans, my_ans ) )