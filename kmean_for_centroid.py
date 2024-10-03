import numpy as np
#from math import dist # to calculate distance
import itertools
#import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

old_label = np.genfromtxt("train_label_cancer.csv", delimiter=',', dtype=None,names=None,usecols=(-1),skip_header=1,encoding='UTF-8')
old_name= np.genfromtxt("train_data_cancer.csv", delimiter=',', dtype=None,names=None, usecols=(range(1,)),skip_header=1,encoding='UTF-8')
old_data = np.genfromtxt("train_data_cancer.csv", delimiter=',', dtype=None,names=None,usecols=(range(1,20531)),skip_header=1,encoding='UTF-8')
new_data=np.genfromtxt("test_data_cancer.csv", delimiter=',', dtype=None,names=None,usecols=(range(1,20531)),skip_header=1,encoding='UTF-8')
new_label = np.genfromtxt("test_label_cancer.csv", delimiter=',', dtype=None,names=None,usecols=(1),skip_header=1,encoding='UTF-8')
name=np.genfromtxt("test_data_cancer.csv", delimiter=',', dtype=None,names=None, usecols=(range(1,)),skip_header=1,encoding='UTF-8')
#scaler = preprocessing.StandardScaler()
#old_data_train = scaler.fit_transform(old_data)
#new_data_train = scaler.transform(new_data)
#old_data=old_data_train
#new_data=new_data_train


def similar(*val, threshold=0.1): # max - 2nd max
    val=[*val]
    val.sort()
    
    return  (val[-1]-val[-2])<= threshold
    #return abs(val[0]-val[1])<=threshold and abs(val[1]-val[2])<=threshold and abs(val[0]-val[2])<=threshold

def preprocess(data):   #standardize
    data_train = data
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_scaled = scaler.transform(data_train)
    return data_scaled

def classify(old_label,old_data,new_label,new_data,mode=0):
    
    if mode == 0:
        clf=LogisticRegression()
        clf.fit(old_data,old_label)

        prob = clf.predict_proba(new_data)
        print(prob)
        Rows, Cols = prob.shape
        record_unknown = [i for i in range(Rows) if similar(*prob[i])]
        print(len(record_unknown))
    if mode == 1:
        label = [0 for i in range(len(old_label))]      #byes
        for i in range(len(old_label)): #intitialize label
                if old_label[i] == "KIRC" :label[i] = 0
                elif old_label[i] == "BRCA" :label[i] = 1
                elif old_label[i] == "LUAD" : label[i] = 2
        reg = linear_model.BayesianRidge()
        reg.fit(old_data,label)

        prob = reg.predict(new_data)
        print(prob)

        Rows = len(prob)
        record_unknown = [i for i in range(Rows) if similar(prob[i]-int(prob[i]),0.5)]
        print(len(record_unknown))  #__debug__

    if mode == 2:
        l1=["KIRC","BRCA","LUAD"]
        nRow, nCol=old_data.shape
        c1=gen_old_centroid(old_data,3,old_label,mode=0)
        distance0,distance1,distance2=[],[],[]
        for i in range(0,nRow):
            if old_label[i] == l1[0]: distance0.append(np.linalg.norm(old_data[i]-c1[0]))
            elif old_label[i] == l1[1] : distance1.append(np.linalg.norm(old_data[i]-c1[1]))
            elif old_label[i] == l1[2] : distance2.append(np.linalg.norm(old_data[i]-c1[2]))
        label = [sum(distance0)/len(distance0),sum(distance1)/len(distance1),sum(distance2)/len(distance2)]
        #print("distance:",distance0,"\n",distance1,"\n",distance2,"\nlabel:",label)

        nRow, nCol=new_data.shape
        predict_label,record_unknown = [],[]
        for i in range(0,nRow):
            count=[]
            if np.linalg.norm(new_data[i]-c1[0]) < label[0]: count.append(0)
            if np.linalg.norm(new_data[i]-c1[1]) < label[1]: count.append(1)
            if np.linalg.norm(new_data[i]-c1[2]) < label[2]: count.append(2)
            if len(count) != 1:
                record_unknown.append(i)
                predict_label.append(-1)
            else: predict_label.append(count[0])
            #print("count: ",count)
        print("unknown:",len(record_unknown))
        return record_unknown,predict_label,l1

    return record_unknown

def gen_old_centroid(data,k,old_label,mode=0):#rate is for calculating centroid of unknown data
    nRow,nCol=data.shape
    label = [0 for i in range(len(old_label))] # initialize label

    if mode == 2:
        origin_label,centroid=cluster(data,3,200,mode=2)
        labels=["KIRC","BRCA","LUAD"]
        n=6

        a = [0 for i in range(n)]
        label_permutations=list(itertools.permutations(labels,len(labels))) #every possible
        for i in range(n):
            count_correct=0
            table ={v:k for v,k in enumerate(label_permutations[i])}
            for j in range(0,len(origin_label)):
                if table[origin_label[j]] == old_label[j]:
                    count_correct += 1
                else:
                    pass
            a[i]=(count_correct)/(len(old_label))
        correct_label = label_permutations[a.index(max(a))]
        print("max:",correct_label)
        return correct_label,centroid

        
    if mode == 0:
        for i in range(len(old_label)): #intitialize label
            if old_label[i] == "KIRC" :label[i] = 0
            elif old_label[i] == "BRCA" :label[i] = 1
            elif old_label[i] == "LUAD" : label[i] = 2
            else:
                label[i] = -1
            #print("mode1")
    if mode == 1:
        label=old_label
        #print("mode2")
    
    count = [0 for i in range(k)]
    centroid = np.array([0 for i in range(k*nCol)]).reshape(k,nCol)
    
    for i in range(0,len(label)):
        centroid[label[i]] = np.add(centroid[label[i]], data[i])
        count[label[i]] += 1
    for i in range(k):
        if count[i]==0:
            #print("i is 0",mode)
            centroid[i] = [0 for i in range(nCol)]
        else:
            centroid[i]=centroid[i]/count[i]
    #centroid = np.array([data[label == i].mean(axis=0) for i in range(k)])

    return centroid

def cluster(data, k,max,c=None,mode=0):

    
    centroids = data[np.random.choice(range(len(data)), k, replace=False)] 
    # randomly pick  k points and they don't repeat stor in a array call centroids
    if mode==1: #for clustering 5 label
        centroids = c
    nRow, nCol=data.shape

    count2=0
    while(1):
        distance=[]
        for i in range(0,nRow):
            for j in range(0,k):
                distance.append(np.linalg.norm(data[i]-centroids[j]))

        distance=np.array(distance)
        
        distance=distance.reshape(nRow,k)
        label= np.argmin(distance, axis=1)
        #argmin return the index of the min value
        #print(label)
        cRow,cCol=centroids.shape
        new_centriod = [0 for i in range(cCol*cRow)]    #initialize new_centroid
        new_centriod = np.array(new_centriod)
        new_centriod=new_centriod.reshape(cRow,cCol)
        #print(new_centriod.shape)  #__debug__
        count = [0 for i in range(k)]
        for i in range(0,len(label)):
            new_centriod[label[i]] = np.add(new_centriod[label[i]], data[i])
            #label[i] indicate the label of data[i]
            count[label[i]] += 1
        #print(count)  #__debug__
        for i in range(k):new_centriod[i]=new_centriod[i]/count[i]
        count2+=1
        #print(count2) #iterate times
        if count2 == max:
            break
        flag=True
        for i in range(k):
            if np.linalg.norm(centroids[i]-new_centriod[i])!=0:
                flag=False
                centroids = new_centriod
                break
        if flag:
            centroids = new_centriod
            break

    if mode <= 1: return label
    if mode == 2: return label,centroids

def gen_rate(label,targetlabel,table):
   
    labels=["COAD","PRAD"]
    n=2
    a = [0 for i in range(n)]
    label_permutations=list(itertools.permutations(labels,len(labels))) #every possible
    for i in range(n):
        count_correct=0
        table1={v+3:k for v,k in enumerate(label_permutations[i])}
        table2={0:table[0],1:table[1],2:table[2]}
        table3 = table1|table2
        for j in range(0,len(label)):
            if table3[label[j]] == targetlabel[j]:
                count_correct += 1
            else:
                pass
        a[i]=((count_correct)/(len(new_label)))
    return max(a),np.argmax(a),label_permutations[np.argmax(a)]

def test():
    
    #index=classify(old_label,old_data_train,new_label,new_data_train,mode=0)
    index,predict_label,table=classify(old_label,old_data,new_label,new_data,mode=2)
    #print("index:",len(index))
    tdata = np.array([new_data[i] for i in index])#tdata is the unknown data
    tlabel = np.array([new_label[i] for i in index])#tlabel is the unknown labels
    label = cluster(tdata,2,200)#label is the output of unknown clustering they are numbers
    #rate=gen_rate(label,tlabel,1)
    #c1=gen_old_centroid(old_data,3,old_label)
    #c2=gen_old_centroid(tdata,2,label,mode=1)
    #data = np.concatenate((old_data,new_data),axis=0)
    #label= np.concatenate((old_label,new_label),axis=0)
    label_index=0
    for i in range(len(predict_label)):
        if(predict_label[i] == -1):
            predict_label[i]=label[label_index]+3
            label_index+=1
    #c=np.concatenate((c1,c2),axis=0)
    #return gen_rate(np.array(list(cluster(data,5,200,c,mode=1))[469:]),np.array(list(label)[469:]))
    return gen_rate(predict_label,new_label,table)


result = [test()[0] for i in range(1)]
print("Accuracy=",sum(result)/len(result))





