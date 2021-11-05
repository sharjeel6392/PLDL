import datetime
import math
import pandas as pd
import warnings
from collections import defaultdict,OrderedDict
import os, json, re, string
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

model = SentenceTransformer('bert-base-nli-mean-tokens')

#Change this function for the file paths if its different.
def load_default_parameters():
    train_file = "data/facebook/processed/fb_train.json"
    dev_file = "data/facebook/processed/fb_dev.json"
    test_file = "data/facebook/processed/fb_test.json"
    output_file = "facebook_kmeans"
    folder_name = "data/facebook/processed/fb/kmeans_predict"
    return train_file,dev_file,test_file,output_file,folder_name

def get_feature_vectors_only(fdict, data):
    #output = {}
    output = defaultdict(list)
    for item in data:
        vect = vectorize(fdict, item["labels"])
        total_labels = float(sum(vect))
        vect[:] = [x /total_labels for x in vect]
        item["message_id"] = item["message_id"]
        output[item["message_id"]] = vect

    return output

def compile_tweet_dict(json_list):
    result = {int(x["message_id"]): x["message"] for x in json_list}
    return result

def create_folder(foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)

def read_json(fname):
    datastore = defaultdict(list)
    if fname:
        with open(fname, 'r') as f:
            datastore = json.load(f)
    return datastore

def get_data_dict (l):
    enuml = enumerate(l)
    fdict = defaultdict(list)
    rdict = defaultdict(list)
    fdict = {k:v for v, k in enuml}
    rdict = {k:v for v, k in fdict.items()}
    return (fdict, rdict)

def vectorize(fdict, labels):
    vect = defaultdict(list)
    vect = [0] * len(fdict)
    for name,number in labels.items():
        vect[fdict[name]] = number
    return vect
    
def write_model_logs_to_json(MODEL_LOG_DIR, results_dict, output_name):
    with open(MODEL_LOG_DIR +"/"+ output_name + ".json", "w") as fp:
        json.dump(results_dict, fp, sort_keys=True, indent=4)
    print ("Saved to "+MODEL_LOG_DIR +"/"+ output_name + ".json")

def read_labeled_data_KMeans(filename):
    answer_counters = defaultdict(list)
    JSONfile = read_json(filename)
    message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, label_dict) = get_data_dict(JSONfile["dictionary"])
    answer_counters = get_feature_vectors_only(fdict, JSONfile["data"])
    return answer_counters,message_dict,label_dict

def preprocess_data(input_train_file_name,input_dev_file_name,input_test_file_name,folder_name):

    create_folder(folder_name)
    create_folder(folder_name + "/logs")
    create_folder(folder_name + "/logs/models")

    train_answer_counters,train_message_dict,label_dict = read_labeled_data_KMeans(input_train_file_name)

    dev_answer_counters,dev_message_dict,label_dict = read_labeled_data_KMeans(input_dev_file_name)

    test_answer_counters,test_message_dict,label_dict = read_labeled_data_KMeans(input_test_file_name)

    return train_answer_counters,dev_answer_counters,label_dict,train_message_dict,dev_message_dict,test_answer_counters,test_message_dict

def deEmoji(sent):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'',sent)

def cleaningData(textDict):
    text = []
    for key in textDict:
        sent = textDict[key]
        words = sent.split()
        table = str.maketrans("","",string.punctuation)
        stripped = [w.translate(table) for w in words]
        words = [word.lower() for word in stripped]
        sent = " ".join(words)
        sent = deEmoji(sent)
        textDict[key] = sent
        text.append(sent)

    return textDict, text

def decompose(components, data, labels):
    axes = ['x','y']
    matrix = PCA(n_components=components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i:axes[i] for i in range(components)}, axis=1, inplace=True)
    df_matrix['labels'] = labels

    return df_matrix

def extarctList(dictionary):

    tempList = [dictionary[key] for key in dictionary]

    return tempList

def readFiles():

    num_clusters = 5
    train_file,dev_file,test_file,output_file,folder_name = load_default_parameters()
    #Reading Data
    train_answer_counters,dev_answer_counters,label_dict,train_message_dict,dev_message_dict,test_answer_counters,test_message_dict = preprocess_data(train_file,dev_file,test_file,folder_name)

    return train_answer_counters, label_dict, train_message_dict

def kl_divergence(p, q):
    loss = 0
    for i in range(len(p)):
        loss += np.sum(np.where(q[i] != 0, p[i] * np.log(p[i] / q[i]), 0))

    return loss

#Problem 1: Generate embeddings

def problem_i(trainText, trainCounter):
    
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    bertTrain = model.encode(trainText)

    embeddings = pd.DataFrame(trainText,columns=['Text'])
    embeddings['embeddings'] = bertTrain

    print(embeddings)

    return bertTrain

def problem_ii(num_clusters, embeddings, trainText):

    cluster = KMeans(n_clusters=num_clusters)
    cluster.fit(embeddings)
    assignment = cluster.labels_

    data = pd.DataFrame(trainText, columns=['Text'])
    data["Clusters"] = assignment
    print(data.head())

    pca_df = decompose(2,embeddings,assignment)
    sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette='Set2')
    plt.show()

    return assignment

def problem_iii(cluster, label, num_clusters=5):

    clusterLabel = [i for i in range(num_clusters)]

    for i in range(num_clusters):
        clusterLabel[i] = [0 for j in range(5)]

    for i in range(len(label)):
        for j in range(5):
            clusterLabel[cluster[i]][j] += label[i][j]

    Label = [0 for i in range(num_clusters)]
    for i in range(num_clusters):
        total = 0
        Label[i] = [clusterLabel[i][j]/sum(clusterLabel[i]) if sum(clusterLabel[i]) != 0 else 0 for j in range(5)]

    return Label

def problem_iv(embeddings,label):

    #inertiaScore = [KMeans(n_clusters=i).fit(embeddings).inertia_ for i in range(4,34)]
    q = [0.2,0.2,0.2,0.2,0.2]
    high = 34
    low = 4
    size = high-low
    minLoss = 999
    minLossInd = -1
    score = [0 for i in range(size)]
    for i in range(low, high):
        clusterModel = KMeans(n_clusters=i)
        clusterModel.fit(embeddings)
        cluster = clusterModel.labels_

        clusterLabel = problem_iii(cluster, label, i)
        loss = 0
        for j in range(i):
            loss += kl_divergence(clusterLabel[j],q)
        loss /= i
        if loss < minLoss:
            minLoss = loss
            minLossInd = i
        score[i-low] = loss

    sns.lineplot(np.arange(4,34), score)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average KL Divergence Loss")
    plt.title("Average KL Loss vs Number of Clusters")

    plt.show()
    print("Minimum loss: ", minLoss, 'for number of clusters: ',minLossInd)

def problem_v(label):
    n_clusters = 5
    cluster = KMeans(n_clusters=n_clusters)
    cluster.fit(label)
    assignment = cluster.labels_

    pca_df = decompose(2,label,assignment)
    sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette='Set2')
    plt.show()

    q = [0.2,0.2,0.2,0.2,0.2]
    high = 34
    low = 4
    size = high-low
    minLoss = 999
    minLossInd = -1
    score = [0 for i in range(size)]
    for i in range(low, high):
        clusterModel = KMeans(n_clusters=i)
        clusterModel.fit(label)
        cluster = clusterModel.labels_

        clusterLabel = problem_iii(cluster, label, i)
        loss = 0
        for j in range(i):
            loss += kl_divergence(clusterLabel[j],q)
        loss /= i
        if loss < minLoss:
            minLoss = loss
            minLossInd = i
        score[i-low] = loss

    sns.lineplot(np.arange(4,34), score)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average KL Divergence Loss")
    plt.title("Average KL Loss vs Number of Clusters")

    plt.show()
    print("Minimum loss: ", minLoss, 'for number of clusters: ',minLossInd)

def problem_vi(embeddings, trainText, label):
    
    n_clusters = 5
    clusterModel = AgglomerativeClustering(n_clusters=n_clusters)
    clusterModel.fit(embeddings)

    assignment = clusterModel.labels_

    data = pd.DataFrame(trainText, columns=['Text'])
    data["Clusters"] = assignment
    print(data.head())

    pca_df = decompose(2,embeddings,assignment)
    sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette='Set2')
    plt.show()

    # Label generation
    cluster = assignment
    clusterLabel = [i for i in range(5)]

    for i in range(5):
        clusterLabel[i] = [0 for i in range(5)]

    for i in range(len(label)):
        clusterLabel[cluster[i]][0] += label[i][0]
        clusterLabel[cluster[i]][1] += label[i][1]
        clusterLabel[cluster[i]][2] += label[i][2]
        clusterLabel[cluster[i]][3] += label[i][3]
        clusterLabel[cluster[i]][4] += label[i][4]

    Label = [0 for i in range(5)]
    for i in range(5):
        total = 0
        Label[i] = [clusterLabel[i][j]/sum(clusterLabel[i]) if sum(clusterLabel[i]) != 0 else 0 for j in range(5)]

    print("Cluster Labels: ")
    for i in range(5):
        print(Label[i])

    # problem vi cluster vs loss

    q = [0.2,0.2,0.2,0.2,0.2]
    high = 34
    low = 4
    size = high-low
    minLoss = 999
    minLossInd = -1
    score = [0 for i in range(size)]
    for i in range(low, high):
        clusterModel = AgglomerativeClustering(n_clusters=i)
        clusterModel.fit(embeddings)
        cluster = clusterModel.labels_

        clusterLabel = problem_iii(cluster, label, i)
        loss = 0
        for j in range(i):
            loss += kl_divergence(clusterLabel[j],q)
        loss /= i
        if loss < minLoss:
            minLoss = loss
            minLossInd = i
        score[i-low] = loss

    sns.lineplot(np.arange(4,34), score)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average KL Divergence Loss")
    plt.title("KL Loss vs Number of Clusters")

    plt.show()
    print("Minimum loss: ", minLoss, 'for number of clusters: ',minLossInd)
    # Label based clustering

    n_clusters = 5
    cluster = AgglomerativeClustering(n_clusters=n_clusters)
    cluster.fit(label)
    assignment = cluster.labels_

    pca_df = decompose(2,label,assignment)
    sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette='Set2')
    plt.show()

    q = [0.2,0.2,0.2,0.2,0.2]
    high = 34
    low = 4
    size = high-low
    minLoss = 999
    minLossInd = -1
    score = [0 for i in range(size)]
    for i in range(low, high):
        clusterModel = AgglomerativeClustering(n_clusters=i)
        clusterModel.fit(label)
        cluster = clusterModel.labels_

        clusterLabel = problem_iii(cluster, label, i)
        loss = 0
        for j in range(i):
            loss += kl_divergence(clusterLabel[j],q)
        loss /= i
        if loss < minLoss:
            minLoss = loss
            minLossInd = i
        score[i-low] = loss

    sns.lineplot(np.arange(4,34), score)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia Score")
    plt.title("Loss vs Number of Clusters")

    plt.show()
    print("Minimum loss: ", minLoss, 'for number of clusters: ',minLossInd)





def main():
    
    train_answer_counters, labels, train_message_dict = readFiles()
    cleanTrainData, cleanTrainText = cleaningData(train_message_dict)
    
    label = extarctList(train_answer_counters)
    
    
    print("Problem 1: Embeddings")
    embeddings = problem_i(cleanTrainText, train_answer_counters)
    input("\nPress enter to continue...")
    
    print("Problem 2: KMeans on Text data ")
    clusters = problem_ii(5, embeddings, cleanTrainText)
    input("\nPress enter to continue...")
    
    print("Problem 3: Label generation")
    clusterLabel = problem_iii(clusters, label)
    for items in clusterLabel:
        print(items)
    input("\nPress enter to continue...")
    
    print("Problem 4: Loss vs number of cluster.")
    problem_iv(embeddings,label)
    input("\nPress enter to continue...")

    print("Problem 5: KMeans on label data and loss vs number of clusters")
    problem_v(label)
    input("\nPress enter to continue...")

    print("Heirarchical clustering (Agglomerative/Bottom up Clustering):")
    problem_vi(embeddings,cleanTrainText, label)

if __name__ == '__main__':
    main() 
