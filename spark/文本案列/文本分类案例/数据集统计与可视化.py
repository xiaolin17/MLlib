import os
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_stats(f_stats):
    with open(f_stats) as file:
        data = pd.read_csv(file)
        return data

def get_categories(data_pd):
    df_file_count = data_pd.groupby('dirname').size()
    df_total_word_count = data_pd.groupby('dirname')[['total_count']].agg(['mean'])
    df_word_count = data_pd.groupby('dirname')[['diff_count']].agg(['mean'])
    df_file_count.plot(kind='bar',rot=75,figsize=(20,12),fontsize=20)
    plt.savefig('file_count.png')
    df_total_word_count.plot(kind='bar',rot=75,figsize=(20,12),fontsize=20)
    plt.savefig('total_word_count.png')
    df_word_count.plot(kind='bar',rot=75,figsize=(20,12),fontsize=20)
    plt.savefig('diff_word_count.png')

def get_word_count(file_path):
    file = open(file_path,"rb")
    word_count={}
    total_count = 0
    for word in file.read().split():
        total_count += 1
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
    file.close()

    different_count = len(word_count.keys())
    return (total_count,different_count)

def overview_dataset(dataset):
    f_overview = open('overview_data.csv','w')
    header = "dirname,total_count,diff_count,file_size"
    f_overview.write(header+"\n")
    for root,dirs,file in os.walk(dataset):
        for dir_name in dirs:
            dir_path = os.path.join(root,dir_name)
            files_name = os.listdir(dir_path)
            for flie_name in files_name:
                flie_path = os.path.join(dir_path,flie_name)
                file_size = os.path.getsize(flie_path)
                (total_count,different_count) = get_word_count(flie_path)
                content = dir_name+','+str(total_count)+','+str(different_count)+','+str(file_size)+"\n"
                f_overview.write(content)
    f_overview.close()

overview_dataset('20news-19997/20_newsgroups')
data = get_stats('overview_data.csv')
get_categories(data)