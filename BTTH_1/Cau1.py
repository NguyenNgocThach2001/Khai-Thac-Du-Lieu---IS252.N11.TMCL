from cmath import nan
import pandas as pd
import numpy as np
import math
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt

def read_csv(path):
    df = pd.read_csv(path)
    return df

def write_csv(path, df):
    df.to_csv(path,index=False) 

def first_look_data(df):
    #Entire the data
    print(df.describe())

    #Loop each column to count cell that is missing value.
    cnt = {}
    for column in df.head():
        cnt[column] = str(len(df[(df[column].isnull())]) / len(df) * 100)
    id = 0
    for x, y in reversed(cnt.items()):
        print(x + " " + y)
        id+=1
        if(id == 5):
            break
    
    # relationship between age, sex, survived
    women = df[df['sex'] == 'female']
    men = df[df['sex'] == 'male']
    fig = plt.figure()
    fig.suptitle('Women', fontsize=20)
    plt.hist(women[women['survived'] == 1].age.dropna(), bins = 18, alpha = 0.8, label='survived', color='blue', edgecolor = 'black')
    plt.hist(women[women['survived'] == 0].age.dropna(), bins = 18, alpha = 0.7, label='not survived', color='red', edgecolor = 'black')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend()
    fig = plt.figure()  
    fig.suptitle('Men', fontsize=20)
    plt.hist(men[men['survived'] == 1].age.dropna(), bins = 18, alpha = 0.8, label='survived', color='blue', edgecolor = 'black')
    plt.hist(men[men['survived'] == 0].age.dropna(), bins = 45, alpha = 0.7, label='not survived', color='red', edgecolor = 'black')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend()


    #relationship between embarked, pclass, survived
    percent_list = []
    for it in df['pclass'].unique():
        pclass_df = df[df['pclass'] == it]
        pclass_percent = len(pclass_df[pclass_df['survived'] == 1].pclass.dropna()) / len(pclass_df) * 100
        percent_list.append(pclass_percent)
    
    fig = plt.figure()  
    fig.suptitle('Pclass', fontsize=20)
    plt.bar(df['pclass'].unique(), percent_list)
    plt.xticks(df['pclass'].unique())
    plt.ylabel('servived percent')
    plt.xlabel('pclass')

    
    percent_list = []
    unique_embarked_listdff = [x for x in df['embarked'].unique() if str(x) != 'nan']
    for it in unique_embarked_listdff:
        embarked_df = df[df['embarked'] == it]
        embarked_percent = len(embarked_df[embarked_df['survived'] == 1].embarked.dropna()) / len(embarked_df) * 100
        percent_list.append(embarked_percent)
    
    fig = plt.figure()  
    fig.suptitle('Embarked', fontsize=20)
    plt.bar(unique_embarked_listdff, percent_list)
    plt.xticks(unique_embarked_listdff)
    plt.ylabel('survived percent')
    plt.xlabel('embarked')

    # Have family

    fig = plt.figure()
    fig.suptitle('Family', fontsize=20)
    plt.hist(df[df['survived'] == 1].sibsp.dropna() + df[df['survived'] == 1].parch.dropna(), bins = 18, alpha = 0.8, label='survived', color='blue', edgecolor = 'black')
    plt.hist(df[df['survived'] == 0].sibsp.dropna() + df[df['survived'] == 0].parch.dropna(), bins = 45, alpha = 0.7, label='not survived', color='red', edgecolor = 'black')
    plt.ylabel('Count')
    plt.xlabel('Family')
    plt.legend()
    plt.show()  

def min_max_normalization(df): 
    max = df['fare'].max()
    min = df['fare'].min()
    df['fare'] = df['fare'].apply(lambda x: (x - min)/(max - min))
    return df

def preprocess_data(df, drop_list):
    #drop trash column
    df = df.drop(columns = drop_list)
    df.reset_index(drop=True, inplace=True)
    #Fill missing with mean (age column)
    mean = df['age'].sum() / len(df)
    df['age'] = df['age'].apply(lambda x: mean if str(x) == "nan" else x)

    # Fill missing with mode (embarked)
    embarked_mode_value = df['embarked'].mode()
    df['embarked'].apply(lambda x: embarked_mode_value if (str(x) == "nan" or str(x) == "") else x)
    #Reduce noise with binning (age column)
    binning_num = 8
    binning_batch = int(math.ceil((len(df) / binning_num))) + 1
    df = df.sort_values('age')
    df['age_name'] = ""
    for id1 in range(binning_num):
        mean = 0
        max_length = len(df)
        for id2 in range(binning_batch):
            id3 = min(max_length - 1, id1 * binning_batch + id2)
            mean += df['age'].iloc[id3]
        mean = mean / binning_batch
        for id2 in range(binning_batch):
            id3 = min(max_length - 1, id1 * binning_batch + id2)
            df['age'].iloc[id3] = mean
            df['age_name'].iloc[id3] = "age" + str(int(mean/ 10) * 10)
    
    # Convert male, female to 0, 1 (sex column)
    sex_convert_dict = {'male':1, 'female':0}
    for row in [df]:
        row['sex'] = row['sex'].map(sex_convert_dict)

    # Convert S, C, Q to 0, 1, 2 (embarked column)
    embarked_convert_dict = {'Q':2, 'C':1, 'S':0}
    for row in [df]:
        row['embarked'] = row['embarked'].map(embarked_convert_dict)

    # Create new column named age_class
    for row in [df]:
        row['age_class'] = row['age'] * row['pclass']

    #max-min normalization (fare column)
    df = min_max_normalization(df)
    return df

def main():
    df = read_csv('new_data/titanic.csv')
    first_look_data(df)
    df = preprocess_data(df, ['ticket','boat','body','home.dest','cabin'])
    write_csv('processed_data/titanic.csv',df)          

main()