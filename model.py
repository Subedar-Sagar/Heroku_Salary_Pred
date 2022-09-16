import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv('C:\\Users\\Dell\\Downloads\\Cloudy_ML\\Deployment\\salary_predict_dataset.csv')

df['experience'].fillna(0,inplace=True)

df['test_score'].fillna(df['test_score'].mean(),inplace=True)

df['interview_score'].fillna(df['interview_score'].mean(),inplace=True)


def str_to_num(x):
    x_dict={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9
            ,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fifteen':15,'zero':0,0:0}
    return x_dict[x]

df['experience']=df['experience'].apply(lambda i : str_to_num(i))


X=df.drop('Salary',axis=1)
y=df['Salary']

lr=LinearRegression()
lr.fit(X,y)

pickle.dump(lr,open('pickle_file.pkl','wb'))

