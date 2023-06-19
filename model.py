# import the library
import pandas as pd
import numpy as np
#pip install words2num
from words2num import w2n
from sklearn.linear_model import LinearRegression
import pickle


# Preprocess the data
df = pd.read_excel('D:/Surendra/Project-Portfolio/full_project/Book1.xlsx')

percentage_of_null_values = ((df.isnull().sum())*100/len(df)).sort_values(ascending = False)
test_mean = df['test_score'].mean()
int_score_mean = df['interview_score'].mean()
df['interview_score'] = df['interview_score'].fillna(int_score_mean)
df['test_score'] = df['test_score'].fillna(test_mean)

def convert_num(word):
   wo_2_nu = {'zero': 1,
              'one': 2, 'two': 2, 'three': 3, 'four': 4, 'five':5, 'six': 6, 'seven': 7, 'eight': 8,
           'nine': 9, 'ten': 10}
   return wo_2_nu[word]

df['experience'] = df['experience'].apply(lambda x: convert_num(x))

X = df.iloc[:, :3]
y = df.iloc[:, -1]

## train the model and store the pickle model
regressor = LinearRegression()
regressor.fit(X,y)
pickle.dump(regressor, open('model.pkl', 'wb'))

# if __name__ == '__main__':
   # app.run(debug = True)

