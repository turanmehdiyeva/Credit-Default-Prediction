#importing libraries and the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') 
%matplotlib inline

loan = pd.read_csv('credit_card_default_TRAIN.csv', header=1)
loan.head()


loan.isna().sum()

loan.columns

gender_values = {
    1:'male',
    2:'female'
}

loan.SEX = loan.SEX.map(gender_values)

loan.loc[(loan['EDUCATION']==5) | (loan['EDUCATION']==6) | (loan['EDUCATION']==0), 'EDUCATION'] = 4

education_values = {
    1:'graduate_shool',
    2:'university',
    3:'high_school',
    4:'others'
}

loan.EDUCATION = loan.EDUCATION.map(education_values)

loan.loc[loan['MARRIAGE']==0, 'MARRIAGE'] = 3

marital_status = {
    1:'married',
    2:'single',
    3:'others'
}

loan.MARRIAGE = loan.MARRIAGE.map(marital_status)

loan.head()

loan.columns = loan.columns.str.lower()
loan.rename(columns={'default payment next month':'defaulted'}, inplace=True)
loan.head()
