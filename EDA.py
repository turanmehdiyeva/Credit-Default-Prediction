#### 1. Is the % of defaulters significantly different between male & female ?
loan.sex.unique()

global_mean = round(loan['defaulted'].mean(),3)
global_mean

def default_stat(column):
    data_group = loan.groupby(column)['defaulted'].agg(['mean'])
    data_group['diff'] = data_group['mean'] - global_mean
    data_group['risk'] = data_group['mean']/global_mean
    return data_group

default_stat('sex')

The default rate for males is 24.6%

The default rate for females is 21.3%

sex_df = default_stat('sex')
values = list(sex_df['mean'])
labels = list(sex_df.index)
plt.title(label='% of Defaulter by Sex', fontsize=15)
sns.barplot(labels,values)
plt.show()

Default of male and female costumers are not very different from each other and global default rate.

#### 2. How does Marital Status effect the proportion of defaulters ?

loan['marriage'].value_counts()

default_stat('marriage')

The default rate for married borrowers is 24.3%

The default rate for single borrowers is 21.3%

The default rate for others is 23.0%

marriage_df = default_stat('marriage')
values = list(marriage_df['mean'])
labels = list(marriage_df.index)
plt.title(label='% of Defaulter by Marital Status', fontsize=15)
sns.barplot(labels,values)
plt.show()

#### 3. Does the Level of Education play a role in the % of defaulters ?

loan['education'].value_counts()

default_stat('education')

The default rate for graduate schoolers is 19.7%

The default rate for university students is 24.099999999999998%

The default rate for high schoolers is 25.900000000000002%

The default rate for others is 6.6000000000000005%

education_df = default_stat('education')
values = list(education_df['mean'])
labels = list(education_df.index)
plt.title(label='% of Defaulter by Education level', fontsize=15)
sns.barplot(labels,values)
plt.show()

The default of high school graduate borrowers are higher while default of graduaters of high institutions are lower.

#### 4. Which age group constitutes for higher proportion of defaulters ?

loan['age'].unique()

default_young = round(loan[loan['age']<30]['defaulted'].mean(),3)
print(f'The default rate for people aged younger than 30 is {default_young*100}%')

default_adult = round(loan[(loan['age']>=30)&(loan['age']<50)]['defaulted'].mean(),3)
print(f'The default rate for middle aged people is {default_adult*100}%')

default_old = round(loan[loan['age']>=50]['defaulted'].mean(),3)
print(f'The default rate for people older than 50 is {default_old*100}%')

defaulters = loan[loan["defaulted"] == 1]
non_defaulters = loan[loan["defaulted"] == 0]
defaulters["Defaulter"] = defaulters["age"]
non_defaulters["Non Defaulter"] = non_defaulters["age"]
f, ax = plt.subplots(figsize=(12, 6))
ax = sns.kdeplot(defaulters["Defaulter"], shade=True, color="r", label='Defaulter')
ax = sns.kdeplot(non_defaulters["Non Defaulter"], shade=True, color="g", label='NON Defaulter')
plt.legend()
plt.show()

The default rate for middle aged people is less than other people.

#### 5. Is the number of defaulters correlated with credit limit ?

loan[['limit_bal']].corrwith(loan['defaulted']).to_frame('Correlation')

#The credit limit and default of borrowers are negatively correlated. People with lower credit balance tend to default more

#### 6. Is there a pattern in past repayment statuses which can help predict probability of a defaulter ?

lst = ['pay_0','pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']

lst2 = []

for i in lst:
    keys = list(loan.groupby(i).mean()[['defaulted']].index)
    values =  list(loan.groupby(i).mean()['defaulted'])
    d = dict(zip(keys, values))
    lst2.append(d)

rep_df = pd.DataFrame(lst2)
rep_df

for i in range(6):
    x = list(rep_df.iloc[i,:])
    plt.title('Proportion of Defaulters Versus Repayment Status', fontsize=17, y = 1.05) 
    plt.ylabel('Proportion of Defaulters', fontsize=14)
    plt.xlabel('Repayment Status', fontsize=14)
    plt.scatter(rep_df.columns,x, label="pay {}".format(i))
    plt.legend()

#### 7. Does the history of credit card bill amount has a correlation with the % of defaulters ?

ls = ['bill_amt1', 'bill_amt2',
       'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6']

loan[ls].corrwith(loan['defaulted']).to_frame('Correlations')

#### **Feature engeenering**

bills_amt  = loan.iloc[:,12:18]

pays_amt = loan.iloc[:,18:-1]

for i in range(6):
    name = 'bill_pay_ratio0{}'.format(i+1)
    loan[name] = (bills_amt.iloc[:,i] - pays_amt.iloc[:,i]) / loan['limit_bal']

loan.head()

loan.columns

categorical = ['sex', 'education', 'marriage']

#Calculating dependency score
from sklearn.metrics import mutual_info_score

def calculate_mi(series):
    return mutual_info_score(series, loan['defaulted'])

data_mi = loan[categorical].apply(calculate_mi)
data_mi = data_mi.sort_values(ascending=False).to_frame(name='Mi')
data_mi

#Dependency between target variable and categorical variables are for Education is strong, but it's not quite good for gender and marriage.
