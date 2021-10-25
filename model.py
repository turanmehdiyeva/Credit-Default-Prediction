#Splitting data
from sklearn.model_selection import train_test_split

train_full, test_X = train_test_split(loan, test_size=0.2, random_state=1)  

train_X, val_X = train_test_split(train_full, test_size=0.2, random_state=1)

train_y = train_X['defaulted'].values.reshape(-1,1)
val_y = val_X['defaulted'].values.reshape(-1,1)
test_y = test_X['defaulted'].values.reshape(-1,1)

del train_X['defaulted']
del val_X['defaulted']
del test_X['defaulted']

#Encoding data
train_dict = train_X.to_dict(orient='rows')

train_dict[0]

from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)
len(X_train[0])

val_dict = val_X.to_dict(orient='rows')

X_val = dv.transform(val_dict)

len(X_val[0])

#### Logistic Regression

#Creating LogisticRegression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', random_state=1)
model.fit(X_train, train_y)

preds = model.predict_proba(X_val)[:,1]

y_pred = (preds>0.49).reshape(-1,1)

from sklearn.metrics import accuracy_score

accuracy_score(val_y, y_pred)

from sklearn.metrics import roc_auc_score

roc_auc_score(val_y, y_pred)

#### Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, train_y)

y_pred = dt.predict_proba(X_train)[:,1]
roc_auc_score(train_y, y_pred)

y_pred = dt.predict_proba(X_val)[:,1]
roc_auc_score(val_y, y_pred)

overfitting

dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X_train, train_y)

from sklearn.tree import export_text

tree_text = export_text(dt, feature_names=dv.feature_names_)
print(tree_text)

y_pred = dt.predict_proba(X_train)[:,1]
roc_auc_score(train_y, y_pred)

y_pred = dt.predict_proba(X_val)[:,1]
roc_auc_score(val_y, y_pred)

for depth in [1,2,3,4,5,6,10,15,20,None]:
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, train_y)
    y_pred = dt.predict_proba(X_val)[:,1]
    auc = roc_auc_score(val_y, y_pred)
    print(f'{depth}-->{auc}')

for m in [4,5,6]:
    print(f'depth: {m}')
    
    for s in [1,5,10,15,20,50,100,200]:
        dt=DecisionTreeClassifier(max_depth=m, min_samples_leaf=s)
        dt.fit(X_train, train_y)
        y_pred = dt.predict_proba(X_val)[:,1]
        auc = roc_auc_score(val_y, y_pred)
        print(f'{s}-->{auc}')
        
    print()

dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=200)
dt.fit(X_train, train_y)

y_pred = dt.predict_proba(X_val)[:,1]
roc_auc_score(val_y, y_pred)

#### Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, train_y)

y_pred = rf.predict_proba(X_val)[:,1]
roc_auc_score(val_y, y_pred)

rf = RandomForestClassifier(n_estimators=10, random_state=3)
rf.fit(X_train, train_y)

y_pred = rf.predict_proba(X_val)[:,1]
roc_auc_score(val_y, y_pred)

aucs = []

for i in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=i, random_state=3)
    rf.fit(X_train, train_y)
    
    y_pred = rf.predict_proba(X_val)[:,1]
    auc = roc_auc_score(val_y, y_pred)
    print(f'{i}-->{auc}')
    
    aucs.append(auc)

plt.plot(range(10,201,10), aucs)
plt.show()

all_aucs = {}

for depth in [5,10,20]:
    print(f'depth: {depth}')
    aucs = []
    
    for i in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=i, max_depth=depth, random_state=1)
        rf.fit(X_train, train_y)
        y_pred = rf.predict_proba(X_val)[:,1]
        auc = roc_auc_score(val_y, y_pred)
        print(f'{i}-->{auc}')
        aucs.append(auc)
        
    all_aucs[depth] = aucs
    print()

num_trees = list(range(10,201,10))
plt.plot(num_trees, all_aucs[5], label='depth=5')
plt.plot(num_trees, all_aucs[10], label='depth=10')
plt.plot(num_trees, all_aucs[20], label='depth=20')
plt.legend()
plt.show()

all_aucs = {}

for m in [3,5,10]:
    print(f'min_samples_leaf: {m}')
    aucs = []
    
    for i in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=i, max_depth=5, min_samples_leaf=m, random_state=1)
        rf.fit(X_train, train_y)
        y_pred = rf.predict_proba(X_val)[:,1]
        auc = roc_auc_score(val_y, y_pred)
        print(f'{i}-->{auc}')
        aucs.append(auc)
        
    all_aucs[m] = aucs
    print()

num_trees = list(range(10,201,10))
plt.plot(num_trees, all_aucs[3], label='min_samples_leaf=3')
plt.plot(num_trees, all_aucs[5], label='min_samples_leaf=5')
plt.plot(num_trees, all_aucs[10], label='min_samples_leaf=10')
plt.legend()
plt.show()

rf = RandomForestClassifier(n_estimators=180, max_depth=5, min_samples_leaf=10, random_state=1)
rf.fit(X_train, train_y)
y_pred = rf.predict_proba(X_val)[:,1]
roc_auc_score(val_y, y_pred)

#### Testing the final model

train_y = (train_full['defaulted']==1).values.reshape(-1,1)

del train_full['defaulted']

dict_train = train_full.fillna(0).to_dict(orient='records')
dict_test = test_X.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(dict_train)
X_test = dv.transform(dict_test)

print(f'{len(X_train[0])}\n{len(X_test[0])}')

rf = RandomForestClassifier(n_estimators=180, max_depth=5, min_samples_leaf=10, random_state=1)
rf.fit(X_train, train_y)
y_pred = rf.predict_proba(X_test)[:,1]
roc_auc_score(test_y, y_pred)
