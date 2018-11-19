
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

#tf値を導出するオブジェクトの設定
tf_vectorizer = TfidfVectorizer(input="filename", use_idf=None, smooth_idf=None, max_df=1.0, min_df=1, sublinear_tf=False, norm=None)


"""１つ目の学習データとテストデータに関して"""
all_file_1=[]
all_label_1=[]

files_train_1_claim = ['data_set/data_1/train_wakati_toiawase_claim_meisi/claim/' + path for path in os.listdir('data_set/data_1/train_wakati_toiawase_claim_meisi/claim')]
files_train_1_toiawase = ['data_set/data_1/train_wakati_toiawase_claim_meisi/toiawase/' + path for path in os.listdir('data_set/data_1/train_wakati_toiawase_claim_meisi/toiawase')]
all_file_1.extend(files_train_1_claim)
all_file_1.extend(files_train_1_toiawase)

label_train_1_claim = np.loadtxt("data_set/data_1/train_wakati_toiawase_claim_meisi/train_claim_label.txt", delimiter="\n", dtype=float)
label_train_1_toiawase = np.loadtxt("data_set/data_1/train_wakati_toiawase_claim_meisi/train_toiawase_label.txt", delimiter="\n", dtype=float)
all_label_1.extend(label_train_1_claim)
all_label_1.extend(label_train_1_toiawase)
print(len(all_label_1))

files_test_1_claim = ['data_set/data_1/test_wakati_toiawase_claim_meisi/claim/' + path for path in os.listdir('data_set/data_1/test_wakati_toiawase_claim_meisi/claim')]
files_test_1_toiawase = ['data_set/data_1/test_wakati_toiawase_claim_meisi/toiawase/' + path for path in os.listdir('data_set/data_1/test_wakati_toiawase_claim_meisi/toiawase')]
all_file_1.extend(files_test_1_claim)
all_file_1.extend(files_test_1_toiawase)
print(len(all_file_1))

label_test_1_claim = np.loadtxt("data_set/data_1/test_wakati_toiawase_claim_meisi/test_claim_label.txt", delimiter="\n", dtype=float)
label_test_1_toiawase = np.loadtxt("data_set/data_1/test_wakati_toiawase_claim_meisi/test_toiawase_label.txt", delimiter="\n", dtype=float)
all_label_1.extend(label_test_1_claim)
all_label_1.extend(label_test_1_toiawase)
print(len(all_label_1))

tf_1 = tf_vectorizer.fit_transform(all_file_1)
tf_1 = tf_1.toarray()
print(tf_1.shape)

train_X_1 = tf_1[:7851,::]
test_X_1 = tf_1[7851:,::]
tf_1 = []

train_y_1 = all_label_1[:7851]
test_y_1 = all_label_1[7851:]
all_label_1 = []
print(train_y_1)
print(test_y_1)


# In[2]:


"""2つ目の学習データとテストデータに関して"""
all_file_2=[]
all_label_2=[]

files_train_2_claim = ['data_set/data_2/train_wakati_toiawase_claim_meisi/claim/' + path for path in os.listdir('data_set/data_2/train_wakati_toiawase_claim_meisi/claim')]
files_train_2_toiawase = ['data_set/data_2/train_wakati_toiawase_claim_meisi/toiawase/' + path for path in os.listdir('data_set/data_2/train_wakati_toiawase_claim_meisi/toiawase')]
all_file_2.extend(files_train_2_claim)
all_file_2.extend(files_train_2_toiawase)

label_train_2_claim = np.loadtxt("data_set/data_2/train_wakati_toiawase_claim_meisi/train_claim_label.txt", delimiter="\n", dtype=float)
label_train_2_toiawase = np.loadtxt("data_set/data_2/train_wakati_toiawase_claim_meisi/train_toiawase_label.txt", delimiter="\n", dtype=float)
all_label_2.extend(label_train_2_claim)
all_label_2.extend(label_train_2_toiawase)
# print(all_label_2)

files_test_2_claim = ['data_set/data_2/test_wakati_toiawase_claim_meisi/claim/' + path for path in os.listdir('data_set/data_2/test_wakati_toiawase_claim_meisi/claim')]
files_test_2_toiawase = ['data_set/data_2/test_wakati_toiawase_claim_meisi/toiawase/' + path for path in os.listdir('data_set/data_2/test_wakati_toiawase_claim_meisi/toiawase')]
all_file_2.extend(files_test_2_claim)
all_file_2.extend(files_test_2_toiawase)
print(len(all_file_2))

label_test_2_claim = np.loadtxt("data_set/data_2/test_wakati_toiawase_claim_meisi/test_claim_label.txt", delimiter="\n", dtype=float)
label_test_2_toiawase = np.loadtxt("data_set/data_2/test_wakati_toiawase_claim_meisi/test_toiawase_label.txt", delimiter="\n", dtype=float)
all_label_2.extend(label_test_2_claim)
all_label_2.extend(label_test_2_toiawase)
print(len(all_label_2))

tf_2 = tf_vectorizer.fit_transform(all_file_2)
tf_2 = tf_2.toarray()
print(tf_2.shape)

train_X_2 = tf_2[:7851,::]
test_X_2 = tf_2[7851:,::]
tf_2 = []

train_y_2 = all_label_2[:7851]
test_y_2= all_label_2[7851:]
all_label_2 = []
print(train_y_2)
print(test_y_2)



# In[3]:


"""3つ目の学習データとテストデータに関して"""
all_file_3=[]
all_label_3=[]

files_train_3_claim = ['data_set/data_3/train_wakati_toiawase_claim_meisi/claim/' + path for path in os.listdir('data_set/data_3/train_wakati_toiawase_claim_meisi/claim')]
files_train_3_toiawase = ['data_set/data_3/train_wakati_toiawase_claim_meisi/toiawase/' + path for path in os.listdir('data_set/data_3/train_wakati_toiawase_claim_meisi/toiawase')]
all_file_3.extend(files_train_3_claim)
all_file_3.extend(files_train_3_toiawase)

label_train_3_claim = np.loadtxt("data_set/data_3/train_wakati_toiawase_claim_meisi/train_claim_label.txt", delimiter="\n", dtype=float)
label_train_3_toiawase = np.loadtxt("data_set/data_3/train_wakati_toiawase_claim_meisi/train_toiawase_label.txt", delimiter="\n", dtype=float)
all_label_3.extend(label_train_3_claim)
all_label_3.extend(label_train_3_toiawase)
# print(all_label_3)

files_test_3_claim = ['data_set/data_3/test_wakati_toiawase_claim_meisi/claim/' + path for path in os.listdir('data_set/data_3/test_wakati_toiawase_claim_meisi/claim')]
files_test_3_toiawase = ['data_set/data_3/test_wakati_toiawase_claim_meisi/toiawase/' + path for path in os.listdir('data_set/data_3/test_wakati_toiawase_claim_meisi/toiawase')]
all_file_3.extend(files_test_3_claim)
all_file_3.extend(files_test_3_toiawase)
print(len(all_file_3))

label_test_3_claim = np.loadtxt("data_set/data_3/test_wakati_toiawase_claim_meisi/test_claim_label.txt", delimiter="\n", dtype=float)
label_test_3_toiawase = np.loadtxt("data_set/data_3/test_wakati_toiawase_claim_meisi/test_toiawase_label.txt", delimiter="\n", dtype=float)
all_label_3.extend(label_test_3_claim)
all_label_3.extend(label_test_3_toiawase)
print(len(all_label_3))

tf_3 = tf_vectorizer.fit_transform(all_file_3)
tf_3 = tf_3.toarray()
print(tf_3.shape)

train_X_3 = tf_3[:7851,::]
test_X_3 = tf_3[7851:,::]
tf_3 = []

train_y_3 = all_label_3[:7851]
test_y_3 = all_label_3[7851:]
all_label_3 = []
print(train_y_3)
print(test_y_3)


# In[4]:


"""4つ目の学習データとテストデータに関して"""
all_file_4=[]
all_label_4=[]

files_train_4_claim = ['data_set/data_4/train_wakati_toiawase_claim_meisi/claim/' + path for path in os.listdir('data_set/data_4/train_wakati_toiawase_claim_meisi/claim')]
files_train_4_toiawase = ['data_set/data_4/train_wakati_toiawase_claim_meisi/toiawase/' + path for path in os.listdir('data_set/data_4/train_wakati_toiawase_claim_meisi/toiawase')]
all_file_4.extend(files_train_4_claim)
all_file_4.extend(files_train_4_toiawase)

label_train_4_claim = np.loadtxt("data_set/data_4/train_wakati_toiawase_claim_meisi/train_claim_label.txt", delimiter="\n", dtype=float)
label_train_4_toiawase = np.loadtxt("data_set/data_4/train_wakati_toiawase_claim_meisi/train_toiawase_label.txt", delimiter="\n", dtype=float)
all_label_4.extend(label_train_4_claim)
all_label_4.extend(label_train_4_toiawase)
# print(all_label_4)

files_test_4_claim = ['data_set/data_4/test_wakati_toiawase_claim_meisi/claim/' + path for path in os.listdir('data_set/data_4/test_wakati_toiawase_claim_meisi/claim')]
files_test_4_toiawase = ['data_set/data_4/test_wakati_toiawase_claim_meisi/toiawase/' + path for path in os.listdir('data_set/data_4/test_wakati_toiawase_claim_meisi/toiawase')]
all_file_4.extend(files_test_4_claim)
all_file_4.extend(files_test_4_toiawase)
print(len(all_file_4))

label_test_4_claim = np.loadtxt("data_set/data_4/test_wakati_toiawase_claim_meisi/test_claim_label.txt", delimiter="\n", dtype=float)
label_test_4_toiawase = np.loadtxt("data_set/data_4/test_wakati_toiawase_claim_meisi/test_toiawase_label.txt", delimiter="\n", dtype=float)
all_label_4.extend(label_test_4_claim)
all_label_4.extend(label_test_4_toiawase)
print(len(all_label_4))

tf_4 = tf_vectorizer.fit_transform(all_file_4)
tf_4 = tf_4.toarray()
print(tf_4.shape)

train_X_4 = tf_4[:7851,::]
test_X_4 = tf_4[7851:,::]
tf_4 = []

train_y_4 = all_label_4[:7851]
test_y_4 = all_label_4[7851:]
all_label_4 = []
print(train_y_4)
print(test_y_4)


# In[ ]:


"""5つ目の学習データとテストデータに関して"""
all_file_5=[]
all_label_5=[]

files_train_5_claim = ['data_set/data_5/train_wakati_toiawase_claim_meisi/claim/' + path for path in os.listdir('data_set/data_5/train_wakati_toiawase_claim_meisi/claim')]
files_train_5_toiawase = ['data_set/data_5/train_wakati_toiawase_claim_meisi/toiawase/' + path for path in os.listdir('data_set/data_5/train_wakati_toiawase_claim_meisi/toiawase')]
all_file_5.extend(files_train_5_claim)
all_file_5.extend(files_train_5_toiawase)

label_train_5_claim = np.loadtxt("data_set/data_5/train_wakati_toiawase_claim_meisi/train_claim_label.txt", delimiter="\n", dtype=float)
label_train_5_toiawase = np.loadtxt("data_set/data_5/train_wakati_toiawase_claim_meisi/train_toiawase_label.txt", delimiter="\n", dtype=float)
all_label_5.extend(label_train_5_claim)
all_label_5.extend(label_train_5_toiawase)
print(len(all_label_5))

files_test_5_claim = ['data_set/data_5/test_wakati_toiawase_claim_meisi/claim/' + path for path in os.listdir('data_set/data_5/test_wakati_toiawase_claim_meisi/claim')]
files_test_5_toiawase = ['data_set/data_5/test_wakati_toiawase_claim_meisi/toiawase/' + path for path in os.listdir('data_set/data_5/test_wakati_toiawase_claim_meisi/toiawase')]
all_file_5.extend(files_test_5_claim)
all_file_5.extend(files_test_5_toiawase)
print(len(all_file_5))

label_test_5_claim = np.loadtxt("data_set/data_5/test_wakati_toiawase_claim_meisi/test_claim_label.txt", delimiter="\n", dtype=float)
label_test_5_toiawase = np.loadtxt("data_set/data_5/test_wakati_toiawase_claim_meisi/test_toiawase_label.txt", delimiter="\n", dtype=float)
all_label_5.extend(label_test_5_claim)
all_label_5.extend(label_test_5_toiawase)
print(len(all_label_5))

tf_5 = tf_vectorizer.fit_transform(all_file_5)
feature_names = tf_vectorizer.get_feature_names()
tf_5 = tf_5.toarray()
print(tf_5.shape)

train_X_5 = tf_5[:7848,::]
test_X_5 = tf_5[7848:,::]
tf_5 = []

train_y_5 = all_label_5[:7848]
test_y_5 = all_label_5[7848:]
all_label_5 = []
print(train_y_5)
print(test_y_5)
# print(feature_names)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []
all_confusion = []

#svmのインスタンス作成
pipe_svc = make_pipeline(SVC(random_state=1))

#グリッドサーチの模索範囲指定
param_range_gamma = [0.1, 0.01, 0.001, 0.0001]
param_range_C = [1.0, 10.0, 100.0, 1000.0]
param_grid = [{"svc__C":param_range_C, "svc__gamma":param_range_gamma, "svc__kernel":["rbf"]}]


#グリッドサーチ内の交差検証を5-層化交差検証にする
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

smote = SMOTE("minority", random_state=2)
enn = EditedNearestNeighbours("all",random_state=2)

train_X = [train_X_1, train_X_2, train_X_3, train_X_4, train_X_5]
train_y = [train_y_1, train_y_2, train_y_3, train_y_4, train_y_5]
test_X = [test_X_1, test_X_2, test_X_3, test_X_4, test_X_5]
test_y = [test_y_1, test_y_2, test_y_3, test_y_4, test_y_5]

for i in [0, 1, 2, 3, 4]:
    
    """データiに関して"""
    #SMOTEENNを行い新しい訓練データを作成
    train_X_smoteenn , train_y_smoteenn = SMOTEENN(random_state=2, smote=smote,enn=enn).fit_sample(train_X[i], train_y[i])
    print("smoteenn_train_%d(label=1): %s" %(i+1, train_X_smoteenn[ train_y_smoteenn == 1 ].shape[0]))
    print("smoteenn_train_%d(label=0): %s" %(i+1, train_X_smoteenn[ train_y_smoteenn == 0 ].shape[0]))


    #グリッドサーチのインスタンス生成（f1値で判断）
    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring="f1",
                      cv=kf,
                      n_jobs=-1)

    #学習データiでグリッドサーチ
    gs = gs.fit(train_X_smoteenn, train_y_smoteenn)
    print("data_%d::best_score: %s" %(i+1, gs.best_score_))
    print("data_%d::best_params: %s" %(i+1, gs.best_params_))
    #最適パラメータモデルの取得
    clf = gs.best_estimator_
    #最適パラメータで学習
    clf.fit(train_X_smoteenn, train_y_smoteenn)
    #テストデータiにおけるラベルの予測
    pre_label_test = clf.predict(test_X[i])

    #テストデータiにおけるそれぞれの評価値をリストに追加
    all_accuracy.append(accuracy_score(test_y[i], pre_label_test))
    all_precision.append(precision_score(test_y[i], pre_label_test))
    all_recall.append(recall_score(test_y[i], pre_label_test))
    all_f1.append(f1_score(test_y[i], pre_label_test))
    all_confusion.append(confusion_matrix(test_y[i], pre_label_test))





#5個のデータセットにおけるそれぞれの評価の平均
print("accuracy: %.3f +- %.3f" %(np.mean(all_accuracy), np.std(all_accuracy)))
print("precision: %.3f +- %.3f" %(np.mean(all_precision), np.std(all_precision)))
print("recall: %.3f +- %.3f" %(np.mean(all_recall), np.std(all_recall)))
print("f1: %.3f +- %.3f" %(np.mean(all_f1), np.std(all_f1)))

