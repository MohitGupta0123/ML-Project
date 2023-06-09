# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import joblib

import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline


df = pd.read_csv("google_review_ratings.csv")


column_names = ['User_id', 'Churches', 'Resorts', 'Beaches', 'Parks', 'Theatres', 'Museums', 'Malls', 'Zoo', 'Restaurants', 'Pubs_bars', 'Local_services', 'Burger_pizza_shops', 'Hotels_other_lodgings', 'Juice_bars', 'Art_galleries', 'Dance_clubs', 'Swimming_pools', 'Gyms', 'Bakeries', 'Beauty_spas', 'Cafes', 'View_points', 'Monuments', 'Gardens', 'Unnamed: 25']
df.columns = column_names


df.drop(columns = ['User_id', 'Unnamed: 25'], axis = 1, inplace = True)

df['Local_services'][df['Local_services'] == '2\t2.']

df.loc[2712, 'Local_services'] = np.NaN

impute = SimpleImputer(missing_values= np.nan, strategy= 'mean')
df = impute.fit_transform(df)

u_column_names =['Churches', 'Resorts', 'Beaches', 'Parks', 'Theatres', 'Museums', 'Malls', 'Zoo', 'Restaurants', 'Pubs_bars', 'Local_services', 'Burger_pizza_shops', 'Hotels_other_lodgings', 'Juice_bars', 'Art_galleries', 'Dance_clubs', 'Swimming_pools', 'Gyms', 'Bakeries', 'Beauty_spas', 'Cafes', 'View_points', 'Monuments', 'Gardens']

idf = pd.DataFrame(df, columns = u_column_names)

idf.loc[2712, 'Local_services']

standard_df = idf.copy()
cols_S = list(standard_df.columns)

sc = StandardScaler()
standard_df = sc.fit_transform(standard_df)
std_df = pd.DataFrame(standard_df, columns = cols_S)      # converting standard_df in to dataframe as it is ndarray after scaling



kmean= KMeans(5)
kmean.fit(standard_df)
labels=kmean.labels_ + 1

pred_Kmeans=pd.concat([standard_df, pd.DataFrame({"CLUSTER":labels})], axis=1)
pred_Kmeans["CLUSTER"] = pred_Kmeans["CLUSTER"]

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


X = std_df

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)

x = pred_Kmeans.drop("CLUSTER", axis = 1)
y = pred_Kmeans["CLUSTER"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state= 0)

clf1 = LogisticRegression()
clf2 = SVC()
clf3 = KNeighborsClassifier()
clf4 = GaussianNB()
clf5 = DecisionTreeClassifier()

clf = [clf1, clf2, clf3, clf4, clf5]
clf_name = ['LR', 'SVC', 'KNN', 'GNB', 'DT']
acc = {}

for model, model_name in zip(clf, clf_name):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    acc[model_name] = accuracy_score(y_test, pred) * 100

print("ACCURACY SCORES")
for i, j in acc.items():
    print(i, ':-', j, '%')

print(acc.keys())
print(acc.values())

parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]},
              {'tol': [0.0009, 0.001, 0.002, 0.003]}]

grid_search = GridSearchCV(estimator = clf1,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose = 3,
                           return_train_score=True)


grid_search.fit(x_train, y_train)  

best_model = grid_search.best_estimator_         # why parameters are not visible_ in case of logistic regression
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

print(best_model, best_parameters, best_score * 100)

lr_tuned = LogisticRegression(penalty = 'l2', C = 100, tol = 0.001)              # parameters that are seen manually in the grid search cv
lr_tuned.fit(x_train, y_train)

print("INFO - Model has trained")

test_acc = lr_tuned.score(x_test, y_test)

file_name = 'travel_review_rating_model.pkl'
joblib.dump(lr_tuned, open(file_name, 'wb'))