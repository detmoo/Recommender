# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:15:03 2018

@author: Oliver
"""

from collections import defaultdict
import surprise as sp
import os
import random

def get_top_n(predictions, n=8):
    '''Return the top-n recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


file_path=os.path.expanduser('~\Documents\Python Scripts\Recommenders\stoch_data.txt')
reader=sp.Reader(line_format='user item rating',rating_scale=(-5,5),sep=",")
data=sp.Dataset.load_from_file(file_path,reader=reader)

raw_ratings = data.raw_ratings
random.shuffle(raw_ratings)
thresh=int(.9*len(raw_ratings))
A_ratings=raw_ratings[:thresh]
B_ratings=raw_ratings[thresh:]

data.raw_ratings=A_ratings

parameter_grid={'n_epochs':[5,50],'lr_all':[0.001,0.01,0.1],'reg_all':[0.05,0.3,0.6],'n_factors':[10,15]}
gs=sp.model_selection.search.GridSearchCV(sp.SVD,parameter_grid,measures=['rmse','mae'],cv=3,refit=False)
gs.fit(data)

#once the fit method has been run, best_estimator defines the algorithm with best settings
algo=gs.best_estimator['rmse']

#train the best outcome of Grid Search on the whole dataset
trainset = data.build_full_trainset()
algo.fit(trainset)

#use all the missing ratings as the test set
testset = trainset.build_anti_testset()
#testset=trainset.build_testset()
A_predictions = algo.test(testset)

data.raw_ratings=B_ratings
testset=data.construct_testset(B_ratings)
B_predictions=algo.test(testset)


antitest=trainset.build_anti_testset()
anti_predictions=algo.test(antitest)


top_n = get_top_n(predictions, n=8)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

print(gs.best_score['rmse'])

#gs.predict((1,2),(1,2,3),true_r,clip=False)
