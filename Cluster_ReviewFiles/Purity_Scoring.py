import numpy as np
from collections import defaultdict
import collections
import csv


def purity_score(input_csv, N):
    Max_Scores = {}

    i = 0
    for row in input_csv:
        scores = [float(score) for score in row]
        Max_Scores[i] = max(scores[:])
        i += 1

    purity_sum = 0
    #print Max_Scores
    for val in Max_Scores.values():
        purity_sum += val

    purity_score = float(purity_sum) / N
    return purity_score


Gaussian_Scores_csv = csv.reader(open('purity_gaussian.csv','rb'),delimiter=',')
#KMeans_Scores_csv   = csv.reader(open('purity_KMeans.csv','rb'),delimiter=',')

N = 377

print "\n"
print "Purity score using Gaussian Clustering: ", purity_score(Gaussian_Scores_csv,N)
#print "Purity score using K-Means Clustering: ", purity_score(KMeans_Scores_csv,N)
print "\n"
