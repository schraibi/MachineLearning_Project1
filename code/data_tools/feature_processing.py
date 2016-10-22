import numpy as np

#returns an array of the sums of correlations for each feature
def correlate_all(tX):
	correlations = []
	for i in range(0, len(tX[0])):
		correlations.append(0)
	for i in range(0, len(tX[0])-1):
		feature = tX[:,i]
		for j in range(i+1, len(tX[0])):
			c = np.correlate(feature, tX[:,j])
			correlations[i] += c 
			correlations[j] += c
	return correlations

#returns the data without useless feature that are dropped
def feature_process(tX):
	correlation_threshold = 40 # we define an arbitray threshold under which correlation is acceptable
	correlations = correlate_all(tX)
	processed_x = tX
	for i in range(len(tX[0])-1, -1, -1):
		if(correlations[i]>correlation_threshold):
			processed_x = np.delete(processed_x,i,1)
	return processed_x