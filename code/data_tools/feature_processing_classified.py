 mport numpy as np

#returns an array of the sums of correlations for each feature
def correlate_all(class1,class2,class3):
	correlations1 = []
	correlations2 = []
	correlations3 = []
	for i in range(0, len(class1[0])):
		correlations1.append(0)
	for i in range(0, len(class1)[0]-1):
		feature = class1[:,i]
		for j in range(i+1, len(class1[0])):
			c = np.correlate(feature, class1[:,j])
			correlations1[i] += c 
			correlations1[j] += c
	

        for i in range(0, len(class2[0])):
		correlations2.append(0)
	for i in range(0, len(class2[0])-1):
		feature = class2[:,i]
		for j in range(i+1, len(class2[0]):)
			c = np.correlate(feature, class2[:,j])
			correlations2[i] += c 
			correlations2[j] += c
	

        for i in range(0, len(class3[0])):
		correlations3.append(0)
	for i in range(0, len(class3[0])-1):
		feature = class3[:,i]
		for j in range(i+1, len(class3)[0]):
			c = np.correlate(feature, class3[:,j])
			correlations3[i] += c 
			correlations3[j] += c
	return correlations1,correlations2,correlations3
                               
#returns the data without useless feature that are dropped
def feature_process(class1,class2,class3):
	correlation_threshold = 40 # we define an arbitray threshold under which correlation is acceptable
	[corr1, corr2, corr3] = correlate_all(class1,class2,class3)
	processed_x1 = class1
        processed_x2 = class2
        processed_x3 = class3
	for i in range(len(class1)1, -1, -1):
		if(corr1[i]>correlation_threshold):
			processed_x1 = np.delete(processed_x1,i,1)

        for i in range(len(class2)-1, -1, -1):
		if(corr2[i]>correlation_threshold):
			processed_x2 = np.delete(processed_x2,i,1)

	for i in range(len(class3)-1, -1, -1):
		if(corr3[i]>correlation_threshold):
			processed_x3 = np.delete(processed_x3,i,1)
	

	return processed_x1, processed_x2, processed_x3
