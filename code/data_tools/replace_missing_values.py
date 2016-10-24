import numpy as np

def calculate_prob(tX, num_of_intervals):
    size = (np.max(tX) - np.min(tX))/num_of_intervals
    intervals_x = [0]*num_of_intervals
    intervals_p = [0]*num_of_intervals
    a = np.array(tX)
    for j in range(num_of_intervals):
        intervals_p[j] = ((np.min(tX)+j*size < a) & (a < np.min(tX)+(j+1)*size)).sum()
        intervals_x[j] = (np.min(tX)+j*size + np.min(tX)+(j+1)*size)/2
    return intervals_x, intervals_p

def make_distribution(x,prob,size):
    values = []
    for i in range(len(prob)):
         values = values + [x[i]]*int(size*prob[i]/sum(prob))
    if size>len(values): 
        max_prob = max(prob)
        max_index = prob.index(max_prob)
        values = values+[x[max_index]]*(size-len(values))
    return values

def replace_missing_values(tX):
	lista = []
	for i in range(tX.shape[1]):
	    if -999 in tX[:,i]: lista.append(i)
	for i in lista:
	    tX_tmp = list(filter((-999.0).__ne__, tX[:,i]))
	    x, prob = calculate_prob(tX_tmp,100)
	    #plt.plot(x,prob)
	    #plt.show()
	    means = make_distribution(x,prob,list(tX[:,i]).count(-999.0))
	    np.random.shuffle(means)
	    np.place(tX[:,i], tX[:,i] == -999.0, means)
	return tX