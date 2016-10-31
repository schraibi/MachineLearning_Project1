def classify_data_before_processing(tX):
    class1 = []
    class2 = []
    class3 = []

    for i in range(len(tX)):
        if tX[i,22] == 0:
            class1.append(tX[i,:])
        elif tX[i,22] == 1:
            class2.append(tX[i,:])
        else:
            class3.append(tX[i,:])

return class1, class2, class3