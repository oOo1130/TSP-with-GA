def getPartialOrder():
    sigma1 = [1,2,3,4,5,6]
    sigma2 = [2,3,1,5,4,6]
    DValue = []

    for idx in range(len(sigma1)):
        for jdx in range(idx, len(sigma1)):
            ivalue = sigma1[idx]
            jvalue = sigma1[jdx]
            if(sigma2.index(ivalue) <= sigma2.index(jvalue)):
                DValue.append((sigma1[idx], sigma1[jdx]))
    print(DValue)

getPartialOrder()