from numpy import *


#通过阈值比较分类函数
def stumpClassify(dataMatrix,dimen,threshVal,threshInseg):
    retArray=ones((shape(dataMatrix)[0],1))
    if threshInseg=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1
    else:
        retArray[dataMatrix[:,dimen]>=threshVal]=-1
    return retArray

#该函数遍历stumpClassify()函数的所有可能的输入值,返回决策树字典、错误率、类别
def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr)
    labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps=10.0
    bestStump={}#用于存储给定权重向量D时所得的最佳单层决策树
    bestClassEst=mat(zeros((m,1)))
    minError=inf
    for i in range(n): #第一层循环在数据的所有的特征上遍历
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps #循环步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=mat(ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError=D.T*errArr
                print("split: dim "+str(i)+","+"thresh "+str(threshVal)+" thresh ineqal: "+str(inequal)+",the weighted error is "+str(weightedError))
                if weightedError<minError:
                    minError=weightedError
                    bestClassEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return  bestStump,minError,bestClassEst
