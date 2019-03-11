from adaboost import *
#测试文件
dataArr,labelArr=loadSimpData()
classifierArr=adaBoostTrains(dataArr,labelArr,30)
adaClassify([0,0],classifierArr)