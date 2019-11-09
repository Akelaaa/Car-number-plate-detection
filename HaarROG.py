#==========================================================================
#Уже подсчитанные характеристики для 1000 тестовых изображений
'''
haarPath = "cascade30h.xml"

recallList [0.8254189944134078, 0.8225352112676056, 0.8208744710860366, 0.8045325779036827, 0.7897727272727273,
            0.7795163584637269, 0.7694483734087695, 0.7460992907801418, 0.7126600284495022, 0.7357954545454546,
            0.701280227596017, 0.6529160739687055, 0.6584699453551912, 0.5549374130737135, 0.4767932489451477,
            0.569060773480663, 0.4197183098591549, 0.3437057991513437, 0.5805084745762712, 0.49291784702549574,
            0.4164305949008499, 0.48933143669985774, 0.40256045519203415, 0.33712660028449504]

precisionList = [0.17620751341681573, 0.23710921640276086, 0.29158316633266534, 0.43193916349809885,
                 0.5656154628687691, 0.6277205040091638, 0.5732349841938883, 0.6985391766268261, 0.75,
                 0.7204450625869263, 0.8029315960912052, 0.8375912408759124, 0.6703755215577191,
                 0.7388888888888889, 0.8033175355450237, 0.6809917355371901, 0.7286063569682152,
                 0.7546583850931677, 0.6650485436893204, 0.7265135699373695, 0.7405541561712846,
                 0.6653771760154739, 0.7546666666666667, 0.7719869706840391]
'''
'''
haarPath = "haarcascade_russian_plate_number.xml"

recallList = [0.8129395218002813, 0.8095909732016925, 0.8053596614950634, 0.7926657263751763, 
              0.7776203966005666, 0.7585227272727273, 0.7577903682719547, 0.7347517730496453, 
              0.7116477272727273, 0.7315340909090909, 0.6941678520625889, 0.6486486486486487, 
              0.6387362637362637, 0.5305555555555556, 0.47692307692307695, 0.561624649859944, 
              0.442090395480226, 0.3356940509915014, 0.5671852899575672, 0.4830028328611898, 
              0.42329545454545453, 0.5007112375533428, 0.42816500711237554, 0.3655761024182077]

precisionList = [0.15579514824797844, 0.21875, 0.26508820798514393, 0.43769470404984423, 0.5473579262213359, 
                 0.5986547085201793, 0.545361875637105, 0.6641025641025641, 0.7389380530973452, 0.6848404255319149, 
                 0.7845659163987139, 0.8201438848920863, 0.6258411843876177, 0.6970802919708029, 0.754424778761062, 
                 0.6334913112164297, 0.681917211328976, 0.6909620991253644, 0.6140888208269525, 0.6779324055666004, 
                 0.7376237623762376, 0.6530612244897959, 0.7270531400966184, 0.7558823529411764]
'''
#==========================================================================
#команда для установки opencv
#pip install opencv-python
#==========================================================================
#Для форматированя данных в программе использован json
'''
"1_11_2014_12_13_38_590.bmp38467": {
  "fileref": "",
  "size": 38467,
  "filename": "1_11_2014_12_13_38_590.bmp",
  "base64_img_data": "",
  "file_attributes": {},
  "regions": {
    "0": {
      "shape_attributes": {
        "name": "rect",
        "x": 186,
        "y": 203,
        "width": 75,
        "height": 21
      },
      "region_attributes": {}
    }
  }
}
'''
#==========================================================================

import math
import random
import PIL

import cv2
import numpy as np

#для хранения данных используем json, наиболее удобный формат
import json

#метрики
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score

#Построение графиков
import matplotlib.pyplot as plt

import sys
#==========================================================================
#вспомогательные процедуры

def changeOpCharac(detectListParam, markedDataParam, key, rateCh, lowerBorder, topBorder):
    #список для найденных уникальных номерных пластин
    #список для общего количества найденных номерных пластин
    findPlates = []
    findUnicPlates = []

    #делаем копии, чтобы работать с локальными параметрами
    detectList = detectListParam.copy()
    markedData = markedDataParam.copy()

    for i in range(len(detectList)):
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0

        mx1 = 0
        mx2 = 0
        my1 = 0
        my2 = 0

        #в данном случае очень важная нумерация с нуля
        for j in range(len(markedData[key]['regions'])):
            #формируем список для размеченных данных из json
            markedNumPlatesList = [
                         markedData[key]['regions'][str(j)]['shape_attributes']['x'],
                         markedData[key]['regions'][str(j)]['shape_attributes']['y'],
                         markedData[key]['regions'][str(j)]['shape_attributes']['x'] + markedData[key]['regions'][str(j)]['shape_attributes']['width'],
                         markedData[key]['regions'][str(j)]['shape_attributes']['y'] + markedData[key]['regions'][str(j)]['shape_attributes']['height']
                        ]

            #print('LL')
            #print(detectList)
            #print('MNPL')
            #print(markedNumPlatesList)

            #x1 < x2
            #упорядочили по x
            if detectList[i][0] < detectList[i][2]:
                x1 = detectList[i][0]
                x2 = detectList[i][2]
            else:
                x1 = detectList[i][2]
                x2 = detectList[i][0]

            #упорядочили по x
            if markedNumPlatesList[0] < markedNumPlatesList[2]:
                mx1 = markedNumPlatesList[0]
                mx2 = markedNumPlatesList[2]
            else:
                mx1 = markedNumPlatesList[2]
                mx2 = markedNumPlatesList[0]

            #y1 < y2
            #упорядочили по y
            if detectList[i][1] < detectList[i][3]:
                y1 = detectList[i][1]
                y2 = detectList[i][3]
            else:
                y1 = detectList[i][3]
                y2 = detectList[i][1]

            #упорядочили по x
            if markedNumPlatesList[1] < markedNumPlatesList[3]:
                my1 = markedNumPlatesList[1]
                my2 = markedNumPlatesList[3]
            else:
                my1 = markedNumPlatesList[3]
                my2 = markedNumPlatesList[1]

            #print(x1, x2, mx1, mx2, y1, y2, my1, my2)

            #находим пересечение отрезков
            xIntersection = max(0, min(x2, mx2) - max(x1, mx1))
            yIntersection = max(0, min(y2, my2) - max(y1, my1))
            #print('xIntersection ' + str(xIntersection))
            #print('yIntersection ' + str(yIntersection))

            #вычисляем площади
            detectNumArea = math.sqrt((x2 - x1)**2) * math.sqrt((y2 - y1)**2)
            detectNumAreaInter = xIntersection * yIntersection
            numArea = math.sqrt((markedNumPlatesList[0] - markedNumPlatesList[2])**2) * math.sqrt((markedNumPlatesList[1] - markedNumPlatesList[3])**2)

            #print('detectNumAreaInter / numArea: ' + str(detectNumAreaInter / numArea))
            #print('detectNumArea / numArea: ' + str(detectNumArea / numArea))

            if (detectNumAreaInter / numArea > lowerBorder) and (detectNumArea / numArea < topBorder):
                findPlates.append(str(j))

            if (detectNumAreaInter / numArea > lowerBorder) and (detectNumArea / numArea < topBorder) and (str(j) not in findUnicPlates):
                findUnicPlates.append(str(j))

    #print(findPlates, ' findPlates')
    #print(detectList, ' detectList')
    #print(findUnicPlates, ' findUnicPlates')
    #print(len(markedData[key]['regions']), ' len(markedData[key][\'regions\'])')

    rateCh.tp += len(findPlates)
    rateCh.fp += len(detectList) - len(findPlates)
    rateCh.fn += len(markedData[key]['regions']) - len(findUnicPlates)

    return rateCh

def drawMarkAndDetect(detectReg, markedRegions, itemKey, image):
    #делаем копии списков, чтобы работать с локальными параметрами
    localMarkedRegions = markedRegions.copy()
    localDetectReg = detectReg.copy()

    markedNumPlates = []
    for i in range(len(localMarkedRegions[itemKey]['regions'])):
        batch = []
        batch.append(localMarkedRegions[itemKey]['regions'][str(i)]['shape_attributes']['x'])
        batch.append(localMarkedRegions[itemKey]['regions'][str(i)]['shape_attributes']['y'])
        batch.append(localMarkedRegions[itemKey]['regions'][str(i)]['shape_attributes']['x'] + localMarkedRegions[itemKey]['regions'][str(i)]['shape_attributes']['width'])
        batch.append(localMarkedRegions[itemKey]['regions'][str(i)]['shape_attributes']['y'] + localMarkedRegions[itemKey]['regions'][str(i)]['shape_attributes']['height'])
        markedNumPlates.append(batch)

    '''
    for (x, y, x1, y1) in localDetectReg:
        cv2.rectangle(image, (x, y), (x1, y1), (random.randint(50, 250), 232, random.randint(50, 250)), -1)

    for (x, y, x1, y1) in markedNumPlates:
        cv2.rectangle(image, (x, y), (x1, y1), (0, 250, 250), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    '''

def makeDetectetData(image, numplateCascade, scaleF, minN):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #определяем, присутствует ли номерная пластина
    #характеристики нужно менять, при оценке каскада
    numPlates = numplateCascade.detectMultiScale(
        gray,
        scaleFactor = scaleF,
        minNeighbors = minN
    )

    localDetectData = []
    #для того, чтобы не появлялась ошибка, когда каскад не нашел номерных пластин
    #создаем список в удобном для нас формате, т.к. для numPlates характерна запись (x, y, w, h)
    if len(numPlates) == 0:
        localDetectData = []
    else:
        for i in range(len(numPlates)):
            bufData = [numPlates[i][0], numPlates[i][1], numPlates[i][0] + numPlates[i][2], numPlates[i][1] + numPlates[i][3]]
            localDetectData.append(bufData)

    return localDetectData


#==========================================================================

def mainProcedure(haarPath, dataPath, drivePath):
    print('CV2 version: ')
    print(cv2.__version__ + '\n')

    # -----------------------------------------------------------------------
    # загрузка данных

    # загружаем каскад
    try:
        numplateCascade = cv2.CascadeClassifier(haarPath)
    except:
        print('ERROR: cv2.CascadeClassifier(haarPath) \n')
        sys.exit()

    try:
        # загружаем файл с размеченной тестовой выборкой
        with open(dataPath, "r") as read_file:
            testData = json.load(read_file)
            # создаем список ключей в словаре
            keys = list(testData.keys())
    except:
        print('ERROR: dataPath \n')
        sys.exit()

    # -----------------------------------------------------------------------
    # тестирование

    class Characteristics:
        # положительные характеристики
        tp = 0
        tn = 0
        # отрицательные характеристики
        fp = 0
        fn = 0

    rateCh = Characteristics()

    # border для определения, правильно найден номер или не правильно
    # для площади пересечения номерных рамок
    lowerBorder = 0.7
    topBorder = 1.8

    # два списка для составления PR-кривой
    precisionList = []
    recallList = []

    # точки, для построения графика
    points = [1.01, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    numNeigh = [3, 5, 7]

    # points = [1.1]
    # numNeigh = [3]
    for pIter in range(len(points)):
        rateCh.tp = 0
        rateCh.tn = 0
        rateCh.fp = 0
        rateCh.fn = 0

        # проходимся по всем тестовым картинкам
        for numIter in range(len(numNeigh)):

            for i in range(len(keys) // 10):

                # для удобства сохраним ключ в отдельную переменную
                itemKey = keys[i]

                print('----------------------------------------------------')
                print(str(i) + '. ' + testData[itemKey]['filename'])

                # считываем изображение
                image = cv2.imread(drivePath + "\\" + testData[itemKey]['filename'])
                if image is None:
                    continue

                detectData = []
                detectData = makeDetectetData(image, numplateCascade, points[pIter], numNeigh[numIter])

                # numPlates это список из списков [[],[]]
                # передаем в функцию list с найденными номерами и размеченные данные
                rateCh = changeOpCharac(detectData, testData, itemKey, rateCh, lowerBorder, topBorder)

                print('   TP: ' + str(rateCh.tp) + ' TN: ' + str(rateCh.tn) + ' FP: ' + str(rateCh.fp) + ' FN: ' + str(
                    rateCh.fn))
                print('   Number of license plates: ', len(testData[itemKey]['regions']))
                print('   Found: {0} numplate!'.format(len(detectData)))
                print('----------------------------------------------------')

                drawMarkAndDetect(detectData, testData, itemKey, image)

            # считаем precision и recall (учитываем деление на ноль)
            try:
                recallList.append(rateCh.tp / (rateCh.tp + rateCh.fn))
                print('recall:' + str(rateCh.tp / (rateCh.tp + rateCh.fn)))
            except:
                recallList.append(0)

            try:
                precisionList.append(rateCh.tp / (rateCh.tp + rateCh.fp))
                print('precision:' + str(rateCh.tp / (rateCh.tp + rateCh.fp)))
            except:
                precisionList.append(0)

            # обнуляем
            rateCh.tp = 0
            rateCh.tn = 0
            rateCh.fp = 0
            rateCh.fn = 0

    # Следующим шагом построим PR-кривую для оценки точности полученной модели:
    print(recallList)
    print(precisionList)

    plt.plot(recallList, precisionList, color='r', label='Log Res')
    plt.title('precision-recall curve for Log Res')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.grid()
    plt.show()
    plt.gcf().clear()

#==========================================================================

haarPath = "cascade30h.xml"
dataPath = 'numplates_region_data.json'
drivePath = 'TestData'

mainProcedure(haarPath, dataPath, drivePath)
#==========================================================================

fig = plt.figure()
fig, ax = plt.subplots()

recallList = [0.8254189944134078, 0.8225352112676056, 0.8208744710860366, 0.8045325779036827, 0.7897727272727273,
            0.7795163584637269, 0.7694483734087695, 0.7460992907801418, 0.7126600284495022, 0.7357954545454546,
            0.701280227596017, 0.6529160739687055, 0.6584699453551912, 0.5549374130737135, 0.4767932489451477,
            0.569060773480663, 0.4197183098591549, 0.3437057991513437, 0.5805084745762712, 0.49291784702549574,
            0.4164305949008499, 0.48933143669985774, 0.40256045519203415, 0.33712660028449504]

precisionList = [0.17620751341681573, 0.23710921640276086, 0.29158316633266534, 0.43193916349809885,
                 0.5656154628687691, 0.6277205040091638, 0.5732349841938883, 0.6985391766268261, 0.75,
                 0.7204450625869263, 0.8029315960912052, 0.8375912408759124, 0.6703755215577191,
                 0.7388888888888889, 0.8033175355450237, 0.6809917355371901, 0.7286063569682152,
                 0.7546583850931677, 0.6650485436893204, 0.7265135699373695, 0.7405541561712846,
                 0.6653771760154739, 0.7546666666666667, 0.7719869706840391]

ax.plot(recallList, precisionList, 's', color='blue')

recallList = [0.8129395218002813, 0.8095909732016925, 0.8053596614950634, 0.7926657263751763,
              0.7776203966005666, 0.7585227272727273, 0.7577903682719547, 0.7347517730496453,
              0.7116477272727273, 0.7315340909090909, 0.6941678520625889, 0.6486486486486487,
              0.6387362637362637, 0.5305555555555556, 0.47692307692307695, 0.561624649859944,
              0.442090395480226, 0.3356940509915014, 0.5671852899575672, 0.4830028328611898,
              0.42329545454545453, 0.5007112375533428, 0.42816500711237554, 0.3655761024182077]

precisionList = [0.15579514824797844, 0.21875, 0.26508820798514393, 0.43769470404984423, 0.5473579262213359,
                 0.5986547085201793, 0.545361875637105, 0.6641025641025641, 0.7389380530973452, 0.6848404255319149,
                 0.7845659163987139, 0.8201438848920863, 0.6258411843876177, 0.6970802919708029, 0.754424778761062,
                 0.6334913112164297, 0.681917211328976, 0.6909620991253644, 0.6140888208269525, 0.6779324055666004,
                 0.7376237623762376, 0.6530612244897959, 0.7270531400966184, 0.7558823529411764]


ax.plot(recallList, precisionList, 's', color='red')

plt.title('precision-recall curve for Log Res')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('recall')
plt.ylabel('precision')
plt.grid()
plt.show()
plt.gcf().clear()

#==========================================================================