# Подключаем модуль
import os  # Operating System
from PIL import Image  # Python Imaging Library


# -----------------------------------------------------------------
# Блок с функциями

def Error(errorStr):
    print('ERROR: ', errorStr)
    file.close()
    raise SystemExit


# -----------------------------------------------------------------
# определяем переменные и открываем файл

print(
    '*** This program gets the width and height of all images in the directory and writes them to a separate file ***')
print('Where to create a data file?')
outputFileName = input()
print('Where are the analyzed files?')
directory = input()

try:
    outputFile = open(outputFileName, 'x')  # открыть для записи, если файла не существует
except:
    Error('Не удалось открыть файл, возможно данное имя уже занято')

# -----------------------------------------------------------------
# Получаем list с названиями всех jpg и bmp файлов

# Получаем список файлов в переменную files
try:
    inputFiles = os.listdir(directory)
except:
    Error('Не удалось получить список файлов в директории')

# Фильтруем список
imagesJPG = list(filter(lambda x: x.endswith('.jpg'), inputFiles))
imagesBMP = list(filter(lambda x: x.endswith('.bmp'), inputFiles))

images = []

images.extend(imagesJPG)
images.extend(imagesBMP)

# print(images)

# -----------------------------------------------------------------
# определяем режим работы программы

print("Enter the program mode (Good / Bad)")
operatingMode = input()

# -----------------------------------------------------------------
# Открываем наши изображения, получаем ширину и высоту и записываем в новый текстовый файл

if operatingMode == 'Good':
    for x in range(len(images)):
        # Good
        addStr = "Good"

        imageWriteName = ''
        imageWriteName = addStr + '\\' + images[x]

        try:
            im = Image.open(imageName)
        except:
            Error('Не удалось открыть изображение')
        width, height = im.size

        # print(width, height, ' ' , imageName)

        try:
            # Good
            outputFile.write(imageWriteName + ' 1' + ' 0' + ' 0' + ' ' + str(width) + ' ' + str(height) + '\n')
        except:
            Error('Не удалось записать данные в файл')

        # формат записи данных
        # {"content": "Изображение","annotation":[{"label":["number_plate"],"points":[{"x":0.0,"y":0.0},{"x":ширина,"y":высота}]}]}

elif operatingMode == 'Bad':
    for x in range(len(images)):
        # Bad
        addStr = 'Bad'

        imageName = ''
        imageName = directory + '\\' + images[x]

        try:
            im = Image.open(imageName)
        except:
            Error('Не удалось открыть изображение')
        width, height = im.size

        # print(width, height, ' ' , imageName)

        try:
            # Bad
            outputFile.write(imageName + '\n')
        except:
            Error('Не удалось записать данные в файл')

        # формат записи данных
        # {"content": "Изображение","annotation":[{"label":["number_plate"],"points":[{"x":0.0,"y":0.0},{"x":ширина,"y":высота}]}]}
else:
    print("Mistake in: program mode (Good / Bad)")

outputFile.close()