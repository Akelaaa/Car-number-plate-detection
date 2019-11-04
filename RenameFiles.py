# Подключаем модуль 
import os  # Operating System


# -----------------------------------------------------------------
# Блок с функциями

def Error(errorStr):
    print('ERROR: ', errorStr)
    file.close()
    raise SystemExit


def GetExtension(fileName):
    extStr = ''
    i = len(fileName) - 1
    while i > 0:
        if fileName[i] == '.':
            break
        i -= 1
    while i < len(fileName):
        extStr += fileName[i]
        i += 1
    if extStr == fileName:
        return ''
    else:
        return extStr


# -----------------------------------------------------------------

print(
    '*** This program renames all files in a folder, replacing the name with a digit without changing the file extension ***')
print('Where are the files located?')
directory = input()

# Получаем список файлов в переменную files 
try:
    inputFiles = os.listdir(directory)
except:
    Error('Не удалось получить список файлов в директории')

#
# for i in range(len(inputFiles)):
#    ext = GetExtension(inputFiles[i])
#    print(ext + ": EXT")
#    print(inputFiles[i])

print('Are you sure you want to rename all the files in the directory?')
print('Enter yes / no')
decision = input()

if decision == 'yes':
    for i in range(len(inputFiles)):
        renameFlag = True

        old_file = os.path.join(directory, inputFiles[i])
        ext = GetExtension(inputFiles[i])

        # подбираем имя файлу
        # переименовываем от 0..n
        digit = 1
        while str(digit) + ext in inputFiles:
            if str(digit) + ext == inputFiles[i]:
                renameFlag = False
            digit += 1

            # если у файла уже название формата 0..n
        # и файл находится на n-ом месте
        # мы его не переименовываем
        if renameFlag:
            new_file = os.path.join(directory, str(digit) + ext)
            inputFiles[i] = str(digit) + ext
            os.rename(old_file, new_file)

elif decision == 'no':
    print('Program shutdown...')
else:
    print('Incorrect input')