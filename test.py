
'''
    Написать следующие функции:
        - для определения размера текстового файла, который будет встраиваться
        - функция для генерации случайных последовательностей печатных аски символов 
        - 

'''
#from pynput.mouse import Button, Controller
import base64
import struct
import string
import secrets 
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing, neighbors, ensemble, neural_network, naive_bayes, model_selection
import random
from sklearn.metrics import accuracy_score
import pickle
from itertools import chain, combinations

#mouse = Controller()


def generate_container():
    pass

def dec_to_hex(num):
    hex_list = "0123456789abcdef"
    hex = hex_list[num//16] + hex_list[num%16] 
    return hex 

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_pixel_data_end(filename):
    with open(filename, "rb") as f:
        # Читаем BITMAPFILEHEADER (первые 14 байт)
        file_header = f.read(14)

        # Проверяем, что это BMP-файл (первые 2 байта должны быть "BM")
        if file_header[:2] != b"BM":
            raise ValueError("Это не BMP-файл.")

        # Извлекаем смещение до данных пикселей (bfOffBits)
        pixel_offset = struct.unpack("<I", file_header[10:14])[0]

        # Читаем BITMAPINFOHEADER (следующие 40 байт)
        info_header = f.read(40)

        # Извлекаем ширину, высоту и глубину цвета
        width = struct.unpack("<i", info_header[4:8])[0]  # Ширина (4 байта, смещение 4)
        height = struct.unpack("<i", info_header[8:12])[0]  # Высота (4 байта, смещение 8)
        bits_per_pixel = struct.unpack("<H", info_header[14:16])[0]  # Глубина цвета (2 байта, смещение 14)

        # Вычисляем размер одной строки с учетом выравнивания
        bytes_per_pixel = bits_per_pixel // 8
        row_size = width * bytes_per_pixel
        padding = (4 - (row_size % 4)) % 4  # Выравнивание до 4 байт
        row_size_with_padding = row_size + padding

        # Вычисляем общий размер данных пикселей
        pixel_data_size = row_size_with_padding * height

        # Вычисляем конец данных пикселей
        pixel_data_end = pixel_offset + pixel_data_size 

        return pixel_data_end

# Пример использования


 

def get_lsb_from_file(filepath):
    with open(filepath, 'rb') as file:

        binValue = file.read()
        hex_dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10,'b':11,'c':12,'d':13,'e':14,'f':15}
        encoded_file = base64.b64encode(binValue)
        image_hex = base64.b64decode(encoded_file).hex()
        #print(type(image_hex))
        pixel_data_start = 2* (struct.unpack("<I", binValue[10:14])[0] - 1)

        lsbit_list = []
        pixel_data_end = get_pixel_data_end(filepath)*2

        #print('start', pixel_data_start)
        #print(pixel_data_end)
        #print(len(image_hex))
        amount = len(range(pixel_data_start + 1, pixel_data_end, 2))
        #print(amount)
        for i in range(pixel_data_start + 1, pixel_data_end, 2):
            lsbit_list.append(hex_dict[image_hex[i]]%2)
        #print(len(lsbit_list))
        data_len = len(lsbit_list)//(2*8*3)

        #print("data length:", data_len)
        return lsbit_list
    

def get_features_vector_from_lsb(lsb_list, attributes=('all')):
    
    e_lsb = 0
    d_lsb = 0 
    max_0_len = 0 
    max_1_len = 0
    count_0 = 0
    count_1 = 0
    diff_list = []
    e_diff = 0
    d_diff = 0 
    diff_count_0 = 0
    diff_count_1 = 0
    cur_0_len = 0
    cur_1_len = 0
    zeros_dict = {}

    for i in range(len(lsb_list)):
        if i != 0:
            cur_diff = abs(lsb_list[i] - lsb_list[i-1])
            if cur_diff == 0:
                diff_count_0 += 1
            else:
                diff_count_1 += 1
            diff_list.append(cur_diff)
        
        if lsb_list[i] == 0:
            count_0 += 1
            cur_0_len += 1
            cur_1_len = 0
            if cur_0_len > max_0_len:
                max_0_len = cur_0_len
        elif lsb_list[i] == 1:
            count_1 += 1
            cur_0_len = 0 
            cur_1_len += 1
            if cur_1_len > max_1_len:
                max_1_len = cur_1_len
    #print(count_0)
    #print(count_1)
    #print()
    #print(count_0, count_1, max_0_len,max_1_len,diff_count_0,diff_count_1)
    p0_lsb = float(count_0)/len(lsb_list)
    p1_lsb = float(count_1)/len(lsb_list)
    p0_diff = float(diff_count_0)/(len(lsb_list)-1)
    p1_diff = float(diff_count_1)/(len(lsb_list)-1)
    e_lsb = round(p1_lsb, 3)
    e_diff = round(p1_diff, 3)
    d_lsb = round(p0_lsb * (e_lsb)**2 + p1_lsb*(1-e_lsb)**2, 3)
    d_diff = round(p0_diff * (e_diff)**2 + p1_diff*(1-e_diff)**2, 3)
    zeros = (''.join(map(str, lsb_list))).split('1')
    zeros_frequence = {}
    zeros_groups = 0 
    e_zero_length = 0

    for z in zeros:
        if z != '':
            zeros_frequence[len(z)] = zeros_frequence.get(len(z),0)+1
            zeros_groups += 1
    for k in zeros_frequence:
        e_zero_length += k*float(zeros_frequence[k])/float(zeros_groups)
    e_zero_length = round(e_zero_length, 3)
    vector = []
    attr_dict = {
        'e_lsb':e_lsb,
        'd_lsb':d_lsb,
        'max_0_len':max_0_len,
        'max_1_len':max_1_len,
        'e_diff':e_diff,
        'd_diff':d_diff, 
        'e_zero_length':e_zero_length
    }
    if attributes != ('all'):
        for attr in attributes:
            vector.append(attr_dict[attr])
    else:
        vector = [e_lsb,d_lsb,max_0_len,max_1_len,e_diff,d_diff, e_zero_length]

    #return [max_0_len,max_1_len, e_zero_length]
    #print(vector)
    return vector

def generate_text_files(max_size):
    os.mkdir("data_folder")
    printable = string.printable
    for j in range(1, max_size+1):
        filename = "data_folder/" + str(j+14+len(str(j+14)))+'.txt'
        content = ""
        for _ in range(j-2):
            content += secrets.choice(printable)
        with open(filename, 'w', encoding='utf-8') as file:
            print(content, file=file)

def generate_password_vectors(n):
    content = ""
    filename = 'passwords.txt'
    for j in range(n):
        a,b, c = gen_sequences()
        content += '\n'+'-'*20 + ' ' + str(j)+' '+'-'*20
        content += '\n\n'+a+'\n'+b+'\n'+c+'\n'
    with open(filename, 'w', encoding='utf-8') as file:
        print(content, file=file)


def gen_sequences():
    a = b = c = ''
    ham_dist_ab = 0
    ham_dist_ac = 0
    ham_dist_bc = 0
    ascii = string.printable
    escape = '\t\n\r'
    printable = '' 
    for c in ascii:
        if c not in escape:
            printable += c
    min_dist = 0.3 
    while ham_dist_ab < min_dist or ham_dist_ac < min_dist or ham_dist_bc < min_dist:
        a = b = c = ''
        for _ in range(16 + secrets.randbelow(17)):
            a += secrets.choice(printable)
        for _ in range(16 + secrets.randbelow(17)):
            b += secrets.choice(printable)
        for _ in range(16 + secrets.randbelow(17)):
            c += secrets.choice(printable)
        ham_dist_ab = get_ham_dist(a,b)
        ham_dist_ac = get_ham_dist(a,c)
        ham_dist_bc = get_ham_dist(b,c)
    return [a,b,c]

def get_ham_dist(a, b):
    a_bin = ''.join(format(ord(char), '08b') for char in a)
    b_bin = ''.join(format(ord(char), '08b') for char in b)
    if len(a_bin) < len(b_bin):
        a_bin += '0'*(len(b_bin)-len(a_bin))
    else:
        b_bin += '0'*(len(a_bin)-len(b_bin))
    
    count = 0
    for i in range(len(a_bin)):
        if a_bin[i] != b_bin[i]:
            count += 1

    return float(count)/len(a_bin)

def get_statistics(filepath, attributes='all',stat_d = {}):
    if os.path.isdir(filepath):
        stat_mat = []
        i = 0 
        for filename in os.listdir(filepath):
            print(i)
            lsb_list = get_lsb_from_file(filepath+"/"+ filename)
            features = get_features_vector_from_lsb(lsb_list, attributes)
            stat_mat.append(features)
            #stat_mat[i].insert(0,filename)
            if filename in stat_d:
                stat_d[filename].append(features)
            else:
                stat_d[filename] = [features]
            i += 1
        with open('stat_vectors.py', "w") as f:
            file_input = str(stat_mat).replace('],','],\n\t')
            print(file_input, file=f)
        #print(stat_mat)
        return stat_mat
    else:
        lsb_list = get_lsb_from_file(filepath)
        return (get_features_vector_from_lsb(lsb_list))
    
def train_model(image_folder,steg_folder):
    all_stats = ['e_lsb','d_lsb','max_0_len','max_1_len','e_diff','d_diff', 'e_zero_length']
    options = [2,3]
    vectors = list(powerset(options))[1:]
    #print(vectors)
    max_accuracy = 0
    best_svm_clf = ''
    image_stats = get_statistics(image_folder,attributes=['max_0_len','max_1_len'])
    steg_stats = get_statistics(steg_folder, attributes=['max_0_len','max_1_len'])
    
    labels = []
    for i in range(len(image_stats)):
        labels.append(0)
    for j in range(len(steg_stats)):
        labels.append(1)
    for v in vectors[::-1]:
        #temp_image_stats = []
        #temp_steg_stats = []
        #for j in range(len(image_stats)):
        #    temp_image_stats.append([])
        #    for i in v:
        #        temp_image_stats[j].append(image_stats[j][i])
        #for j in range(len(steg_stats)):
        #    temp_steg_stats.append([])
        #    for i in v:
        #        temp_steg_stats[j].append(steg_stats[j][i])
        train_dataset = image_stats + steg_stats
        #print(len(train_dataset),len(image_stats),len(steg_stats), len(labels))
        #for i in range(len(train_dataset)):
        #    print(train_dataset[i], " : ", labels[i] )
        scale_coef = 10 
        scaler = StandardScaler()
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        image_dataset = scaler1.fit_transform(image_stats)
        steg_dataset = scaler2.fit_transform(steg_stats)
        i_train, i_test, iy_train, iy_test = train_test_split(image_dataset, [0]*len(image_stats), test_size=0.2)
        s_train, s_test, sy_train, sy_test = train_test_split(steg_dataset, [1]*len(steg_stats), test_size=0.2)
        
        X = scaler.fit_transform(train_dataset)
        x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
        #clf = svm.SVC(kernel='rbf')                             # Accuracy: 0.65 на полном векторе
        clf = svm.SVC(kernel = 'linear', C=1, gamma='auto')     # Accuracy: 0.66875 на полном векторе признаков
        clf.fit(train_dataset, labels)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_svm_clf = clf
            with open('svm_clf.pkl', 'wb') as f:
                pickle.dump(best_svm_clf, f)
            print("saved svm is:")
            print(v)
        stats = []
        for ind in v:
            stats.append(all_stats[ind])
        print('-'*100)
        print(stats)
        '''
        print("\nSVM classifier:")
       
        #print('Accuracy:', accuracy)
        print("Accuracy on train set: ", clf.score(x_train, y_train))
        print("Accuracy on test set: ", clf.score(x_test, y_test))
        '''
        # bayes classifier
        '''
        classifier = naive_bayes.GaussianNB()
        classifier.fit(x_train, y_train) # fit svm classifier to the train data
        y_prediction = classifier.predict(x_test)
        print("\nSVM classifier: ")
        print("Accuracy on train set: ", classifier.score(x_train, y_train)/10)
        print("Accuracy on test set: ", classifier.score(x_test, y_test)/10)
        '''

        # knn 
        n_neighbors = 5
        classifier = neighbors.KNeighborsClassifier(n_neighbors)
        classifier.fit(x_train, y_train) # fit svm classifier to the train data
        print("\nKNN classifier: ")
        #print("Accuracy on train set: ", classifier.score(x_train, y_train))
        #print("Accuracy on test set: ", classifier.score(x_test, y_test))
        print("Ошибка первого рода:", classifier.score(s_train, sy_train)/scale_coef)
        print("Ошибка второго рода:", classifier.score(i_train, iy_train)/scale_coef)
        with open('knn_clf.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        #mlp 
        '''
        alpha = .01
        classifier = neural_network.MLPClassifier(alpha=alpha, max_iter=100)
        classifier.fit(x_train, y_train) # fit svm classifier to the train data
        y_prediction = classifier.predict(x_test)
        print("MLP NN classifier: ")
        print("Accuracy on train set: ", classifier.score(x_train, y_train))
        print("Accuracy on test set: ", classifier.score(x_test, y_test))
        '''

    with open('svm_clf.pkl', 'wb') as f:
        pickle.dump(best_svm_clf, f)
    #print('max_accuracy: ', max_accuracy)
    return best_svm_clf

def get_model_from_file(model_file):
    with open(model_file, 'rb') as f:
        clf_loaded = pickle.load(f)
        return clf_loaded


def test_model(clf, image_folder,steg_folder):

    print("Counting properties...")
    image_stats = get_statistics(image_folder, )
    steg_stats = get_statistics(steg_folder)
    #print(image_stats)
    labels = []
    for i in range(len(image_stats)):
        labels.append(0)
    for j in range(len(steg_stats)):
        labels.append(1)
    #v = (0, 2, 3, 4, 5, 6)

    temp_image_stats = []
    temp_steg_stats = []
    #for j in range(len(image_stats)):
    #    temp_image_stats.append([])
    #    for i in v:
    #        temp_image_stats[j].append(image_stats[j][i])
    #for j in range(len(steg_stats)):
    #    temp_steg_stats.append([])
    #    for i in v:
    #        temp_steg_stats[j].append(steg_stats[j][i])
    train_dataset = image_stats + steg_stats
    #print(len(train_dataset),len(image_stats),len(steg_stats), len(labels))
    #for i in range(len(train_dataset)):
    #    print(train_dataset[i], " : ", labels[i] )

    scaler = StandardScaler()
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    #print(train_dataset[:10])
    print("Preparing dataset...")
    X = scaler.fit_transform(train_dataset)

    image_dataset = scaler1.fit_transform(image_stats)
    steg_dataset = scaler2.fit_transform(steg_stats)
    i_train, i_test, iy_train, iy_test = train_test_split(image_dataset, [0]*len(image_stats), test_size=0.2)
    s_train, s_test, sy_train, sy_test = train_test_split(steg_dataset, [1]*len(steg_stats), test_size=0.2)
    
    
    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
   
    #clf = svm.SVC(kernel='rbf')                             # Accuracy: 0.65 на полном векторе
    
    #clf = svm.SVC(kernel = 'linear', C=1, gamma='auto')     # Accuracy: 0.66875 на полном векторе признаков
    #clf.fit(train_dataset, labels)
    #print(y_train)
    print("Training dataset...")
    y_pred = clf.predict(x_test)
    count_err1 = 0
    count_err2 = 0
    count_0 = 0
    count_1 = 0
    #print(clf.predict([x_test[7]]))
    for i in range(len(x_test)):
        if y_test[i] == 0:
            count_0 += 1
            if clf.predict([x_test[i]])[0] == 1:
                count_err2 += 1
        elif y_test[i] == 1:
            count_1 += 1
            if clf.predict([x_test[i]])[0] == 0:
                count_err1 += 1
    
    # проверить, что классификатор часть относит к 0, а часть к 1(то есть что он реально бинарный)
    accuracy = accuracy_score(y_test, y_pred)
    err_first = 1.0 - clf.score(i_train, iy_train)
    err_second = 1.0 - clf.score(s_train, sy_train)
    print("Ошибка первого рода: ", float(count_err1) / count_1)
    print("Ошибка второго рода: ", float(count_err2) / count_0)
    #print(v,':')
    print("Accuracy on train set: ", clf.score(x_train, y_train))
    print("Accuracy on test set: ", clf.score(x_test, y_test))


    return accuracy

def create_lsb_containers(input_folder_path, output_folder_path, fill_rate):
    '''
    - функция считывает все файлы, для каждого генерит последовательность случайных текстовых символов
    - рассчитывает размер встраиваемого сообщения как width*height * 3
    - формирует случайный массив подобного размера  
    '''
    os.mkdir(output_folder_path)
    
    for filename in os.listdir(input_folder_path):
        create_lsb_container(input_folder_path + '/'+ filename, output_folder_path + '/'+ filename, fill_rate)


def create_lsb_container(filepath, output_file_path, fill_rate=1.0):
    with open(filepath, 'rb+') as file:
        binValue = file.read()
        hex_dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10,'b':11,'c':12,'d':13,'e':14,'f':15}
        int_to_hex = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'a',11:'b',12:'c',13:'d',14:'e',15:'f'}
        encoded_file = base64.b64encode(binValue)
        image_hex = base64.b64decode(encoded_file).hex()
        pixel_data_start = 2* (struct.unpack("<I", binValue[10:14])[0] - 1)

        lsbit_list = []
        pixel_data_end = get_pixel_data_end(filepath)*2
        amount = len(range(pixel_data_start + 1, pixel_data_start + int(float(pixel_data_end - pixel_data_start)*fill_rate),2))
        #print(amount)
        randbits = bin(secrets.randbits(amount*2))[2:]
        rand_s = ''.join(random.choices(string.printable, k=amount))
        #print(rand_s)
        randbits = []
        for i in rand_s:
            for j in bin(ord(i))[2:]:
                randbits.append(j)
        #print(randbits)
        #print(randbits.count('0'), randbits.count('1'))
        #print(len(randbits))
        #print(len(randbits))
        #print('start', pixel_data_start)
        #print(pixel_data_end)
        #print(len(image_hex))
        j = 0 
        #print(amount)
        #print(len(image_hex))
        #print(pixel_data_start + int(float(pixel_data_end - pixel_data_start)*fill_rate))
        new_image_hex = image_hex[:pixel_data_start+1]
        for i in range(pixel_data_start + 1, pixel_data_start + int(float(pixel_data_end - pixel_data_start)*fill_rate)):
            #lsbit_list.append(hex_dict[image_hex[i]]%2)
            cur_symb = image_hex[i]
            if (i - (pixel_data_start + 1))%2 == 0 and randbits[j] == '1':
                if hex_dict[image_hex[i]]%2 == 1: 
                    #image_hex = image_hex[:i] + int_to_hex[hex_dict[image_hex[i]]-1] + image_hex[i+1:]
                    cur_symb = int_to_hex[hex_dict[image_hex[i]]-1]
                else:
                    #image_hex = image_hex[:i] + int_to_hex[hex_dict[image_hex[i]]+1] + image_hex[i+1:]
                    cur_symb = int_to_hex[hex_dict[image_hex[i]]+1]
            new_image_hex += cur_symb
            j += 1
        new_image_hex += image_hex[len(new_image_hex):]
        #print(len(new_image_hex), len(image_hex))
        binary_data = bytes.fromhex(new_image_hex)
        with open(output_file_path, "wb") as file:
            file.write(binary_data)
        #print(len(lsbit_list))
        data_len = len(lsbit_list)//(2*8*3)
        #print("data length:", data_len)
        return lsbit_list

def main():
    svm_clf = train_model('sample_image', 'sample_steg')
    # print(errors)
    #image_stats = get_statistics('image9191.bmp')
    #steg_stats = get_statistics('bmp_train/image9191.bmp')
    #print(steg_stats)
   
    #get_lsb_from_file('modified.bmp')
    #create_lsb_containers('sample_image',  'sample_steg', 1.0)

    #+`create_lsb_container('COCO_train2014_000000000009.bmp', 'lsb_steg_50%.bmp', 0.5)
    #create_lsb_container('COCO_train2014_000000000009.bmp', 'lsb_steg_100%.bmp', 1.0)


    #print(get_statistics('bmp_train/COCO_train2014_000000000009.bmp'))
    #print(get_statistics('large_bmp_image/COCO_train2014_000000000009.bmp'))
    #generate_text_files(10000)
    #svm_clf = get_model_from_file('svm_clf.pkl')
    #errors = test_model(svm_clf, 'large_bmp_steg', 'full_lsb_steg_images')
    #image_stats = get_statistics('sample_image')
    #print(svm_clf.predict([[ 2, 1, 1.1]]))
    #print(svm_clf.score(image_stats, [0]*len(image_stats)))
    ##print(svm_clf.predict([steg_stats[0]]))
    #print(svm_clf.score(steg_stats, [1]*len(steg_stats)))
    #errors = test_model(svm_clf, 'images_train', 'steg_images_train', 'images_test', 'steg_images_test')
    #print(errors)

    #X = [[0, 0], [1, 1]]
    #y = [0, 1]
    #clf = svm.SVC()
    #clf.fit(X, y)
    #print(clf.predict([[2., 2.]]))


if __name__ == '__main__':
    main()










'''
%matplotlib inline
import numpy as np 
import matploylib.pyplot as plt 
import seaborn as sns; sns.set()

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

fix,ax = plt.subplots(3,5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])


from sklearn.svm import SVC 
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150,whiten=True, random_state=42)
svc = SVC(kernel='rbf',class_weight ='balanced')
model = make_pipeline(pca,svc)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(faces.data, faces.target,
random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 

model = SVC()

param_grid = {
    'C': [1,5,10,50],
    'gamma':[0.0001,]
}

'''