'''
    Задачи:
    - добавить кнопку выбрать директорию
        - считывать директорию
        - производить поиск по всем файлам с расширением .jpg, выводить для каждого файла информацию: 
        Найдены/не найдены сигнатуры. Файл содержит/не содержит exif. Содержит файл записи (в секции комментариев, в начале 
        или после маркера конца файла). В конце выводить на экран и в консоль итоговую статистику: Найдены сигнатуры в N файлах: 
        image.jpg - camouflage,
        или сигнатуры в файлах не найдены

    - выложить на гитхаб 
    
    - добавить проверку exif тегов
    Признаки подозрительных exif тегов:
    1) в текстовом теге - все цифры  isnumeric()
    2) в числовом - наличие символов 
    3) невозможное значение или выход за пределы реальных значений 
    4) длина значения тега больше 50 символов 
    5) неизвестный id exif тега 
    вывод списка подозрительных тегов со значениями
    6) проверить на реальных фото с тегами
    Еще признаки:
    + размер больше 50мб 
    +  несоответствие заявленного расширения реальному 
    - составить словарь limits для целочисленных тегов и проверять каждый тег выходит ли он за рамки
    - составить словарь options для тегов которые имеют фиксированный набор возможных значений
    - определить секторы(маркеры), которые хранят свою длину в следующих 2-х байтах
    # Keep reading markers and the chunk data until we reach
    # the 'Start of Scan' marker (ffda) 
    - добавить возможность выбирать какие режимы проверки внедрять в отчет
    - добавить режим службы, чтобы программа запущенная в фоновом режиме проводила анализ выбранной директории 
    и составляла отчет по расписанию
    и в конфигурационном файле сохраняла бы режимы и даты проверок 
    - автоматизировать метод слепого стегоанализа(пользователь вводит 2 контейнера, затем 4 файла со стегами)


'''
from tkinter import * 
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import re
import base64
from tkinter import ttk 
from exif import Image
import os
from os.path import isfile, join, basename
import imghdr
import json
import configparser
import time
import zipfile
import datetime

label = ''
res_label = ''
image_chosen = False
check_pressed = False
has_comment_section = False
has_unknown_marker = False
has_text_after_eof = False
has_message_before_start_byte = False
image_has_exif = False 
exif_version = 0
image_hex = ''
signature_dict = {}
schedules_dict = {}
root = Tk()
root.withdraw()
config = configparser.ConfigParser()  
existing_markers = ['ffc0','ffc1','ffc2','ffc3','ffc4','ffc5','ffc6','ffc7','ffc8','ffc9','ffca','ffcb','ffcc','ffcd','ffce','ffcf',
                    'ffd0','ffd1','ffd2','ffd3','ffd4','ffd5','ffd6','ffd7','ffd8','ffd9','ffda','ffdb','ffdc','ffdd','ffde','ffdf',
                    'ffe0','ffe1','ffe2','ffe3','ffe4','ffe5','ffe6','ffe7','ffe8','ffe9','ffea','ffeb','ffec','ffed','ffee','ffef',
                    'fff0','fff1','fff2','fff3','fff4','fff5','fff6','fff7','fff8','fff9','fffa','fffb','fffc','fffd','fffe']

fix_len_literal_exifs = ['date_time','related_sound_file','date_time_original','date_time_digitized',
                         'offset_time','offset_time_original','offset_time_digitized',
                         'image_unique_id']
variable_length_literal_exifs = ['image_description', 'make', 'model','software',
                         'artist','copyright','maker_note','user_comment',
                         'sub_sec_time','sub_sec_time_original','sub_sec_time_digitized',
                         'camera_owner_name','body_serial_number','lens_make',
                         'lens_model','lens_serial_number','image_title','photographer',
                         'image_editor','camera_firmware','raw_developing_software',
                         'image_editing_software','metadata_editing_software',
                         'spectral_sensitivity', 'gps_satellites','gps_map_datum','interoperability_index']
exifs_limits = {

}

def choose_directory():
    pass

def choose_handler():
    global label, image_chosen, image_hex, image_has_exif, exif_version, res_label, has_message_before_start_byte, check_pressed, has_comment_section, has_unknown_marker, has_text_after_eof

    if check_pressed:
        res_label.destroy()

    filename = askopenfilename() 
    print(filename)
    with open(filename, 'rb') as f:
        binValue = f.read()
        encoded_string = base64.b64encode(binValue)

    image_hex = base64.b64decode(encoded_string).hex()
    check_soi = image_hex[0:4] == 'ffd8'
    check_eof = image_hex[-4:] == 'ffd9'
    check_bmp = image_hex[:4] == '424d'
    has_comment_section = False
    has_unknown_marker = False


    if image_hex.find('fffe') != -1:
        has_comment_section = True
    print(image_hex[:4])

    
    if check_soi:
        print("Файл формата JPG")
        found_markers = re.findall(r'ff..', image_hex)
        for marker in found_markers:
            if marker not in jpeg_markers:
                has_unknown_marker = True
        signature_found = False
        if len(image_hex)-image_hex.rfind('ffd9')>4:
            has_text_after_eof = True
        if image_hex.find('ffd8') != 0:
            has_message_before_start_byte = True
        if image_chosen:
            label.destroy()
        else:
            image_chosen = True
        label = ttk.Label(text=filename.split('/')[-1], background="#cccccc", padding=1, font=("Times New Roman", 12, "bold"))
        label.place(relx=0.18, rely=0.65)
    elif check_bmp:
        print("Файл формата BMP")
        if image_chosen:
            label.destroy()
        else:
            image_chosen = True

        label = ttk.Label(text=filename.split('/')[-1], background="#cccccc", padding=8, font=("Times New Roman", 12, "bold"))
        label.place(relx=0.18, rely=0.65)
    else:
        print("Неподдерживаемый формат файла!")
        check_pressed = True
        res_label = ttk.Label(text='Неподдерживаемый формат!', font=("Times New Roman", 15, "bold"), borderwidth=3, relief="solid", background="#f5425d", padding=15)
        res_label.place(relx=0.26, rely=0.37)



def check_handler():
    global res_label, check_pressed, image_has_exif, exif_version, has_comment_section, has_unknown_marker, has_message_before_start_byte
    if check_pressed:
        res_label.destroy()
    else:
        check_pressed = True

    signature_found = False
    if image_chosen:
        for (program_name, signature) in signature_dict.items():
            if image_hex.find(signature) != -1:
                signature_found = True
                res_label = ttk.Label(text=f"Изображение содержит сигнатуру \n программы {program_name}",font=("Times New Roman", 15, "bold"),borderwidth=3, relief="solid", background="#f5ea73", padding=10)
                res_label.place(relx=0.25, rely=0.35)
        if (not signature_found):
            if has_comment_section:
                res_label = ttk.Label(text='Файл содержит запись в секции\n для комментариев',font=("Times New Roman", 18, "bold"), borderwidth=3, relief="solid", background="#f5425d", padding=15)
            #elif has_unknown_marker:
            #    res_label = ttk.Label(text='Файл содержит неизвестный форматный маркер',font=("Times New Roman", 18, "bold"), borderwidth=3, relief="solid", background="#f5425d", padding=15)
            elif has_text_after_eof:
                res_label = ttk.Label(text='Файл содержит текст после маркера конца файла', font=("Times New Roman", 18, "bold"), borderwidth=3, relief="solid", background="#f5425d", padding=15)
            elif has_message_before_start_byte:
                res_label = ttk.Label(text="Файл содержит текст перед маркером начала файла", font=("Times New Roman", 18, "bold"), borderwidth=3, relief="solid", background="#bdf569", padding=15)
            else:
                res_label = ttk.Label(text='Сигнатуры не обнаружены',font=("Times New Roman", 18, "bold"), borderwidth=3, relief="solid", background="#bdf569", padding=15)
            res_label.place(relx=0.26, rely=0.37)
            has_comment_section = False
            has_text_after_eof = False 
            has_message_before_start_byte = False 
            

def import_signatures():
    config.read("settings.ini")
    for program in config['Signatures']:
        signature_dict[program] = config['Signatures'][program]

def import_schedules():
    config.read("settings.ini")
    for path in config['Schedules']:
        schedules_dict[path] = config['Schedules'][path]
    

def display_signatures():
    print("\nСписок сигнатур на текущий момент:")
    signature_json = json.dumps(signature_dict, indent=4, ensure_ascii=False)
    print(signature_json)

def delete_signature():
    display_signatures()
    print('\nВведите название и версию программы, для которой желаете удалить сигнатуру или:\n"-1" - Для возврата в предыдущее меню\n"0" - Для завершения работы')
    program_name = input().replace(' ','_')
    if program_name == "-1":
       edit_signatures() 
    elif program_name == "0":
        print("Завершаю работу")
    else:
        if program_name not in signature_dict:
            print("Нет записи для данной программы.")
        else:
            del signature_dict[program_name]
            parser = configparser.ConfigParser()
            parser.read('settings.ini')
            parser.remove_option('Signatures', program_name)
            with open('settings.ini', 'w') as configfile:
                parser.write(configfile)
        display_signatures()
        edit_signatures()

def add_signature():
    print('\nВведите название и версию программы или:\n"-1" - Для возврата в меню редактора базы сигнатур\n"0" - Для завершения работы программы')
    program_name = input().replace(' ','_')
    if program_name == "-1":
        edit_signatures() 
    else:
        print('\nВведите сигнатуру программы или:\n"-1" - Чтобы изменить название программы\n"-2" - Чтобы вернуться в меню редактора базы сигнатур\n')
        user_input = input().replace(' ','')
        if user_input == -1:
            add_signature()
        elif user_input == -2:
            edit_signatures()
        else:
            signature_dict[program_name] = user_input
            print("\nСигнатура добавлена\n")
            display_signatures()
            parser = configparser.ConfigParser()
            parser.read('settings.ini')
            if not(parser.has_section("Signatures")):
                parser.add_section("Signatures")
            parser.set('Signatures', program_name, user_input)
            with open('settings.ini', 'w') as configfile:
                parser.write(configfile)
            edit_signatures()

def ask_and_save_signatures():
    user_choice = None
    while user_choice is None:
        print('\nСохранить изменения в сигнатурную базу?\n"1"- Да\n"0" - Нет')
        user_input = input().lower()
        if user_input in ["1","0"]:
            user_choice = user_input
    if user_choice == "1":
        new_config = configparser.ConfigParser()
        new_config.add_section('Signatures')
        for program_name in signature_dict:
            new_config.set('Signatures', program_name, signature_dict[program_name])
        with open('settings.ini', 'w', encoding='utf-8') as configfile:
            new_config.write(configfile)




def start_gui():
    root['bg'] = '#285078'
    root.title('Стегоанализ изображений (Диметрий Волков, Б21-502)')
    root.wm_attributes('-alpha', 1.0)
    root.geometry('650x450')

    root.resizable(width=False, height=False)

    frame = Frame(root, bg='#cccccc')
    frame.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)

    btn = Button(frame, text='Выбрать файл',  font=("Times New Roman", 15, "bold"), bg='yellow', bd=1,relief="solid", command=choose_handler,padx=10, pady=5)
    btn2 = Button(frame, text='Проверить', font=("Times New Roman", 15, "bold"), bg='#22ff22', bd=1, relief="solid",command=check_handler, padx=5,pady=5)
    btn3 = Button(frame, text='Выбрать директорию', font=("Times New Roman", 15, "bold"), bg='yellow',bd=1,relief="solid",command=choose_directory,padx=10,pady=5)
    btn.place(relx=0.1, rely=0.8)
    btn2.place(relx=0.7, rely=0.8)
    btn3.place(relx=0.1, rely=0.65)
    root.mainloop()
    

def start_console():
    user_choice = None
    while user_choice is None:
        print("\nВыберите дальнейшее действие:")
        print('"1" - Провести стегоанализ файла формата JPG,\n"2" - Провести стегоанализ директории\n"3" - Редактировать базу сигнатур\n"4" - Редактировать расписание проверок\n"0" - Завершить работу"\n-1" - Вернуться в главное меню\n')
        user_input = input()
        if user_input not in ["-1", "0","1","2","3"]:
            print('Некорректный ввод. Ожидается число от -1 до 3')
        else:
            user_choice = int(user_input)
    if user_choice == -1:
        main()
    elif user_choice == 0:
        print("Завершаю работу")
    elif user_choice == 1:
        modes = get_modes()
        file_path = choose_object(object_type="файл")
        result = analyze_file(file_path, modes)
        report = result[0]
        pretty_json = json.dumps(report, indent=4, ensure_ascii=False)
        print(pretty_json)
    elif user_choice == 2:
        modes = get_modes()
        directory_path = choose_object(object_type="каталог")
        start = time.time()
        result = analyze_directory(directory_path, {}, modes)
        files_with_signatures = result[1]
        end = time.time()
        report = result[0]
        save_report(report, end-start, files_with_signatures)
        if len(files_with_signatures)>0:
            create_archive(files_with_signatures)
    elif user_choice == 3:
        edit_signatures()
    elif user_choice == 4:
        edit_schedules()

def choose_object(object_type):
    object_path = None
    while object_path is None:
        print(f"Выберите {object_type}:")
        if object_type == "файл":
            object_path = filedialog.askdirectory()
        elif object_type == "каталог":
            object_path = filedialog.askopenfilename()
        if object_path:
            user_choice = None
            while user_choice is None:
                print(f'Выбран {object_type}: {user_choice}.\nПодтвердить?\n"1" - Да\n"0" - Выбрать заново')
                user_input = input()
                if user_input == "1":
                    return object_path
                elif user_input == "0":
                    user_choice = user_input
                    choose_object()
    return object_path

def get_modes():
    modes = []
    existing_modes = {"1":"Сигнатуры","2":"Секции","3":"Маркеры","4":"EXIF"}
    mode_is_chosen = False
    while not(mode_is_chosen):
        print('Введите через пробел список проверок, которые необходимо осуществить для файлов:\n"1" - Поиск сигнатур\n"2" - Проверка секций файла\n"3" - Проверка маркеров\n"4" - Проверка EXIF-тегов\n')
        user_input = list(input().split())
        has_invalid_mode = False
        for mode in user_input:
            if mode not in existing_modes:
                has_invalid_mode = True 
        if not(has_invalid_mode):
            mode_is_chosen = True 
            for mode in user_input:
                if existing_modes[mode] not in modes:
                    modes.append(existing_modes[mode])
    user_choice = None
    while user_choice is None:
        print("Список проводимых проверок:", modes)
        print('Подтвердить?\n"1" - Да\n"0" - Выбрать заново\n"-1" - Вернуться в главное меню')
        user_input = input()
        if user_input == "1":
            return modes 
        elif user_input == "0":
            user_choice = user_input
            get_modes()
        elif user_input == "-1":
            user_choice = user_input
            start_console()
    return modes
    


def create_archive(files_with_signatures):
    archive_filename = ''
    existing_files = os.listdir('.')

    while archive_filename == '':
        print("\nВведите уникальное название для zip-архива из файлов, содержащих сигнатуры")
        user_input = input()
        if user_input[-4:] != ".zip":
            print("\nНазвание файла должно оканчиваться на .zip")
        elif user_input in existing_files:
            print("\nФайл с таким именем уже существует. Попробуйте ещё раз")
        else:
            archive_filename = user_input

    with zipfile.ZipFile(archive_filename, 'w') as found_files:
        for filepath in files_with_signatures:
            found_files.write(filepath, basename(filepath))

def save_report(report, time_elapsed, files_with_signatures):
    report_filename = ''
    existing_files = os.listdir('.')
    while report_filename == '':
        print("\nВведите уникальное название для текстового файла, в который будет сохранен отчет")
        user_input = input()

        if user_input[-4:] != ".txt":
            print("\nНазвание файла должно оканчиваться на .txt")
        elif user_input in existing_files:
            print("\nФайл с таким именем уже существует. Попробуйте ещё раз")
        else:
            report_filename = user_input
        
    with open(report_filename, 'w', encoding='utf-8') as file:
        for filename in report:
            print( '-'*100, file=file)
            print(filename+":", file=file)
            report_json = json.dumps(report[filename], indent=4, ensure_ascii=False)
            print(report_json, file=file)
        print("\n"+'*'*100, file=file)
        print(f"Результаты:\nБыл проведен стегоанализ для {len(report)} файлов. \nВремя работы алгоритма: {time_elapsed} секунд.\n", file=file)
        found_amount = len(files_with_signatures)
        file_word = 'файл' if found_amount < 2 else 'файлов'
        if found_amount == 0:
            print("В выбранной директории файлов с сигнатурами не обнаружено", file=file)
        else:
            print(f"В выбранной директории обнаружено {found_amount} {file_word} с сигнатурой:", file=file)
            files_json = json.dumps(files_with_signatures, indent=4, ensure_ascii=False)
            print(files_json, file=file)



def edit_signatures():
    user_choice = None
    print("\nВыберите дальнейшее действие:")
    while user_choice is None:
        print('"1" - Добавить новую сигнатуру\n"2" - Удалить существующую сигнатуру\n"3" - Вывести список сигнатур\n"0" - Завершить работу\n"-1" - Вернуться на предыдущий шаг меню\n')
        user_input = input()
        if user_input not in ["-1", "0","1","2","3"]:
            print('Некорректный ввод. Ожидается число от -1 до 3')
        else:
            user_choice = user_input
    if user_choice == "-1":
        start_console()
    elif user_choice == "0":
        print("Завершаю работу")
    elif user_choice == "1":
        add_signature()
    elif user_choice == "2":
        delete_signature()
    elif user_choice == "3":
        display_signatures()
        edit_signatures()

def edit_schedules():
    '''
        Выводит список действий: вывести текущее расписание, изменить расписание,
        запланировать проверку 
        убрать проверку 
        Изменить параметры существующей проверки 
        слева в конфиге файл или директория 
        справа режимы проверки в виде номеров и дата
        запланировать проверку
        частота: 1 раз
        дни недели, время 
    '''

    user_choice = None
    print("\nВыберите дальнейшее действие:")
    while user_choice is None:
        print('"1" - Запланировать новую проверку\n"2" - Отменить существующую проверку\n"3" - Вывести список существующих проверок\n"0" - Завершить работу\n"-1" - Вернуться на предыдущий шаг меню\n')
        user_input = input()
        if user_input not in ["-1", "0","1","2","3"]:
            print('Некорректный ввод. Ожидается число от -1 до 3')
        else:
            user_choice = user_input
    if user_choice == "-1":
        start_console()
    elif user_choice == "0":
        print("Завершаю работу")
    elif user_choice == "1":
        add_schedule()
    elif user_choice == "2":
        delete_schedule()
    elif user_choice == "3":
        display_schedules()
        edit_schedules()
    

def add_schedule():
    user_choice = None
    objects = {"1":"файл","2":"каталог"}
    while user_choice is None:
        print('Выберите объект проверки:\n"1" - Файл\n"2" - Директория')
        user_input = input()
        if user_input in ["1","2"]:
            user_choice = user_input
    
    object_path = choose_object(objects[user_choice])
    frequency = get_frequency()
    modes = get_modes()
    parser = configparser.ConfigParser()
    parser.read('settings.ini')
    if not(parser.has_section("Schedules")):
        parser.add_section("Schedules")
    schedules_dict[object_path] = f"{frequency}_{",".join(modes)}"
    parser.set('Schedules', object_path, schedules_dict[object_path])
    with open('settings.ini', 'w') as configfile:
        parser.write(configfile)

def delete_schedule():
    parser = configparser.ConfigParser()
    parser.read("settings.ini")

    object_path = None 
    while object_path is None:
        print('Введите абсолютный путь до файла или директории, для которой необходимой отменить проверки или "-1" для возврата в предыдущий пункт меню:\n')
        user_input = input()
        if user_input == "-1":
            object_path = ""
            edit_schedules()
        elif parser.has_option("Schedules", user_input):
            parser.remove_option("Schedules", user_input)
            with open('settings.ini', 'w') as configfile:
                parser.write(configfile)
            object_path = user_input 
        else:
            print("Для данного пути проверки не запланированы. Попробуйте еще раз")
    

def get_frequency():
    user_choice = None
    while user_choice is None:
        print('Как часто необходимо проводить проверки?\n"0" - Выполнить проверку единожды\n"1" - Каждый час\n"2" - Ежедневно\n"3" - В определенные дни недели')
        user_input = input()
        if user_input in ["0","1","2","3"]:
            user_choice = user_input
    if user_choice == 0:
        date = get_datetime()
        return f"once_{date}"
    elif user_choice == 1:
        minutes = None
        while minutes is None:
            print("В какую минуту необходимо проводить проверку? Введите число от 0 до 59")
            user_input = input()
            if user_input.isnumeric() and int(user_input) >= 0 and int(user_input) <= 59:
                minutes = user_input 
                return f"hourly_{user_input}"
    elif user_choice == 2:
        check_time = None
        while check_time is None:
            print("Введите время проведения проверки в формате: hh:mm")
            user_input = input()
            valid_hours = len(user_input) == 5 and user_input[:2].isnumeric() and int(user_input[:2]) >= 0 and int(user_input[:2]) <= 59
            valid_minutes = len(user_input) == 5 and user_input[-2:].isnumeric() and int(user_input[-2:]) >= 0 and int(user_input[-2:]) <= 23
            if valid_hours and valid_minutes:
                return f"daily_{user_input}"
    elif user_choice == 3:
        weekday = None
        while weekday is None:
            print('Выберите день недели:\n"1" - Понедельник\n"2" - Вторник\n"3" - Среда\n"4" - Четверг\n"5" - Пятница\n"6" - Суббота\n"7" - Воскресенье')
            user_input = input()
            if user_input in "1234567":
                weekday = user_input 
        date_and_time = get_datetime()
        return f"weekly_{weekday}_{date_and_time}"



def get_datetime():
    check_date = None
    while check_date is None:
        print('Введите дату в формате: "гггг-мм-дд"')
        user_input = input()
        try:
            check_date = datetime.date.fromisoformat(user_input)
        except ValueError:
            print('Недопустимый формат даты, должен быть: "гггг-мм-дд"')
            check_date = None
    check_time = None 
    while check_time is None:
        print("Введите время проверки в формате чч:мм:сс")
        user_input = input()
        if len(user_input) == 8:
            check_hours = user_input[:2].isnumeric() and int(user_input[:2]) >= 0 and int(user_input[:2]) <= 23
            check_minutes = user_input[3:5].isnumeric() and int(user_input[3:5]) >= 0 and int(user_input[3:5]) <= 59
            check_seconds = user_input[-2:].isnumeric() and int(user_input[-2:]) >= 0 and int(user_input[-2:]) <= 59
            if check_hours and check_minutes and check_seconds:
                check_time = user_input 
    return f"{check_date}_{check_time}"

def analyze_directory(folder_path, total_report, modes=["Сигнатуры","EXIF","Маркеры","Секции"]):
    ''' 
     считывает директорию и по каждому пути файла сохраняет информацию: сигнатуры найдены/нет, есть ли exif, есть ли записи с полях(начало/конец/секция комментариев)
    '''

    files_with_signatures = {}
    if os.path.isdir(folder_path):
        for f in os.listdir(folder_path):
            path = folder_path + "/" + f
            if isfile(path):
                if imghdr.what(path) == 'jpeg': 
                    result = analyze_file(path, modes)
                    file_report = result[0]
                    total_report[path] = file_report
                    if len(result[1]) != 0:
                        files_with_signatures[path] = result[1]
            else:
                analyze_directory(path, total_report, modes)

        return [total_report, files_with_signatures]
    else:
        pass
    
def analyze_file(filename, check_list=["EXIF","Сигнатуры","Маркеры", "Секции"]):
    ''' 
    "Поиск по базе сигнатур":
    {
        {"Deegger Embedder v"}: "Нет".
        {"Camouflage v123}:"Нет"
    }
    "Анализ полей формата":
    {
        "начало файла": нет
        "секция комментариев": "нет"
        "конец файла": нет,
        "exif теги": нет
    }
    "Анализ информационных полей":
    {
        RS- анализ: нет 
        WS-анализ: ...
    }
    '''
    copy_filename = filename 
    file_report = {}
    encoded_file = ''
    file_report['Анализ структуры файла'] = {}
    file_report["Поиск сигнатур"] = {}
    file_report["Общий анализ файла"] = {}

    with open(filename, 'rb') as filename:
        binValue = filename.read()
        encoded_file = base64.b64encode(binValue)

    image_hex = base64.b64decode(encoded_file).hex()
    start_of_image = image_hex.find('ffda')

    if "Секции" in check_list:
        if image_hex.find('ffd8') != 0:
            file_report["Анализ структуры файла"]["Запись в начале файла"] = "Да"
        else:
            file_report["Анализ структуры файла"]["Запись в начале файла"] = "Нет"
        start_of_comments = (image_hex[:start_of_image]).find('fffe')
        if start_of_comments != -1:
            end_of_comments = image_hex[start_of_comments+4:].find('ff')
            if end_of_comments > start_of_comments + 4:
                file_report["Анализ структуры файла"]["Запись в секции для комментариев"]= image_hex[start_of_comments+4:end_of_comments]
            else:
                file_report["Анализ структуры файла"]["Запись в секции для комментариев"] = "Нет"
        else:
            file_report["Анализ структуры файла"]["Запись в секции для комментариев"]= "Нет"

        if len(image_hex)-image_hex.rfind('ffd9')>4:
            file_report["Анализ структуры файла"]["Запись в конце файла"] = "Да"
        else:
            file_report["Анализ структуры файла"]["Запись в конце файла"] = "Нет"
    signatures_found = []

        
    if "Сигнатуры" in check_list:
        for program in signature_dict:
            if signature_dict[program] in image_hex:
                file_report["Поиск сигнатур"][program] = "Да"
                signatures_found.append(program)
            else:
                file_report["Поиск сигнатур"][program] = "Нет"
    
    if os.path.getsize(copy_filename) > 50_000_000:
        file_report["Общий анализ файла"]["Аномальный размер файла"] = "Да"
    else:
        file_report["Общий анализ файла"]["Аномальный размер файла"] = "Нет"
    
    if copy_filename[-4:] != ".jpg":
        file_report["Общий анализ файла"]["Несоответствие расширения содержимому"] = "Да"
    else:
        file_report["Общий анализ файла"]["Несоответствие расширения содержимому"] = "Нет"
    
    if "EXIF" in check_list or "Маркеры" in check_list:
        if "EXIF" in check_list:
            with open(copy_filename, 'rb') as image_file:
                image = Image(image_file)
                exif_list = dir(image)
                if image.has_exif:
                    file_report["Анализ структуры файла"]["Содержит exif-теги"] = "Да"
                    suspicious_exif_tags = []
                    file_report["Анализ структуры файла"]["Подозрительные exif-теги"] = ''
                    try:
                        for tag in exif_list:
                            if tag in fix_len_literal_exifs and image[tag].isnumeric():
                                suspicious_exif_tags.append(tag)
                            elif tag in variable_length_literal_exifs and (image[tag].isnumeric() or len(image[tag]))>= 100:
                                suspicious_exif_tags.append(tag)
                        for tag in suspicious_exif_tags:
                            file_report["Анализ структуры файла"]["Подозрительные exif-теги"][tag] = image[tag]
                        if len(suspicious_exif_tags) == 0:
                            file_report["Анализ структуры файла"]["Подозрительные exif-теги"] = "Нет"
                    except:
                        pass
                else:
                    file_report["Анализ структуры файла"]["Содержит exif-теги"] = "Нет"
        
        ind = 0
        if "Маркеры" in check_list:
            marker_indexes = list(re.finditer(r'ff..', image_hex[:start_of_image]))
        
            unknown_markers = []
            suspicious_markers = []
            for ind in range(len(marker_indexes)):
                match = marker_indexes[ind]
                marker = image_hex[match.start(0):match.end(0)]
                if marker not in existing_markers:
                    unknown_markers.append(marker)
                
            
                marker_size = convert_hexnum(image_hex[match.end(0):match.end(0)+4])
                if ind < len(marker_indexes) - 1 and ((marker_indexes[ind+1]).start(0) - marker_indexes[ind].end(0))!= marker_size:
                    suspicious_markers.append(marker)
                elif ind == len(marker_indexes) - 1 and marker_size != (start_of_image - marker_indexes[ind].end(0)):
                    suspicious_markers.append(marker)


            
            if len(unknown_markers) > 0:
                file_report["Анализ структуры файла"]["Неизвестные маркеры"] = unknown_markers
            else:
                file_report["Анализ структуры файла"]["Неизвестные маркеры"] = "Нет"
            if len(suspicious_markers) > 0:
                file_report["Анализ структуры файла"]["Подозрительные маркеры"] = suspicious_markers
            else:
                file_report["Анализ структуры файла"]["Подозрительные маркеры"] = "Нет"
            
    return [file_report, signatures_found]

def convert_hexnum(hexstr):
    hexdict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10,'b':11,'c':12,'d':13,'e':14,'f':15}
    res = 0
    for c in hexstr[::-1]:
        res *= 16 
        res += hexdict[c]
    return res 


def test_performance():
    c = list(input().split())
    modes = []
    existing_modes = {"1":"Сигнатуры","2":"Секции","3":"Маркеры","4":"EXIF"}
    for mode in c:
        modes.append(existing_modes[mode])
    folder_selected = filedialog.askdirectory()
    start = time.time()
    result = analyze_directory(folder_selected, {}, modes)
    files_with_signatures = result[1]
    end = time.time()
    report = result[0]
    save_report(report, end-start, files_with_signatures)
    if len(files_with_signatures) > 0:
        create_archive(files_with_signatures)
    
def start_schedules():
    for path in schedules_dict:
        arguments = schedules_dict[path].split('_')
        if arguments[0] == "once":
            pass
        elif arguments[0] == "hourly":
            schedule.every().hour.at(f":{arguments}").do(job)
            pass
        elif arguments[0] == "daily":
            pass 
        elif arguments[0] == "weekly":
            pass 
        



def main():
    import_signatures()
    import_schedules()
    start_schedules()

    mode = -1
    while mode == -1:
        print('\nВыберите режим работы:\n"1" - Консольный\n"2" - Графический\n"0" - Завершить работу\n')
        user_input = input()
        if user_input not in ["0", "1","2"]:
            print('Режим работы программы может принимать значения "0","1" или "2"')
        else:
            mode = int(user_input)
    if mode == 1:
        start_console()
    elif mode == 2:
        start_gui()
    else:
        print("Завершаю работу")

if __name__ == '__main__':
    test_performance()
