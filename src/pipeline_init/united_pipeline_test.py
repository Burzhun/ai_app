from united_pipeline import united_call_verbose, united_call

# Путь к файлу - необходимо менять при запуске! Чтобы отработало также загляните в src/cv_utils/utils.py
# и там замените YA_DIR_ID в методе api_request
file_name = '/Users/ilyakasimov/Documents/GitHub/Obuch2i-NLP/src/cv_utils/sample_img.jpg'
res_1 = united_call_verbose(file_name=file_name)

print(res_1)
