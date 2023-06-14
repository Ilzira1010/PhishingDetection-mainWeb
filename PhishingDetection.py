# import re
# import LinkFeatureGenerator as lfg
# import MLModelsApplyer as mlmapl
# import NlpModelsSaver as nlpSaver
# import NlpModelsApplyer as nlpapl
# import numpy as np
#
# with open('filename.html', 'r') as file:
#     text = file.read()
#
# url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
#
# def detect_print():
#     urls = re.findall(url_pattern, text)
#
#     for url in urls:
#         features = lfg.create_features(url)
#         mlmapl.apply(features)
#
#     nlp_features1d = nlpSaver.create_nlp_features(text)
#     nlp_features = np.reshape(nlp_features1d, (1, len(nlp_features1d)))
#     nlpapl.nlp_apply(nlp_features)
#
# def detectPhising(content):
#     urls = re.findall(url_pattern, content)
#     y_pred = []
#     if (len(urls) > 0):
#         features = lfg.create_features(urls[0])
#         y_pred.append(mlmapl.apply_to_one_res(features))
#
#     nlp_features1d = nlpSaver.create_nlp_features(text)
#     nlp_features = np.reshape(nlp_features1d, (1, len(nlp_features1d)))
#     y_pred.append(nlpapl.nlp_apply_one_result(nlp_features))
#
#     return defence_hack(content)
#     #if (sum(y_pred) / len(y_pred) > 0.5):
#         #return "Является фишингом"
#     #else:
#         #return "Не является фишингом"
#
# def defence_hack(content):
#     if content.endswith(" "):
#         return "Является фишингом"
#     else:
#         return "Не является фишингом"
import re
import LinkFeatureGenerator as lfg
import MLModelsApplyer as mlmapl
import NlpModelsSaver as nlpSaver
import NlpModelsApplyer as nlpapl
import numpy as np

with open('filename.html', 'r') as file:
    text = file.read()

url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def detect_print():
    urls = re.findall(url_pattern, text)

    for url in urls:
        features = lfg.create_features(url)
        mlmapl.apply(features)

    nlp_features1d = nlpSaver.create_nlp_features(text)
    nlp_features = np.reshape(nlp_features1d, (1, len(nlp_features1d)))
    nlpapl.nlp_apply(nlp_features)

def detectPhising(content):
    urls = re.findall(url_pattern, content)
    y_pred = []
    if (len(urls) > 0):
        features = lfg.create_features(urls[0])
        y_pred.append(mlmapl.apply_to_one_res(features))

    nlp_features1d = nlpSaver.create_nlp_features(text)
    nlp_features = np.reshape(nlp_features1d, (1, len(nlp_features1d)))
    y_pred.append(nlpapl.nlp_apply_one_result(nlp_features))

    if (sum(y_pred) / len(y_pred) > 0.5):
        return "Является фишингом"
    else:
        return "Не является фишингом"