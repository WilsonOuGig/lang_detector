import glob
import json
from time import time

import pandas as pd

from lang_detector_inference import LanguageDetector

lang_to_code = {"English": "en",
                "Spanish": "es",
                "French": "fr",
                "Italian": "it",
                "German": "de",
                "Portuguese": "pt",
                "Dutch": "nl",
                "Swedish": "sv",
                "Norwegian": "no",
                "Danish": "da",
                "Finnish": "fi",
                "Latin": "la",
                "Romanian": "ro",
                "Catalan": "ca",
                "Hungarian": "hu",
                "Polish": "pl",
                "Czech": "cs",
                "Slovak": "sk",
                "Slovenian": "sl",
                "Croatian": "hr"}

code_to_lang = {code: lang for lang, code in lang_to_code.items()}

max_test_cnt = 2000  # test this number of text fragment for each langauge

result_list = []  # [lang, code, cnt, tp, accuracy]
error_list = [] # [language, language_detected, prob, text]

lang_detector = LanguageDetector()
block_size = lang_detector.block_size

start_time = time()

for file in glob.glob("data/processed/*-qa.json"):
    with open(file, 'rt') as f:
        text_list = json.load(f)

    code = file.split('.')[0].split("/")[-1]
    lang = code_to_lang[code]

    print(f"QA model for {lang=}")


    result = [lang, code, 0, 0, 0]
    cnt = 0
    for text in text_list:
        text = text[:block_size]  # truncate tex to the max length that the model takes
        lang_detected, prob = lang_detector.detect(text)
        result[2] += 1  # increase total count
        if lang_detected == code:
            result[3] += 1  # increase correct detection count
        else:
            error_list.append([lang, code_to_lang[lang_detected], prob, text])
        if result[2] == max_test_cnt:
            break

    result[4] = result[3] / result[2] * 100  # calculate accuracy
    result_list.append(result)

end_time = time()

print(f'time={end_time - start_time} seconds.')

df_metrics = pd.DataFrame(result_list, columns=['language', 'code', 'total', 'correct', 'accuracy'])

df_metrics.to_excel("data/metrics_accuracy.xlsx")

df_error = pd.DataFrame(error_list, columns=['lang', 'lang_detected', 'prob', 'text'])
df_error.to_excel("data/error.xlsx")