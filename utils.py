import re
import pandas as pd

# def remove_emojis(text):
#     emoji_pattern = re.compile("["
#                                "\U0001F600-\U0001F64F"  # Emoticons
#                                "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
#                                "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
#                                "\U0001F700-\U0001F77F"  # Alchemical Symbols
#                                "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
#                                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
#                                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
#                                "\U0001FA00-\U0001FA6F"  # Chess Symbols
#                                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
#                                "\U00002702-\U000027B0"  # Dingbats
#                                "]+", flags=re.UNICODE)
#     return emoji_pattern.sub(r'', text)
#
#
# import re


def remove_emojis_and_eol_from_list(char_list):
    # Emoji and end-of-line Unicode ranges
    pattern = re.compile("["
                         "\U0001F600-\U0001F64F"  # Emoticons
                         "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
                         "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                         "\U0001F700-\U0001F77F"  # Alchemical Symbols
                         "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                         "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                         "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                         "\U0001FA00-\U0001FA6F"  # Chess Symbols
                         "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                         "\U00002702-\U000027B0"  # Dingbats
                         "\n\r"  # Newline and carriage return
                         "\n"  # Newline
                         "]+", flags=re.UNICODE)

    # Filter out emojis and end-of-line characters from the list
    char_list_without_emojis_and_eol = [char for char in char_list if not pattern.match(char)]

    return char_list_without_emojis_and_eol




def read_parquet(file):
    df = pd.read_parquet(file, engine='fastparquet')
    text = " ".join(df["text"].values.tolist())
    return text


