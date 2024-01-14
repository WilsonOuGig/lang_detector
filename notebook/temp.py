from datasets import load_dataset

# ds = load_dataset("wikimedia/wikipedia", "20231101.en")




import re

def remove_emojis(text):
    emoji_pattern = re.compile("["
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
                               "\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Example usage
text_with_emojis = "Hello, 😀 how  北大 are you? 🌍"
text_without_emojis = remove_emojis(text_with_emojis)

print("Original Text:", text_with_emojis)
print("Text without Emojis:", text_without_emojis)






def remove_emojis(text):
    # Emoji Unicode ranges
    emoji_pattern = re.compile("["
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
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Example usage
text_with_emojis = "Hello, 😀 你好吗？ 🌍   み (mi), け (ke), へ (he), め (me), こ (ko), そ (so), と (to), の (no), も (mo), よ (yo) and ろ (ro 대한제국; 大韓帝國; "
text_without_emojis = remove_emojis(text_with_emojis)

print("Original Text:", text_with_emojis)
print("Text without Emojis:", text_without_emojis)