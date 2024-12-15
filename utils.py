import operator
import re
def is_chinese_char(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'


def is_chinese_string(string):
    """判断是否全为汉字"""
    return all(is_chinese_char(c) for c in string)
def get_errors_for_diff_length(corrected_text, origin_text):
    """Get errors between corrected text and origin text"""
    new_corrected_text = ""
    errors = []
    i, j = 0, 0
    unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

    while i < len(origin_text) and j < len(corrected_text):
        if origin_text[i] in unk_tokens:
            new_corrected_text += origin_text[i]
            i += 1
        elif corrected_text[j] in unk_tokens:
            new_corrected_text += corrected_text[j]
            j += 1
        # Deal with Chinese characters
        elif is_chinese_char(origin_text[i]) and is_chinese_char(corrected_text[j]):
            # If the two characters are the same, then the two pointers move forward together
            if origin_text[i] == corrected_text[j]:
                new_corrected_text += corrected_text[j]
                i += 1
                j += 1
            else:
                # Check for insertion errors
                if j + 1 < len(corrected_text) and origin_text[i] == corrected_text[j + 1]:
                    errors.append(('', corrected_text[j], j))
                    new_corrected_text += corrected_text[j]
                    j += 1
                # Check for deletion errors
                elif i + 1 < len(origin_text) and origin_text[i + 1] == corrected_text[j]:
                    errors.append((origin_text[i], '', i))
                    i += 1
                # Check for replacement errors
                else:
                    errors.append((origin_text[i], corrected_text[j], i))
                    new_corrected_text += corrected_text[j]
                    i += 1
                    j += 1
        else:
            new_corrected_text += origin_text[i]
            if origin_text[i] == corrected_text[j]:
                j += 1
            i += 1
    errors = sorted(errors, key=operator.itemgetter(2))
    return new_corrected_text, errors

def sentence_piece(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")

def convert_english_to_chinese_punctuation(text):
    # 定义英文符号到中文符号的映射
    punctuation_map = {
        ',': '，',
        '.': '。',
        '?': '？',
        '!': '！',
        ':': '：',
        ';': '；',
        '(': '（',
        ')': '）',
        '[': '【',
        ']': '】',
        '{': '｛',
        '}': '｝',
        '"': '“',
        "'": '‘',
        '<': '《',
        '>': '》',
        '/': '／',
        '\\': '＼',
        '@': '＠',
        '#': '＃',
        '$': '＄',
        '%': '％',
        '^': '＾',
        '&': '＆',
        '*': '＊',
        '-': '－',
        '_': '＿',
        '=': '＝',
        '+': '＋',
        '`': '｀',
        '~': '～',
        '|': '｜'
    }

    # 遍历文本中的每个字符，进行符号转换
    converted_text = ''.join(punctuation_map.get(char, char) for char in text)
    return converted_text
