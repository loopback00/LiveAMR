# -*- coding: utf-8 -*-
"""
@author:Zhujiahao(zhujh2001@qq.com)
"""
import os
import sys
sys.path.append('../..')
import time
from typing import List
import operator
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils import get_errors_for_diff_length


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def split_text_into_sentences_by_length(text, length=512):
    """
    将文本切分为固定长度的句子
    :param text: str
    :param length: int, 每个句子的最大长度
    :return: list, (sentence, idx)
    """
    result = []
    for i in range(0, len(text), length):
        result.append((text[i:i + length], i))
    return result

# /home/zhusy/Test/model/mengzi-t5-base-chinese-correction
class T5Corrector:
    # "/home/zhusy/pycorrect_pro/pycorrector-master/examples/t5/outputs-mengzi-t5-base-chinese-correction-7.22-recorrect"
    def __init__(self, model_name_or_path: str = "/usr/LiveDetect/model/llm_model"):
        t1 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.to(device)
        logger.debug("Device: {}".format(device))
        logger.debug('Loaded t5 correction model: %s, spend: %.3f s.' % (model_name_or_path, time.time() - t1))

    def _predict(self, sentences, batch_size=16, max_length=128, silent=True):
        """Predict sentences with t5 model"""
        corrected_sents = []
        for batch in tqdm([sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)],
                          desc="Generating outputs", disable=silent):
            inputs = self.tokenizer(batch, padding=True, max_length=max_length, truncation=True,
                                    return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length)
            for i, sent in enumerate(batch):
                decode_tokens = self.tokenizer.decode(outputs[i], skip_special_tokens=True).replace(' ', '')
                corrected_sent = decode_tokens
                corrected_sents.append(corrected_sent)
        return corrected_sents

    def correct_batch(self, sentences: List[str], max_length: int = 128, batch_size: int = 128, silent: bool = True):
        """

        :param sentences: list[str], sentence list
        :param max_length: int, max length of each sentence
        :param batch_size: int, bz
        :param silent: bool, show log
        :return: list of dict, {'source': 'src', 'target': 'trg', 'errors': [(error_word, correct_word, position), ...]}
        """
        input_sents = []
        sent_map = []
        for idx, sentence in enumerate(sentences):
            if len(sentence) > max_length:
                # split long sentence into short ones
                short_sentences = [i[0] for i in split_text_into_sentences_by_length(sentence, max_length)]
                input_sents.extend(short_sentences)
                sent_map.extend([idx] * len(short_sentences))
            else:
                input_sents.append(sentence)
                sent_map.append(idx)

        # batch predict
        corrected_sents = self._predict(
            input_sents,
            batch_size=batch_size,
            max_length=max_length,
            silent=silent,
        )

        # concatenate the results of short sentences
        corrected_sentences = [''] * len(sentences)
        for idx, corrected_sent in zip(sent_map, corrected_sents):
            corrected_sentences[idx] += corrected_sent

        new_corrected_sentences = []
        corrected_details = []
        for idx, corrected_sent in enumerate(corrected_sentences):
            new_corrected_sent, sub_details = get_errors_for_diff_length(corrected_sent, sentences[idx])
            new_corrected_sentences.append(corrected_sent)
            corrected_details.append(sub_details)
        return [{'source': s, 'target': c, 'errors': e} for s, c, e in
                zip(sentences, new_corrected_sentences, corrected_details)]

    def correct(self, sentence: str, **kwargs):
        """Correct a sentence with t5 csc model"""
        return self.correct_batch([sentence], **kwargs)[0]

    def release_resources(self):
        """Release resources used by the model and tokenizer."""
        logger.debug("Releasing model and tokenizer resources.")
        # Delete the model and tokenizer
        del self.model
        del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Resources released successfully.")

# t5=T5Corrector()
# re= t5.correct_batch(["测试。"])