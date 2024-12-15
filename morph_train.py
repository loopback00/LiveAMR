# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass, field
from typing import Optional
import os
import argparse
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import HfArgumentParser, TrainingArguments, Trainer, set_seed
from datasets import load_dataset, Dataset
from loguru import logger
import os

# 设置 CUDA 可见设备为 "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
pwd_path = os.path.abspath(os.path.dirname(__file__))


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_len: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="/home/zhusy/Test/model/mengzi-t5-base-chinese-correction", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    language: str = field(
        default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None."}
    )
    train_language: Optional[str] = field(
        default=None, metadata={"help": "Train language if it is different from the evaluation language."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=False,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    enable_train: Optional[bool] = field(
        default=False,
        metadata={"help": "do training"},
    )
    enable_predict: Optional[bool] = field(
        default=False,
        metadata={"help": "do predict"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=os.path.join( '/home/qiang2023/zhujiahao/Fine_Bert/Data/8.29 train.tsv'),
                        help='train dataset')
    parser.add_argument('--test_path', type=str, default=os.path.join('/home/qiang2023/zhujiahao/Fine_Bert/Data/test1_1600.tsv'),
                        help='test dataset')
    parser.add_argument('--output_dir', type=str, default='/home/qiang2023/zhujiahao/Fine_Bert/Model/finetune/test-t5')
    parser.add_argument('--model_name_or_path', type=str, default='/home/qiang2023/zhujiahao/model/llm_model', help='pretrained model')
    parser.add_argument('--max_len', type=int, default=128, help='max length')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--logging_steps', type=int, default=100, help='logging steps num')
    parser.add_argument('--warmup_steps', type=int, default=200, help='logging steps num')
    parser.add_argument('--eval_steps', type=int, default=250, help='eval steps num')
    parser.add_argument('--num_epochs', type=int, default=1, help='train epochs num')
    parser.add_argument('--max_steps', type=int, default=-1, help='train max steps')
    parser.add_argument("--do_train", action="store_true", help="whether not to do train")
    parser.add_argument("--do_eval", action="store_true", help="whether not to do eval")
    args = parser.parse_args()

    return args


class CscDataset(object):
    def __init__(self, file_path):
        self.data = json.load(open(file_path, 'r', encoding='utf-8'))

    def load(self):
        data_list = []
        for item in self.data:
            data_list.append(item['original_text'] + '\t' + item['correct_text'])
        return {'text': data_list}


def train():
    args = parse_args()
    args_dict = {
        "model_name_or_path": args.model_name_or_path,
        "max_len": args.max_len,
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.logging_steps,
        "evaluation_strategy": "steps",
        "eval_steps": args.eval_steps,
        "num_train_epochs": args.num_epochs,
        "do_train": args.do_train,
        "do_eval": args.do_eval,
        "fp16": False,
        "max_steps": args.max_steps,


    }
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(args_dict)
    set_seed(training_args.seed)
    if args.train_path.endswith('.tsv'):
        dataset = load_dataset('text', data_files={'train': [args.train_path], 'test': args.test_path})
        logger.info(dataset)
        train_dataset = dataset['train']
        valid_dataset = dataset['test']
    elif args.train_path.endswith('.json'):
        d = CscDataset(args.train_path)
        data_dict = d.load()
        train_dataset = Dataset.from_dict(data_dict, split='train')

        d = CscDataset(args.test_path)
        data_dict = d.load()
        valid_dataset = Dataset.from_dict(data_dict, split='test')
        logger.info(train_dataset)
        logger.info(valid_dataset)
    else:
        raise ValueError('train_path must be tsv or json')

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        max_length=data_args.max_len
    )

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # overwriting the default max_length
    tokenizer.model_max_length = data_args.max_len
    model.config.max_length = data_args.max_len
    logger.info(f'train_dataset: {train_dataset[:3]}')

    def tokenize_dataset(tokenizer, dataset, max_len):
        def convert_to_features(example_batch):
            src_texts = []
            trg_texts = []
            for example in example_batch['text']:
                terms = example.split('\t', 1)
                src_texts.append(terms[0])
                trg_texts.append(terms[1])
            input_encodings = tokenizer.batch_encode_plus(
                src_texts,
                truncation=True,
                padding='max_length',
                max_length=max_len,
            )
            target_encodings = tokenizer.batch_encode_plus(
                trg_texts,
                truncation=True,
                padding='max_length',
                max_length=max_len,
            )

            encodings = {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'target_ids': target_encodings['input_ids'],
                'target_attention_mask': target_encodings['attention_mask']
            }

            return encodings

        dataset = dataset.map(convert_to_features, batched=True)

        columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
        dataset.with_format(type='torch', columns=columns)

        dataset = dataset.rename_column('target_ids', 'labels')
        dataset = dataset.rename_column('target_attention_mask', 'decoder_attention_mask')
        dataset = dataset.remove_columns(['text'])
        return dataset

    train_dataset = tokenize_dataset(tokenizer, train_dataset, data_args.max_len)
    valid_dataset = tokenize_dataset(tokenizer, valid_dataset, data_args.max_len)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    trainer.train()

    trainer.save_model()

    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == '__main__':
    train()
