from transformers import  BertTokenizer,BertForSequenceClassification,RobertaForSequenceClassification
from loguru import logger
from sklearn.metrics import classification_report
import torch
import  argparse
import os

from dataclasses import dataclass, field
from typing import Optional
from  transformers import  HfArgumentParser,TrainingArguments,set_seed,Trainer,RobertaForSequenceClassification
from datasets import  load_dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings

from Utils.processfile import readtsv, writetsv

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
# pwd_path = os.path.abspath(os.path.dirname(__file__))

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
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
        default=None,
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
    parser.add_argument('--train_path', type=str,
                        default="/home/qiang2023/zhujiahao/Fine_Bert/Data/train/train_short_temp.tsv",
                        help='train dataset')
    parser.add_argument('--test_path', type=str,
                        default="/home/qiang2023/zhujiahao/Fine_Bert/Data/test/test_short.tsv",
                        help='test dataset')
    parser.add_argument('--model_name_or_path', type=str, default='/home/qiang2023/zhujiahao/model/bert-base-chinese',
                        help='pretrained model')
    parser.add_argument('--save_dir', type=str, default='/home/qiang2023/zhujiahao/Fine_Bert/Model/finetune/bert-base-chinese-all-sen-temp2', help='save dir')
    parser.add_argument('--max_len', type=int, default=512, help='max length')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--logging_steps', type=int, default=500, help='logging steps num')
    parser.add_argument('--warmup_steps', type=int, default=50, help='logging steps num')
    parser.add_argument('--eval_steps', type=int, default=100, help='eval steps num')
    parser.add_argument('--epochs', type=int, default=40, help='train epochs num')
    parser.add_argument('--max_steps', type=int, default=5000, help='train max steps') # default 5000
    parser.add_argument("--do_train", default=True,action="store_true", help="whether not to do train")
    parser.add_argument("--do_eval", default=True,action="store_true", help="whether not to do eval")
    args = parser.parse_args()

    return args
def tokenize_batch(tokenizer,dataset,max_len):
    def convert_to_features(batch_data):
        src_texts=[]
        trg_texts=[]
        for example in batch_data["text"]:
            items=example.split("\t",1)
            if len(items)==2:
                src_texts.append(items[0])
                trg_texts.append(items[1])
            else:
                warnings.warn("this example is splite wrong with t tag")
        input_encodings=tokenizer.batch_encode_plus(
            src_texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
        )
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels':[int(label) for label in trg_texts]
        }
        return encodings
    dataset=dataset.map(convert_to_features,batched=True)
    columns=["input_ids","attention_mask","labels"]
    dataset.with_format(type='torch', columns=columns)
    dataset = dataset.remove_columns(['text'])
    return dataset



def train():
    args=parse_args()
    args_dict={
        "model_name_or_path": args.model_name_or_path,
        "max_len": args.max_len,
        "output_dir": args.save_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
       "warmup_steps": args.warmup_steps,
    }

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(args_dict)
    set_seed(training_args.seed)

    dataset=load_dataset("text",data_files={"train":args.train_path,"test":args.test_path})
    train_dataset = dataset['train']
    valid_dataset = dataset['test']
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,max_length=512,clean_up_tokenization_spaces=True)
    train_dataset=tokenize_batch(tokenizer,train_dataset,512)
    valid_dataset=tokenize_batch(tokenizer,valid_dataset,512)
    model=BertForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)




def test_model():
    device="cuda"
    model_path = "/home/qiang2023/zhujiahao/Fine_Bert/Model/finetune/bert-base-chinese-allpara"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    # logger.info(model)
    dataset = load_dataset("text", data_files={"test":"/home/qiang2023/zhujiahao/Fine_Bert/Data/test/test3.tsv"})
    logger.info(dataset)
    test_dataset = dataset['test']
    groud_y=[]
    pre_y=[]
    for example in test_dataset:
        temp_list=example["text"].split("\t",1)
        groud_y.append(int(temp_list[1]))
        inputs = tokenizer(temp_list[0], return_tensors="pt").to(device)
        outputs = model(**inputs).logits
        predicted_label = torch.argmax(outputs, dim=1).item()
        pre_y.append(predicted_label)
    report=classification_report(groud_y, pre_y, target_names=["0", "1", "2"])
    print(report)




if __name__ == '__main__':
    train()







