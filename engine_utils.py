import numpy as np
import torch
import random
import os
import logging
import argparse

from torch.utils.data import TensorDataset
from utils.utils_ner import convert_examples_to_features, read_examples_from_file, build_typos_neg_examples, build_ent_mask_examples

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


# default params
def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline", action="store_true",
                        help="Whether to run training.")

    # True for uncased BERT
    parser.add_argument(
        "--do_lower_case",
        default=True,
        # action="store_true",
        help="Set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="Model type selected in the list: [bert, roberta]",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="/root/MODELS/bert-base-uncased/",
        type=str,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--labels",
        default="/root/RobustNER/data/conll2003/labels.txt",
        type=str,
        help="Path to a file containing all labels. "
             "If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name", default="", type=str,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
             "Sequences longer than this will be truncated, "
             "sequences shorter will be padded.",
    )

    parser.add_argument("--loss_type", default="lsr", type=str,
                        help="The loss function to optimize.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--bert_lr", default=5e-5, type=float,
                        help="The initial learning rate for BERT.")
    parser.add_argument("--classifier_lr", default=5e-5, type=float,
                        help="The initial learning rate of classifier.")
    parser.add_argument("--crf_lr", default=1e-3, type=float,
                        help="The initial learning rate of crf")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=float, default=1.0,
                        help="Log every X updates steps.")

    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix "
             "as model_name ending and ending with step number",
    )

    parser.add_argument(
        "--overwrite_output_dir", default=True, action="store_true",
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", default=True, action="store_true",
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    return parser


def arg_process(args):
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda", args.gpu_id)
    args.device = device

    return args


def prepare_optimizer_scheduler(args, model, training_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    bert_parameters = eval('model.{}'.format(args.model_type)).named_parameters()
    crf_parameters = list(model.z_decoder.named_parameters()) \
                     + list(model.v_decoder.named_parameters())

    other_parameters = list(model.z_classifier.named_parameters()) \
                       + list(model.v_classifier.named_parameters()) \
                       + list(model.bn_encoder.named_parameters()) \
                       + list(model.oov_reg.named_parameters()) \
                       + list(model.cross_category_reg.named_parameters())

    args.bert_lr = args.bert_lr if args.bert_lr else args.learning_rate
    args.classifier_lr = args.classifier_lr if args.classifier_lr else args.learning_rate
    args.crf_lr = args.crf_lr if args.crf_lr else args.learning_rate

    optimizer_grouped_parameters = [
        # other params
        {"params": [p for n, p in other_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.classifier_lr},
        {"params": [p for n, p in other_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.classifier_lr},

        # bert params
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.bert_lr},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.bert_lr},

        # crf params
        {"params": [p for n, p in crf_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.crf_lr},
        {"params": [p for n, p in crf_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.crf_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=training_steps
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    return optimizer, scheduler


def load_and_cache_examples(args, mode, data_dir=None):
    data_dir = args.data_dir if data_dir is None else data_dir
    # ***********create online features*****************
    logger.info("Creating features from dataset file at %s", data_dir)
    # 读文件
    examples = read_examples_from_file(data_dir, mode)
    # 将输入转为模型输入数值

    return examples


# 获取原来数据集和负样本数据集特征
def get_ds_features(args, examples, tokenizer, labels, pad_token_label_id):
    if args.mode in ['bn', 'oov']:
        # input_ids, input_mask, valid_mask, segment_ids, label_ids
        features = dataset_2_features(args, examples, tokenizer, labels,
                                      pad_token_label_id)
    # elif args.mode == 'oov':
    #     # input_ids, input_mask, valid_mask, segment_ids, label_ids
    #     features = dataset_2_features(args, examples, tokenizer, labels,
    #                                   pad_token_label_id)
        # TODO
        neg_examples = build_typos_neg_examples(examples, tokenizer,
                                                pmi_json=args.pmi_json)
        neg_features = dataset_2_features(args, neg_examples, tokenizer, labels,
                                          pad_token_label_id,
                                          log_prefix='negative')
        features += neg_features
    elif args.mode == 'cc':
        # input_ids, input_mask, valid_mask, segment_ids, label_ids
        features = dataset_2_features(args, examples, tokenizer, labels,
                                      pad_token_label_id)
        # TODO
        pos_examples, neg_examples = build_cc_features(examples, tokenizer)
        pos_features = dataset_2_features(args, pos_examples, tokenizer, labels,
                                          pad_token_label_id,
                                          log_prefix='negative')
        neg_features = dataset_2_features(args, neg_examples, tokenizer, labels,
                                          pad_token_label_id,
                                          log_prefix='negative')
        features = features + pos_features + neg_features
    else:
        raise Exception('Unsupport mode {}'.format(args.mode))

    return TensorDataset(*features)


# 按照实体是否为
def build_cc_features(examples, tokenizer):
    return None, None


# 获取 mask 机制模型数据集特征
def get_mask_ds_features(args, examples, tokenizer, labels, pad_token_label_id):
    masked_examples = build_ent_mask_examples(examples, tokenizer,
                                              mask_ratio=0.85,
                                              mask_token='[MASK]',
                                              pmi_json=args.pmi_json,
                                              preserve_ratio=args.pmi_preserve)
    features = dataset_2_features(args, masked_examples, tokenizer, labels,
                                  pad_token_label_id)
    return TensorDataset(*features)


def dataset_2_features(args, examples, tokenizer, labels, pad_token_label_id, log_prefix=''):
    features = convert_examples_to_features(
        examples,
        labels,
        args.max_seq_length,
        tokenizer,
        cls_token_at_end=bool(args.model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences,
        # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(args.model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=tokenizer.pad_token_type_id,
        pad_token_label_id=pad_token_label_id,
        log_prefix=log_prefix
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    # valid_mask 用来mask, subword 的输出
    all_valid_mask = torch.tensor([f.valid_mask for f in features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features],
                                   dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features],
                                 dtype=torch.long)

    return all_input_ids, all_input_mask, all_valid_mask, all_segment_ids, all_label_ids


def resuming_training(args, train_dataloader):
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader))
        steps_trained_in_current_epoch = global_step % (len(train_dataloader))

        logger.info("  Continuing training from checkpoint, "
                    "will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch)

    return epochs_trained, steps_trained_in_current_epoch


def model_save(args, output_dir, model, tokenizer):
    # output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


# save predictions result
def predictions_save(origin_file, predictions, output_file):

    with open(output_file, "w") as writer:
        
        with open(origin_file, "r") as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = line.split()[0] + " " + predictions[
                        example_id].pop(0) + "\n"
                    writer.write(output_line)
                else:
                    logger.warning("No prediction for '%s'.", line.split()[0])
                    output_line = line.split()[0] + " " + 'O' + "\n"
                    writer.write(output_line)
