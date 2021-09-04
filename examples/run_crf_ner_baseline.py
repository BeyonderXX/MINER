# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import argparse
import logging
import random
import sys

sys.path.append('../')

import numpy as np
import torch
from utils.utils_metrics import get_entities_bio, f1_score, classification_report
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from models.model_ner import MODEL_FOR_CRF_NER_MAPPING, AutoModelForCrfNer

from utils.utils_ner import convert_examples_to_features, get_labels, read_examples_from_file, collate_fn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CRF_NER_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), ())
TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    # added by remove up code
    tb_writer = SummaryWriter(args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.epoch = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epoch

    args.logging_steps = eval(args.logging_steps)
    if isinstance(args.logging_steps, float):
        args.logging_steps = int(args.logging_steps * len(train_dataloader)) // args.gradient_accumulation_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_parameters = eval('model.{}'.format(args.model_type)).named_parameters()
    classifier_parameters = model.classifier.named_parameters()
    crf_parameters = model.crf.named_parameters()

    args.bert_lr = args.bert_lr if args.bert_lr else args.learning_rate
    args.classifier_lr = args.classifier_lr if args.classifier_lr else args.learning_rate
    args.crf_lr = args.crf_lr if args.crf_lr else args.learning_rate
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.bert_lr},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.bert_lr},

        {"params": [p for n, p in classifier_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.classifier_lr},
        {"params": [p for n, p in classifier_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.classifier_lr},

        {"params": [p for n, p in crf_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.crf_lr},
        {"params": [p for n, p in crf_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.crf_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epoch)

    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (1),
    )

    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    best_score = 0.0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.epoch), desc="Epoch")
    set_seed(args)  # Added here for reproductibility

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "valid_mask": batch[2],
                      "labels": batch[4], }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[3] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    if (args.evaluate_during_training):
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                              prefix=global_step)
                        for key, value in results.items():
                            if isinstance(value, float) or isinstance(value, int):
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        if best_score < results['f1']:
                            best_score = results['f1']
                            # output_dir = os.path.join(args.output_dir, "best_checkpoint")
                            output_dir = args.output_dir
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

            # # accelerate debug
            # break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="", skip_label=-1):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_train_batch_size
    # args.eval_batch_size = 1
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    trues = None
    model.eval()

    # TODO
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "valid_mask": batch[2],
                      "labels": batch[4],
                      "decode": True,
                      "skip_label": skip_label}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            # TODO
            outputs = model(**inputs)
            tmp_eval_loss, tags = outputs[:2]

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1

        if preds is None:
            preds = tags.detach().cpu().numpy()
            trues = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, tags.detach().cpu().numpy(), axis=0)
            trues = np.append(trues, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    label_map = {i: label for i, label in enumerate(labels)}

    trues_list = [[] for _ in range(trues.shape[0])]
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(trues.shape[0]):
        for j in range(trues.shape[1]):
            if trues[i, j] != pad_token_label_id:
                trues_list[i].append(label_map[trues[i][j]])
                # 超长的 case ， 默认剩余长度预测为 O
                if preds[i][j] != skip_label:
                    preds_list[i].append(label_map[preds[i][j]])
                else:
                    preds_list[i].append("O")

    true_entities = get_entities_bio(trues_list)
    pred_entities = get_entities_bio(preds_list)
    results = {
        "loss": eval_loss,
        "f1": f1_score(true_entities, pred_entities),
        'report': classification_report(true_entities, pred_entities)
    }

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("***** Eval results {} *****\n".format(prefix))
        writer.write("***** Eval loss : {} *****\n".format(eval_loss))
        for key in sorted(results.keys()):
            if key == 'report_dict':
                continue
            logger.info("{} = {}".format(key, str(results[key])))
            writer.write("{} = {}\n".format(key, str(results[key])))
    return results, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
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
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=pad_token_label_id,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    if not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_valid_mask = torch.tensor([f.valid_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_valid_mask, all_segment_ids, all_label_ids)
    return dataset


def arg_parse():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="/root/RobustNER/data/conll2003/",
        type=str,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="/root/MODELS/bert-base-uncased/",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
            ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default="/root/RobustNER/out/bert_uncase/baseline/",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument("--gpu_id", default=3, type=int,
                        help="GPU number id")

    parser.add_argument(
        "--labels",
        default="/root/RobustNER/data/conll2003/labels.txt",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
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
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--do_train", default=True, action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", default=True, action="store_true",
                        help="Whether to run predictions on the test set.")

    parser.add_argument(
        "--evaluate_during_training",
        default=True,
        # action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    # bert uncase setting
    parser.add_argument(
        "--do_lower_case", default=True, action="store_true",
        help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True,
        help="Set this flag if model is trained with accents."
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True,
        help="Set this flag if model is trained without accents."
    )
    parser.add_argument("--use_fast", action="store_const", const=True,
                        help="Set this flag to use fast tokenization.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
    parser.add_argument(
        "--epoch", default=50, type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override epoch.",
    )
    parser.add_argument("--warmup_steps", default=50, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=str, default='0.1',
                        help="Log every X updates steps.")

    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
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

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    return parser.parse_args()


def arg_process(args):
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda", args.gpu_id)
    args.n_gpu = 1
    args.device = device

    return args


def main():
    args = arg_process(arg_parse())

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16
    )

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=None,
    )
    #####
    setattr(config, 'loss_type', args.loss_type)
    #####
    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=None,
        **tokenizer_args,
    )
    model = AutoModelForCrfNer.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=None,
    )

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    if args.do_eval:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        # checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
        model = AutoModelForCrfNer.from_pretrained(args.output_dir)
        model.to(args.device)
        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix='dev')
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            writer.write('***** Predict in dev dataset *****')
            writer.write("{} = {}\n".format('report', str(results['report'])))

    if args.do_predict:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        # checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
        model = AutoModelForCrfNer.from_pretrained(args.output_dir)
        model.to(args.device)
        results, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test", prefix='test')

        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            writer.write('***** Predict in dev dataset *****')
            writer.write("{} = {}\n".format('report', str(results['report'])))

        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "test.txt"), "r") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not predictions[example_id]:
                            example_id += 1
                    elif predictions[example_id]:
                        output_line = line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
                        writer.write(output_line)
                    else:
                        logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])


if __name__ == "__main__":
    main()
