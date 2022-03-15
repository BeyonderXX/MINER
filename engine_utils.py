import numpy as np
import torch
import random
import os
import logging
import argparse


from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def set_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


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
        # default='bert-large-uncased',
        type=str,
        help="Path to pre-trained model or shortcut name",
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
        "--max_seq_len",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
             "Sequences longer than this will be truncated, "
             "sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_span_len",
        default=4,
        type=int,
        help="The maximum word num of a span.",
    )

    parser.add_argument("--bert_lr", default=5e-5, type=float,
                        help="The initial learning rate for BERT.")
    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The initial learning rate of classifier.")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--warmup_steps", default=0.1, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=float, default=1.0,
                        help="Log every X updates steps.")

    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix "
             "as model_name ending and ending with step number",
    )
    # TODO, try different seed
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    return parser


def arg_process(args):
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda", args.gpu_id)
    args.device = device

    return args


def prepare_optimizer_scheduler(args, model, training_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    bert_parameters = eval('model.{}'.format(args.model_type)).named_parameters()

    other_parameters = list(model.bn_encoder.named_parameters()) \
                       + list(model.span_layer.named_parameters()) \
                       + list(model.span_classifier.named_parameters()) \
                       + list(model.oov_reg.named_parameters()) \
                       + list(model.z_reg.named_parameters())
    # + list(model.z_reg.named_parameters())

    optimizer_grouped_parameters = [
        # other params
        {"params": [p for n, p in other_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.lr},
        {"params": [p for n, p in other_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.lr},

        # bert params
        {"params": [p for n, p in bert_parameters if
                    not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.bert_lr},

        {"params": [p for n, p in bert_parameters if
                    any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.bert_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=training_steps
    )

    return optimizer, scheduler


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
def predictions_save(origin_file, predictions, output_file, labels):
    
    pred_label_idx = [x['pred_label_idx'].tolist() for x in predictions]
    all_span_idxs = [x['all_span_idxs'].tolist() for x in predictions]
    span_label_ltoken = [x['span_label_ltoken'].tolist() for x in predictions]
    
    # idx2label = {0:'O', 1:'ORG', 2:'PER', 3:'LOC', 4:'MISC'}
    idx2label = {k: v for k, v in enumerate(labels)}

    with open(output_file, "w") as writer:
        with open(origin_file, "r") as f:
            words = []
            lines = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append(words)
                        words = []
                else:
                    splits = line.split()
                    if len(splits) not in [2, 4]:
                        print('line segs num {}'.format(len(splits)))
                        print(line)
                    words.append(splits[0])
            if words:
                lines.append(words)

            cnt = 0
            for i in range(len(pred_label_idx)):
                for span_idxs, lps, lts in zip(all_span_idxs[i], pred_label_idx[i], span_label_ltoken[i]):
                    word_list = (lines[cnt])[:]

                    for sid, lp,lt in zip(span_idxs, lps, lts):
                        if lp != 0 or lt != 0:
                            plabel = idx2label[int(lp)]
                            tlabel = idx2label[int(lt)]
                            sidx, eidx = sid
                            for k in range(int(sidx), int(eidx) + 1):
                                word_list[k] = word_list[k] + ' ' + tlabel + ' ' + plabel
                    for k in range(len(word_list)):
                        if ' ' not in word_list[k]:
                            word_list[k] = word_list[k] + ' O O'
                                         
                    text = '\n'.join(word_list)
                    writer.write(text + '\n\n')
                    cnt += 1
            
            f.close()
        writer.close()

