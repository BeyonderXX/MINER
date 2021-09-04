# coding=utf-8
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
""" Fine-tuning the library models for named entity recognition
on CoNLL-2003 (Bert or Roberta). """

import sys

sys.path.append('../')

from utils.utils_metrics import get_entities_bio, f1_score, classification_report
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer

from models.model_ner import AutoModelForCrfNer

from utils.utils_ner import get_labels, collate_fn
from engine_utils import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]
trans_list = ["CrossCategory", "EntityTyposSwap", "OOV", "ToLonger"]


def train(args, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    train_dataset = load_and_cache_examples(args, tokenizer, labels,
                                            pad_token_label_id, mode="train")
    tb_writer = SummaryWriter(args.output_dir)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=collate_fn)
    training_steps = len(train_dataloader) * args.epoch

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = prepare_optimizer_scheduler(args, model, training_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Total train batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", training_steps)

    global_step = 0
    best_score = 0.0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.epoch), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "valid_mask": batch[2],
                      # RoBERTa don"t use segment_ids
                      "token_type_ids": batch[3],
                      "labels": batch[4]
                      }
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if global_step % len(train_dataloader) == 0:
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / len(train_dataloader), global_step)
                logging_loss = tr_loss

                if args.evaluate_during_training:
                    results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                          prefix=global_step)
                    for key, value in results.items():
                        if isinstance(value, float) or isinstance(value, int):
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                    if best_score < results['f1']:
                        best_score = results['f1']
                        output_dir = os.path.join(args.output_dir, "best_checkpoint")
                        model_save(args, output_dir, model, tokenizer)

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode,
             prefix='', data_dir=None):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels,
                                           pad_token_label_id,
                                           mode=mode, data_dir=data_dir)
    args.eval_batch_size = args.batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation {0} {1} *****".format(mode, prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    preds = None
    trues = None
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "valid_mask": batch[2],
                      # RoBERTa don"t use segment_ids
                      "token_type_ids": batch[3],
                      "decode": True
                      }
            outputs = model(**inputs)
            tags = outputs[0]

        nb_eval_steps += 1

        if preds is None:
            preds = tags.detach().cpu().numpy()
            trues = batch[4].detach().cpu().numpy()
        else:
            preds = np.append(preds, tags.detach().cpu().numpy(), axis=0)
            trues = np.append(trues, batch[4].detach().cpu().numpy(), axis=0)

    label_map = {i: label for i, label in enumerate(labels)}
    trues_list = [[] for _ in range(trues.shape[0])]
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(trues.shape[0]):
        for j in range(trues.shape[1]):
            if trues[i, j] != pad_token_label_id:
                trues_list[i].append(label_map[trues[i][j]])
                # 超长的 case ， 默认剩余长度预测为 O
                if preds[i][j] != -1:
                    preds_list[i].append(label_map[preds[i][j]])
                else:
                    preds_list[i].append("O")

    true_entities = get_entities_bio(trues_list)
    pred_entities = get_entities_bio(preds_list)

    results = {
        "f1": f1_score(true_entities, pred_entities),
        'report': classification_report(true_entities, pred_entities)
    }

    # save metrics result
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {0} {1} *****".format(mode, prefix))
        writer.write("***** Eval results {0} {1} *****\n".format(mode, prefix))
        for key in sorted(results.keys()):
            if key == 'report_dict':
                continue
            logger.info("{} = {}".format(key, str(results[key])))
            writer.write("{} = {}\n".format(key, str(results[key])))

    return results, preds_list


# main parameters
def get_args():
    parser = arg_parse()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        # default="/root/RobustNER/data/conll2003/origin/",
        default="/root/RobustNER/data/debug/",
        type=str,
        help="The input data dir. Should contain the training files for the "
             "CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--output_dir",
        default="/root/RobustNER/out/bert_uncase/bn_5e_5/",
        type=str,
        help="The output directory where the model predictions and "
             "checkpoints will be written.",
    )
    # Other parameters
    parser.add_argument("--gpu_id", default=3, type=int,
                        help="GPU number id")

    parser.add_argument(
        "--epoch", default=50, type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--hidden_dim", default=300, type=int,
        help="post encoder out dim"
    )

    parser.add_argument(
        "--beta", default=5e-5, type=float,
        help="beta params."
    )

    parser.add_argument(
        "--sample_size", default=5, type=int,
        help="sample num from p(z|x)"
    )

    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_robustness_eval", action="store_true",
                        help="Whether to evaluate robustness")

    parser.add_argument(
        "--evaluate_during_training",
        default=True,
        help="Whether to run evaluation during training at each logging step.",
    )

    return arg_process(parser.parse_args())


def fast_evaluate(args, ckpt_dir, tokenizer_args, labels, pad_token_label_id,
                  mode, prefix='', data_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, **tokenizer_args)
    model = AutoModelForCrfNer.from_pretrained(
        ckpt_dir,
        args=args,
        baseline=args.baseline
    )
    model.to(args.device)
    results, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id,
                          mode=mode, prefix=prefix, data_dir=data_dir)
    output_eval_file = os.path.join(ckpt_dir, "{0}_results.txt".format(mode))

    with open(output_eval_file, "a") as writer:
        writer.write('***** Predict in {0} {1} dataset *****\n'.format(mode, prefix))
        writer.write("{} = {}\n".format('report', str(results['report'])))

    return results, predictions


def main():
    args = get_args()
    set_seed(args) # Added here for reproductibility

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("Process device: %s", args.device)

    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    num_labels = len(labels)

    # Use cross entropy ignore index as padding label id
    pad_token_label_id = CrossEntropyLoss().ignore_index
    args.model_type = args.model_type.lower()

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=None
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

    # 可以只加载 预训练 模型, 也可以加载全部保存的参数
    model = AutoModelForCrfNer.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=None,
        args=args,
        baseline=args.baseline
    )

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        # Saving final-practices: if you use defaults names for the model,
        # you can reload it using from_pretrained()
        model_save(args, args.output_dir, model, tokenizer)

    best_ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")

    # Evaluation
    if args.do_eval:
        # eval best checkpoint
        fast_evaluate(args, best_ckpt_dir, tokenizer_args, labels,
                      pad_token_label_id, mode="dev", prefix="best ckpt")

    if args.do_predict:
        # eval final checkpoint
        fast_evaluate(args, args.output_dir, tokenizer_args, labels,
                      pad_token_label_id, mode="test", prefix="final ckpt")
        # eval best checkpoint
        results, predictions = fast_evaluate(
            args, best_ckpt_dir, tokenizer_args, labels, pad_token_label_id,
            mode="test", prefix="best ckpt")

        # Save predictions
        test_file = os.path.join(args.data_dir, "test.txt")
        output_test_predictions = os.path.join(best_ckpt_dir, "test_predictions.txt")
        predictions_save(test_file, predictions, output_test_predictions)

    if args.do_robustness_eval:
        base_dir = '/root/RobustNER/data/conll2003/'
        # eval best checkpoint
        for trans in trans_list:
            logger.info(
                "Start evaluate Robustness of {0} transformation".format(trans)
            )
            trans_dir = os.path.join(base_dir, trans, "trans")
            assert os.path.exists(trans_dir)
            results, predictions = fast_evaluate(
                args, best_ckpt_dir, tokenizer_args, labels, pad_token_label_id,
                mode="test", prefix="best ckpt {0}".format(trans),
                data_dir=trans_dir
            )
            # Save predictions
            test_file = os.path.join(trans_dir, "test.txt")
            out_trans_predictions = os.path.join(
                best_ckpt_dir, "{0}_predictions.txt".format(trans)
            )
            predictions_save(test_file, predictions, out_trans_predictions)

            logger.info(
                "Finish evaluate Robustness of {0} transformation".format(trans)
            )


if __name__ == "__main__":
    main()
