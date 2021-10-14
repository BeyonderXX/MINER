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

import fitlog
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
trans_list = ["CrossCategory", "EntityTyposSwap", "OOV"]


def get_args():
    parser = arg_parse()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="/root/RobustNER/data/conll2003/origin/",
        # default="/root/RobustNER/data/debug/",
        type=str,
        help="The input data dir. Should contain the training files for the "
             "CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--output_dir",
        default="/root/RobustNER/out/bert_uncase/debug/",
        type=str,
        help="The output directory where the model predictions and "
             "checkpoints will be written.",
    )

    parser.add_argument(
        "--mode", default='bn', type=str,
        help="bn for information bottleneck, oov for out of distribution, "
             "cc for cross category."
    )

    # Other parameters
    parser.add_argument("--gpu_id", default=3, type=int,
                        help="GPU number id")

    parser.add_argument(
        "--epoch", default=10, type=float,
        help="Total number of training epochs to perform."
    )

    parser.add_argument(
        "--hidden_dim", default=50, type=int,
        help="bottleneck encoder out dim"
    )

    parser.add_argument(
        "--beta", default=0, type=float,
        help="weights of p(z|v) norm regular, 0 represents not use"
    )

    parser.add_argument(
        "--mi_estimator", default='VIB', type=str,
        help="MI estimator for I(X;Z), support VIB, CLUB"
    )

    # I(Z; Y) parameters
    parser.add_argument(
        "--sample_size", default=5, type=int,
        help="sample num from p(z|x)"
    )

    parser.add_argument(
        "--alpha", default=1e-3, type=float,
        help="weights of kl div of p(y|v) and p(y|z)"
    )

    parser.add_argument(
        "--gama", default=1e-3, type=float,
        help="weights of oov regular"
    )

    parser.add_argument(
        "--theta", default=1e-3, type=float,
        help="weights of cross category regular"
    )

    # negative sample build mode
    parser.add_argument(
        "--rep_mode", default="typos",
        help="which strategy to replace entity, support typos and ngram now."
    )

    parser.add_argument(
        "--pmi_json", default='/root/RobustNER/data/conll2003/pmi.json'
    )
    parser.add_argument(
        "--pmi_preserve", default=0.3, type=float,
        help="what extent of PMI ranking subword preserve."
    )

    # training parameters
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


def train(args, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    train_examples = load_and_cache_examples(args, mode="train")
    tb_writer = SummaryWriter(args.output_dir)
    training_steps = (len(train_examples) - 1 / args.epoch + 1) * args.epoch

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = prepare_optimizer_scheduler(args, model, training_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Total train batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", training_steps)

    global_step = 0
    best_score = 0.0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.epoch), desc="Epoch")
    epoch_num = 0

    for _ in train_iterator:
        epoch_num += 1
        # input_ids, input_mask, valid_mask, segment_ids, label_ids
        # neg(input_ids, input_mask, valid_mask, segment_ids, label_ids)
        train_dataset = get_ds_features(args, train_examples, tokenizer,
                                        labels, pad_token_label_id)

        # mask mode
        # train_dataset = get_mask_ds_features(args, train_examples, tokenizer,
        #                                      labels, pad_token_label_id)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=args.batch_size,
                                      collate_fn=collate_fn)
        logger.info("Training epoch num {0}".format(epoch_num))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'origin_features': {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "valid_mask": batch[2],
                    "token_type_ids": batch[3],
                    "labels": batch[4]
                }
            }

            if args.mode == 'oov':
                inputs['switched_features'] = {
                    "input_ids": batch[5],
                    "attention_mask": batch[6],
                    "valid_mask": batch[7],
                    "token_type_ids": batch[8],
                    "labels": batch[9]
                }

            if args.mode not in ['bn', 'oov']:
                raise Exception('Unsupported mode {}'.format(args.mode))

            outputs = model(**inputs)
            loss_dic = outputs[0]
            loss = loss_dic['loss']
            loss.backward()

            fitlog.add_loss(loss.tolist(), name="Loss", step=global_step)
            tr_loss += loss.item()
            description = "".join(["{0}:{1}, ".format(k, round(v.item(), 3)) for k, v in loss_dic.items()]).strip(', ')

            epoch_iterator.set_description(description)
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
                                          prefix="{}".format(global_step))
                    weighted_score = results['f1']

                    # TODO, add dev mode
                    if args.do_robustness_eval and epoch_num > 5:
                        robust_f1 = robust_evaluate(args, args.output_dir, None, labels,
                            pad_token_label_id, prefix="{}".format(global_step), model=model, tokenizer=tokenizer)
                        # select best ckpt base weighted f1 score
                        weighted_score = 0.35 * results['f1'] + 0.65 * robust_f1
                    else:
                        weighted_score *= weighted_score

                    for key, value in results.items():
                        if isinstance(value, float) or isinstance(value, int):
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                    if best_score < weighted_score:
                        best_score = weighted_score
                        output_dir = os.path.join(args.output_dir, "best_checkpoint")
                    else:
                        output_dir = args.output_dir
                    model_save(args, output_dir, model, tokenizer)

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode,
             prefix='', data_dir=None):
    eval_examples = load_and_cache_examples(args, mode=mode, data_dir=data_dir)

    eval_dataset = get_ds_features(args, eval_examples, tokenizer, labels,
                               pad_token_label_id)

    # accelerate evaluation speed
    args.eval_batch_size = 256
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
            inputs = {
                "origin_features": {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "valid_mask": batch[2],
                    "token_type_ids": batch[3],

                }
            }
            # without labels, direct out tags
            outputs = model(decode=True, **inputs)
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


# return average f1 in various robust testset
def robust_evaluate(args, ckpt_dir, tokenizer_args, labels, pad_token_label_id,
                    prefix="best ckpt", model=None, tokenizer=None):
    base_dir = '/root/RobustNER/data/conll2003/v0/'
    robust_f1 = 0

    # eval best checkpoint
    for trans in trans_list:
        logger.info(
            "Start evaluate Robustness of {0} {1} transformation".format(prefix, trans)
        )
        trans_dir = os.path.join(base_dir, trans, "trans")
        assert os.path.exists(trans_dir)
        results, predictions = fast_evaluate(
            args, ckpt_dir, tokenizer_args, labels, pad_token_label_id,
            mode="test", prefix="{0} {1}".format(prefix, trans),
            data_dir=trans_dir, model=model, tokenizer=tokenizer
        )
        fitlog.add_metric(
            {"test": {"{0}_{1}_f1".format(prefix, trans): results["f1"]}},
            step=0
        )
        robust_f1 += results["f1"]

        # Save predictions
        if prefix == "best ckpt":
            test_file = os.path.join(trans_dir, "test.txt")
            out_trans_predictions = os.path.join(
                ckpt_dir, "{0}_{1}_predictions.txt".format(prefix, trans)
            )
            predictions_save(test_file, predictions, out_trans_predictions)

            logger.info(
                "Finish evaluate Robustness of {0} {1} transformation".format(prefix, trans)
            )

    return robust_f1 / len(trans_list)


def fast_evaluate(args, ckpt_dir, tokenizer_args, labels, pad_token_label_id,
                  mode, prefix='', model=None, tokenizer=None, data_dir=None):
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, **tokenizer_args)

    if not model:
        model = AutoModelForCrfNer.from_pretrained(
            ckpt_dir,
            args=args
        )
        model.to(args.device)

    results, predictions = evaluate(args, model, tokenizer, labels,
                                    pad_token_label_id, mode=mode,
                                    prefix=prefix, data_dir=data_dir)
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
        # args.output_dir,
        args.model_name_or_path,
        config=config,
        cache_dir=None,
        args=args
    )

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    fitlog.add_hyper(args)  # 通过这种方式记录ArgumentParser的参数
    fitlog.add_hyper_in_file(__file__)  # 记录本文件中写死的超参数

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    best_ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")

    # Evaluation
    if args.do_eval:
        # eval best checkpoint
        fast_evaluate(args, best_ckpt_dir, tokenizer_args, labels,
                      pad_token_label_id, mode="dev", prefix="best ckpt")

    if args.do_predict:
        # eval final checkpoint
        results, _ = fast_evaluate(args, args.output_dir,
                                   tokenizer_args, labels,
                                   pad_token_label_id,
                                   mode="test", prefix="final ckpt")
        fitlog.add_metric({"test": {"final_ckpt_f1": results["f1"]}}, step=0)

        # eval best checkpoint
        results, predictions = fast_evaluate(
            args, best_ckpt_dir, tokenizer_args, labels, pad_token_label_id,
            mode="test", prefix="best ckpt")
        fitlog.add_metric({"test": {"best_ckpt_f1": results["f1"]}}, step=0)

        # Save predictions
        test_file = os.path.join(args.data_dir, "test.txt")
        output_test_predictions = os.path.join(best_ckpt_dir, "test_predictions.txt")
        predictions_save(test_file, predictions, output_test_predictions)

    if args.do_robustness_eval:
        # eval best checkpoint
        robust_evaluate(args, best_ckpt_dir, tokenizer_args, labels,
                        pad_token_label_id, prefix="best ckpt")


if __name__ == "__main__":
    fitlog.commit(__file__)  # auto commit your codes
    fitlog.set_log_dir('logs/')  # set the logging directory
    fitlog.add_hyper_in_file(__file__)  # record your hyper-parameters

    main()
    fitlog.finish()
