import argparse
import time

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--a",
    default=1,
    # default="/root/RobustNER/data/conll_debug/",
    type=int
)

parser.add_argument(
    "--b",
    default=10,
    # default="/root/RobustNER/data/conll_debug/",
    type=int
)

parser.add_argument(
    "--c",
    default=11,
    type=int
)

# Required parameters
parser.add_argument(
    "--data_dir",
    default="/root/RobustNER/data/WNUT2017/origin/",
    # default="/root/RobustNER/data/conll_debug/",
    type=str,
    help="The input data dir. Should contain the training files for the "
         "CoNLL-2003 NER task.",
)

parser.add_argument(
    "--labels",
    default="/root/RobustNER/data/WNUT2017/labels.txt",
    type=str,
    help="Path to a file containing all labels. "
         "If not specified, CoNLL-2003 labels are used.",
)

parser.add_argument(
    "--output_dir",
    default="/root/RobustNER/out/bert_uncase/conll_debug/",
    type=str,
    help="The output directory where the model predictions and "
         "checkpoints will be written.",
)

# Other parameters
parser.add_argument("--gpu_id", default=6, type=int,
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
    "--trans_weight", default=1.0, type=float,
    help="weight of trans sample loss"
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
    "--pmi_json", default='/root/RobustNER/data/WNUT2017/pmi.json'
)

parser.add_argument(
    "--entity_json", default='/root/RobustNER/data/WNUT2017/entity.json'
)
# 0 means typos， 1 means switch
parser.add_argument(
    "--switch_ratio", default=0.5, type=float,
    help="Entity switch ratio."
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
# parser.add_argument("--do_robustness_eval", action="store_true",
#                     help="Whether to evaluate robustness")

while(True):
    time.sleep(0.1)
    print(111)
