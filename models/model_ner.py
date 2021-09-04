from collections import OrderedDict

from transformers import BertConfig, RobertaConfig
from transformers import AutoConfig, PretrainedConfig

from models.bert_ner import BertCrfForNer
from models.bn_bert_ner import BertCrfWithBN
from models.roberta_ner import RobertaCrfForNer


MODEL_FOR_CRF_NER_MAPPING = OrderedDict(


    [
        (RobertaConfig, RobertaCrfForNer),
        (BertConfig, BertCrfWithBN)
    ]
)





class AutoModelForCrfNer:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_CRF_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_CRF_NER_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        baseline=False, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # for config_class, model_class in MODEL_FOR_CRF_NER_MAPPING.items():
        #     if isinstance(config, config_class):
        #         # roberta&crf 继承自 transformers BERT基类，其依据什么加载对应模型？
        #         # 通过 roberta&crf 成员 RobertaModel 实例的 base_model_prefix 与 已保存参数名对应
        #         return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        # default BertCrfWithBN
        if baseline:
            return BertCrfForNer.from_pretrained(pretrained_model_name_or_path,
                                                 *model_args, config=config,
                                                 **kwargs)
        else:
            return BertCrfWithBN.from_pretrained(pretrained_model_name_or_path,
                                                 *model_args, config=config,
                                                 **kwargs)

        # raise ValueError(
        #     "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
        #     "Model type should be one of {}.".format(
        #         config.__class__,
        #         cls.__name__,
        #         ", ".join(c.__name__ for c in MODEL_FOR_CRF_NER_MAPPING.keys()),
        #     )
        # )
