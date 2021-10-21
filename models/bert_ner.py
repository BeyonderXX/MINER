import torch
import torch.nn as nn
from transformers.modeling_bert import BertModel
from transformers.modeling_bert import BertPreTrainedModel

from .losses.crf import CRF
from .model_utils import valid_sequence_output


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.decoder = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            valid_mask=None,
            labels=None,
            decode=False,
            skip_label=-1
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]

        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if decode:
            tags = self.decoder.decode(logits, attention_mask)
            outputs = (tags,)
        else:
            outputs = (logits,)

        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.decoder(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (skip_label * loss,) + outputs

        return outputs  # (loss), scores


class BertSpanNER(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.decoder = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            valid_mask=None,
            labels=None,
            decode=False,
            skip_label=-1
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if decode:
            tags = self.decoder.decode(logits, attention_mask)
            outputs = (tags,)
        else:
            outputs = (logits,)

        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.decoder(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (skip_label * loss,) + outputs

        return outputs  # (loss), scores