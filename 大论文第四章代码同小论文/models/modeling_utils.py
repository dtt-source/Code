import torch
import torch.nn as nn
from transformers import PreTrainedModel
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple, Union, Dict

@dataclass
class BaseModelOutput:
    logits: Any = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    loss: Any = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def Bertpooling(bert_outputs, pool_type='pooler'):
    encoded_layers = bert_outputs.hidden_states
    sequence_output = bert_outputs.last_hidden_state

    if pool_type == 'pooler':
        pooled_output = bert_outputs.pooler_output

    elif pool_type == 'first-last-avg':
        first = encoded_layers[1]
        last = encoded_layers[-1]
        seq_length = first.size(1)

        first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  
        last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  
        pooled_output = torch.avg_pool1d(
            torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2),
            kernel_size=2).squeeze(-1)

    elif pool_type == 'last-avg':
        pooled_output = torch.mean(sequence_output, 1)

    elif pool_type == 'cls':
        pooled_output = sequence_output[:, 0]

    else:
        raise TypeError('Please the right pool_type from cls, pooler, first-last-avg and last-avg !!!')

    return pooled_output




class PreTrainedModelWrapper(nn.Module):
    r"""
    A wrapper class around a (`transformers.PreTrainedModel`) to be compatible with the
    (`~transformers.PreTrained`) class in order to keep some attributes and methods of the
    (`~transformers.PreTrainedModel`) class.

    Attributes
    ----------
    pretrained_model: (`transformers.PreTrainedModel`)
        The model to be wrapped.
    parent_class: (`transformers.PreTrainedModel`)
        The parent class of the model to be wrapped.
    supported_args: (`list`)
        The list of arguments that are supported by the wrapper class.
    """
    transformers_parent_class = None
    supported_args = None

    def __init__(self, pretrained_model=None, **kwargs):
        super().__init__()
        self.pretrained_model = pretrained_model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model.

        Parameters
        ----------
        pretrained_model_name_or_path: (`str` or `transformers.PreTrainedModel`)
            The path to the pretrained model or its name.
        *model_args:
            Additional positional arguments passed along to the underlying model's
            `from_pretrained` method.
        **kwargs:
            Additional keyword arguments passed along to the underlying model's
            `from_pretrained` method. We also pre-process the kwargs to extract
            the arguments that are specific to the `transformers.PreTrainedModel`
            class and the arguments that are specific to trl models.
        """


        
        
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model = cls.transformers_parent_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        elif isinstance(pretrained_model_name_or_path, PreTrainedModel):
            pretrained_model = pretrained_model_name_or_path
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string or a PreTrainedModel, "
                f"but is {type(pretrained_model_name_or_path)}"
            )

        
        model = cls(pretrained_model)

        return model

    @classmethod
    def _split_kwargs(cls, kwargs):
        """
        Separate the kwargs from the arguments that we support inside
        `supported_args` and the ones that we don't.
        """
        supported_kwargs = {}
        unsupported_kwargs = {}

        for key, value in kwargs.items():
            if key in cls.supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value

        return supported_kwargs, unsupported_kwargs

    def push_to_hub(self, *args, **kwargs):
        r"""
        Push the pretrained model to the hub.
        """
        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        r"""
        Save the pretrained model to a directory.
        """
        return self.pretrained_model.save_pretrained(*args, **kwargs)




