import torch
from torch.nn.parameter import Parameter
import logging
from pathlib import Path
import math
import os
from typing import Any
from urllib.parse import urlparse
from typing import Union, Optional, List
from torch.nn.functional import embedding
from collections import defaultdict

from typing import (
    Union
)

from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Any,
)


class CustomDetHash:
    def det_hash_object(self) -> Any:
        """
        By default, `det_hash()` pickles an object, and returns the hash of the pickled
        representation. Sometimes you want to take control over what goes into
        that hash. In that case, implement this method. `det_hash()` will pickle the
        result of this method instead of the object itself.
        If you return `None`, `det_hash()` falls back to the original behavior and pickles
        the object.
        """
        raise NotImplementedError()


T = TypeVar("T", bound="FromParams")

_SubclassRegistry = Dict[str, Tuple[type, Optional[str]]]
_registry = defaultdict(dict)
_T = TypeVar("_T")


def register(
        name: str, constructor: Optional[str] = None,
        exist_ok: bool = False
) -> Callable[[Type[_T]], Type[_T]]:
    """
    Register a class under a particular name.
    # Parameters
    name : `str`
        The name to register the class under.
    constructor : `str`, optional (default=`None`)
        The name of the method to use on the class to construct the object.  If this is given,
        we will use this method (which must be a `@classmethod`) instead of the default
        constructor.
    exist_ok : `bool`, optional (default=`False`)
        If True, overwrites any existing models registered under `name`. Else,
        throws an error if a model is already registered under `name`.
    # Examples
    To use this class, you would typically have a base class that inherits from `Registrable`:
    ```python
    class Vocabulary(Registrable):
        ...
    ```
    Then, if you want to register a subclass, you decorate it like this:
    ```python
    @Vocabulary.register("my-vocabulary")
    class MyVocabulary(Vocabulary):
        def __init__(self, param1: int, param2: str):
            ...
    ```
    Registering a class like this will let you instantiate a class from a config file, where you
    give `"type": "my-vocabulary"`, and keys corresponding to the parameters of the `__init__`
    method (note that for this to work, those parameters must have type annotations).
    If you want to have the instantiation from a config file call a method other than the
    constructor, either because you have several different construction paths that could be
    taken for the same object (as we do in `Vocabulary`) or because you have logic you want to
    happen before you get to the constructor (as we do in `Embedding`), you can register a
    specific `@classmethod` as the constructor to use, like this:
    ```python
    @Vocabulary.register("my-vocabulary-from-instances", constructor="from_instances")
    @Vocabulary.register("my-vocabulary-from-files", constructor="from_files")
    class MyVocabulary(Vocabulary):
        def __init__(self, some_params):
            ...
        @classmethod
        def from_instances(cls, some_other_params) -> MyVocabulary:
            ...  # construct some_params from instances
            return cls(some_params)
        @classmethod
        def from_files(cls, still_other_params) -> MyVocabulary:
            ...  # construct some_params from files
            return cls(some_params)
    ```
    """
    registry = _registry[name]

    def add_subclass_to_registry(subclass: Type[_T]) -> Type[_T]:
        # Add to registry, raise an error if key has already been used.
        if name in registry:
            if exist_ok:
                message = (
                    f"{name} has already been registered as {registry[name][0].__name__}, but "
                    f"exist_ok=True, so overwriting with {name}"
                )
                logging.info(message)
            else:
                message = (
                    f"Cannot register {name} as {name}; "
                    f"name already in use for {registry[name][0].__name__}"
                )
                raise Exception(message)
        registry[name] = (subclass, constructor)
        return subclass

    return add_subclass_to_registry


class TokenEmbedder(torch.nn.Module):
    """
    A `TokenEmbedder` is a `Module` that takes as input a tensor with integer ids that have
    been output from a [`TokenIndexer`](/api/data/token_indexers/token_indexer.md) and outputs
    a vector per token in the input.  The input typically has shape `(batch_size, num_tokens)`
    or `(batch_size, num_tokens, num_characters)`, and the output is of shape `(batch_size, num_tokens,
    output_dim)`.  The simplest `TokenEmbedder` is just an embedding layer, but for
    character-level input, it could also be some kind of character encoder.
    We add a single method to the basic `Module` API: `get_output_dim()`.  This lets us
    more easily compute output dimensions for the
    [`TextFieldEmbedder`](/api/modules/text_field_embedders/text_field_embedder.md),
    which we might need when defining model parameters such as LSTMs or linear layers, which need
    to know their input dimension before the layers are called.
    """

    default_implementation = "embedding"

    def get_output_dim(self) -> int:
        """
        Returns the final output dimension that this `TokenEmbedder` uses to represent each
        token.  This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError


class TimeDistributed(torch.nn.Module):
    """
    Given an input shaped like `(batch_size, time_steps, [rest])` and a `Module` that takes
    inputs like `(batch_size, [rest])`, `TimeDistributed` reshapes the input to be
    `(batch_size * time_steps, [rest])`, applies the contained `Module`, then reshapes it back.
    Note that while the above gives shapes with `batch_size` first, this `Module` also works if
    `batch_size` is second - we always just combine the first two dimensions, then split them.
    It also reshapes keyword arguments unless they are not tensors or their name is specified in
    the optional `pass_through` iterable.
    """

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *inputs, pass_through: List[str] = None, **kwargs):

        pass_through = pass_through or []

        reshaped_inputs = [self._reshape_tensor(input_tensor) for input_tensor in inputs]

        # Need some input to then get the batch_size and time_steps.
        some_input = None
        if inputs:
            some_input = inputs[-1]

        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value

                value = self._reshape_tensor(value)

            reshaped_kwargs[key] = value

        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)

        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")

        # Now get the output back into the right shape.
        # (batch_size, time_steps, **output_size)
        tuple_output = True
        if not isinstance(reshaped_outputs, tuple):
            tuple_output = False
            reshaped_outputs = (reshaped_outputs,)

        outputs = []
        for reshaped_output in reshaped_outputs:
            new_size = some_input.size()[:2] + reshaped_output.size()[1:]
            outputs.append(reshaped_output.contiguous().view(new_size))

        if not tuple_output:
            outputs = outputs[0]

        return outputs

    @staticmethod
    def _reshape_tensor(input_tensor):
        input_size = input_tensor.size()
        if len(input_size) <= 2:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        # Squash batch_size and time_steps into a single axis; result has shape
        # (batch_size * time_steps, **input_size).
        squashed_shape = [-1] + list(input_size[2:])
        return input_tensor.contiguous().view(*squashed_shape)


class Embedding(TokenEmbedder):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to:
        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding
        4. project the resultant embeddings to some other dimension (which only makes sense with
           non-trainable embeddings).
    Note that if you are using our data API and are trying to embed a
    [`TextField`](../../data/fields/text_field.md), you should use a
    [`TextFieldEmbedder`](../text_field_embedders/text_field_embedder.md) instead of using this directly.
    Registered as a `TokenEmbedder` with name "embedding".
    # Parameters
    num_embeddings : `int`
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim : `int`
        The size of each embedding vector.
    projection_dim : `int`, optional (default=`None`)
        If given, we add a projection layer after the embedding layer.  This really only makes
        sense if `trainable` is `False`.
    weight : `torch.FloatTensor`, optional (default=`None`)
        A pre-initialised weight matrix for the embedding lookup, allowing the use of
        pretrained vectors.
    padding_index : `int`, optional (default=`None`)
        If given, pads the output with zeros whenever it encounters the index.
    trainable : `bool`, optional (default=`True`)
        Whether or not to optimize the embedding parameters.
    max_norm : `float`, optional (default=`None`)
        If given, will renormalize the embeddings to always have a norm lesser than this
    norm_type : `float`, optional (default=`2`)
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : `bool`, optional (default=`False`)
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse : `bool`, optional (default=`False`)
        Whether or not the Pytorch backend should use a sparse representation of the embedding weight.
    vocab_namespace : `str`, optional (default=`None`)
        In case of fine-tuning/transfer learning, the model's embedding matrix needs to be
        extended according to the size of extended-vocabulary. To be able to know how much to
        extend the embedding-matrix, it's necessary to know which vocab_namspace was used to
        construct it in the original training. We store vocab_namespace used during the original
        training as an attribute, so that it can be retrieved during fine-tuning.
    pretrained_file : `str`, optional (default=`None`)
        Path to a file of word vectors to initialize the embedding matrix. It can be the
        path to a local file or a URL of a (cached) remote file. Two formats are supported:
            * hdf5 file - containing an embedding matrix in the form of a torch.Tensor;
            * text file - an utf-8 encoded text file with space separated fields.
    vocab : `Vocabulary`, optional (default = `None`)
        Used to construct an embedding from a pretrained file.
        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "embedding", it gets specified as a top-level parameter, then is passed in to this module
        separately.
    # Returns
    An Embedding module.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = None,
        projection_dim: int = None,
        weight: torch.FloatTensor = None,
        padding_index: int = None,
        trainable: bool = True,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        vocab_namespace: str = "tokens",
    ) -> None:
        super().__init__()

        if num_embeddings is None:
            raise Exception(
                "Embedding must be constructed with either num_embeddings or a vocabulary."
            )

        _vocab_namespace: Optional[str] = vocab_namespace
        if num_embeddings is None:
            raise Exception(
                "Embedding must be constructed with either num_embeddings or a vocabulary."
            )
        else:
            # If num_embeddings is present, set default namespace to None so that extend_vocab
            # call doesn't misinterpret that some namespace was originally used.
            _vocab_namespace = None  # type: ignore

        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._vocab_namespace = _vocab_namespace

        self.output_dim = projection_dim or embedding_dim

        if weight is not None:
            raise Exception(
                "Embedding was constructed with both a weight and a pretrained file."
            )

        elif weight is not None:
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        else:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)

        # Whatever way we have constructed the embedding, it should be consistent with
        # num_embeddings and embedding_dim.
        if self.weight.size() != (num_embeddings, embedding_dim):
            raise Exception(
                "A weight matrix was passed with contradictory embedding shapes."
            )

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection = None

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass tokens to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = tokens.size()
        tokens = combine_initial_dims(tokens)

        embedded = embedding(
            tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Now (if necessary) add back in the extra dimensions.
        embedded = uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded


class SpanExtractor(torch.nn.Module):
    """
    Many NLP models deal with representations of spans inside a sentence.
    SpanExtractors define methods for extracting and representing spans
    from a sentence.
    SpanExtractors take a sequence tensor of shape (batch_size, timesteps, embedding_dim)
    and indices of shape (batch_size, num_spans, 2) and return a tensor of
    shape (batch_size, num_spans, ...), forming some representation of the
    spans.
    """

    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ):
        """
        Given a sequence tensor, extract spans and return representations of
        them. Span representation can be computed in many different ways,
        such as concatenation of the start and end spans, attention over the
        vectors contained inside the span, etc.
        # Parameters
        sequence_tensor : `torch.FloatTensor`, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : `torch.LongTensor`, required.
            A tensor of shape `(batch_size, num_spans, 2)`, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the `sequence_tensor`.
        sequence_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, sequence_length) representing padded
            elements of the sequence.
        span_indices_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, num_spans) representing the valid
            spans in the `indices` tensor. This mask is optional because
            sometimes it's easier to worry about masking after calling this
            function, rather than passing a mask directly.
        # Returns
        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
        """
        raise NotImplementedError

    def get_input_dim(self) -> int:
        """
        Returns the expected final dimension of the `sequence_tensor`.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the expected final dimension of the returned span representation.
        """
        raise NotImplementedError


class SpanExtractorWithSpanWidthEmbedding(SpanExtractor):
    """
    `SpanExtractorWithSpanWidthEmbedding` implements some common code for span
    extractors which will need to embed span width.
    Specifically, we initiate the span width embedding matrix and other
    attributes in `__init__`, leave an `_embed_spans` method that can be
    implemented to compute span embeddings in different ways, and in `forward`
    we concatenate span embeddings returned by `_embed_spans` with span width
    embeddings to form the final span representations.
    We keep SpanExtractor as a purely abstract base class, just in case someone
    wants to build a totally different span extractor.
    # Parameters
    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    # Returns
    span_embeddings : `torch.FloatTensor`.
        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
    """

    def __init__(
        self,
        input_dim: int,
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        self._span_width_embedding: Optional[Embedding] = None
        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(
                num_embeddings=num_width_embeddings, embedding_dim=span_width_embedding_dim
            )
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise Exception(
                "To use a span width embedding representation, you must"
                "specify both num_width_embeddings and span_width_embedding_dim."
            )

    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ):
        """
        Given a sequence tensor, extract spans, concatenate width embeddings
        when need and return representations of them.
        # Parameters
        sequence_tensor : `torch.FloatTensor`, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : `torch.LongTensor`, required.
            A tensor of shape `(batch_size, num_spans, 2)`, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the `sequence_tensor`.
        sequence_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, sequence_length) representing padded
            elements of the sequence.
        span_indices_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, num_spans) representing the valid
            spans in the `indices` tensor. This mask is optional because
            sometimes it's easier to worry about masking after calling this
            function, rather than passing a mask directly.
        # Returns
        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
        """
        # shape (batch_size, num_spans, embedding_dim)
        span_embeddings = self._embed_spans(
            sequence_tensor, span_indices, sequence_mask, span_indices_mask
        )
        if self._span_width_embedding is not None:
            # width = end_index - start_index + 1 since `SpanField` use inclusive indices.
            # But here we do not add 1 beacuse we often initiate the span width
            # embedding matrix with `num_width_embeddings = max_span_width`
            # shape (batch_size, num_spans)
            widths_minus_one = span_indices[..., 1] - span_indices[..., 0]

            if self._bucket_widths:
                widths_minus_one = bucket_values(
                    widths_minus_one, num_total_buckets=self._num_width_embeddings  # type: ignore
                )

            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(widths_minus_one)
            span_embeddings = torch.cat([span_embeddings, span_width_embeddings], -1)

        if span_indices_mask is not None:
            # Here we are masking the spans which were originally passed in as padding.
            return span_embeddings * span_indices_mask.unsqueeze(-1)

        return span_embeddings

    def get_input_dim(self) -> int:
        return self._input_dim

    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        """
        Returns the span embeddings computed in many different ways.
        """
        raise NotImplementedError


class EndpointSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    """
    Represents spans as a combination of the embeddings of their endpoints. Additionally,
    the width of the spans can be embedded and concatenated on to the final combination.
    The following types of representation are supported, assuming that
    `x = span_start_embeddings` and `y = span_end_embeddings`.
    `x`, `y`, `x*y`, `x+y`, `x-y`, `x/y`, where each of those binary operations
    is performed elementwise.  You can list as many combinations as you want, comma separated.
    For example, you might give `x,y,x*y` as the `combination` parameter to this class.
    The computed similarity function would then be `[x; y; x*y]`, which can then be optionally
    concatenated with an embedded representation of the width of the span.
    Registered as a `SpanExtractor` with name "endpoint".
    # Parameters
    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    combination : `str`, optional (default = `"x,y"`).
        The method used to combine the `start_embedding` and `end_embedding`
        representations. See above for a full description.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    use_exclusive_start_indices : `bool`, optional (default = `False`).
        If `True`, the start indices extracted are converted to exclusive indices. Sentinels
        are used to represent exclusive span indices for the elements in the first
        position in the sequence (as the exclusive indices for these elements are outside
        of the the sequence boundary) so that start indices can be exclusive.
        NOTE: This option can be helpful to avoid the pathological case in which you
        want span differences for length 1 spans - if you use inclusive indices, you
        will end up with an `x - x` operation for length 1 spans, which is not good.
    """

    def __init__(
        self,
        input_dim: int,
        combination: str = "x,y",
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
        use_exclusive_start_indices: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,
            span_width_embedding_dim=span_width_embedding_dim,
            bucket_widths=bucket_widths,
        )
        self._combination = combination

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            self._start_sentinel = Parameter(torch.randn([1, 1, int(input_dim)]))

    def get_output_dim(self) -> int:
        combined_dim = get_combined_dim(self._combination, [self._input_dim, self._input_dim])
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> None:
        # shape (batch_size, num_spans)
        span_starts, span_ends = [index.squeeze(-1) for index in span_indices.split(1, dim=-1)]

        if span_indices_mask is not None:
            # It's not strictly necessary to multiply the span indices by the mask here,
            # but it's possible that the span representation was padded with something other
            # than 0 (such as -1, which would be an invalid index), so we do so anyway to
            # be safe.
            span_starts = span_starts * span_indices_mask
            span_ends = span_ends * span_indices_mask

        if not self._use_exclusive_start_indices:
            if sequence_tensor.size(-1) != self._input_dim:
                raise ValueError(
                    f"Dimension mismatch expected ({sequence_tensor.size(-1)}) "
                    f"received ({self._input_dim})."
                )
            start_embeddings = batched_index_select(sequence_tensor, span_starts)
            end_embeddings = batched_index_select(sequence_tensor, span_ends)

        else:
            # We want `exclusive` span starts, so we remove 1 from the forward span starts
            # as the AllenNLP `SpanField` is inclusive.
            # shape (batch_size, num_spans)
            exclusive_span_starts = span_starts - 1
            # shape (batch_size, num_spans, 1)
            start_sentinel_mask = (exclusive_span_starts == -1).unsqueeze(-1)
            exclusive_span_starts = exclusive_span_starts * ~start_sentinel_mask.squeeze(-1)

            # We'll check the indices here at runtime, because it's difficult to debug
            # if this goes wrong and it's tricky to get right.
            if (exclusive_span_starts < 0).any():
                raise ValueError(
                    f"Adjusted span indices must lie inside the the sequence tensor, "
                    f"but found: exclusive_span_starts: {exclusive_span_starts}."
                )

            start_embeddings = batched_index_select(sequence_tensor, exclusive_span_starts)
            end_embeddings = batched_index_select(sequence_tensor, span_ends)

            # We're using sentinels, so we need to replace all the elements which were
            # outside the dimensions of the sequence_tensor with the start sentinel.
            start_embeddings = (
                start_embeddings * ~start_sentinel_mask + start_sentinel_mask * self._start_sentinel
            )

        combined_tensors = combine_tensors(
            self._combination, [start_embeddings, end_embeddings]
        )

        return combined_tensors


def _get_suggestion(name: str, available: List[str]) -> Optional[str]:
    # First check for simple mistakes like using '-' instead of '_', or vice-versa.
    for ch, repl_ch in (("_", "-"), ("-", "_")):
        suggestion = name.replace(ch, repl_ch)
        if suggestion in available:
            return suggestion
    # If we still haven't found a reasonable suggestion, we return the first suggestion
    # with an edit distance (with transpositions allowed) of 1 to `name`.
    return None


def is_url_or_existing_file(url_or_filename: Union[str, Path, None]) -> bool:
    """
    Given something that might be a URL (or might be a local path),
    determine check if it's url or an existing file path.
    """
    if url_or_filename is None:
        return False
    url_or_filename = os.path.expanduser(str(url_or_filename))
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https", "s3", "gs") or os.path.exists(url_or_filename)


def combine_initial_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a (possibly higher order) tensor of ids with shape
    (d1, ..., dn, sequence_length)
    Return a view that's (d1 * ... * dn, sequence_length).
    If original tensor is 1-d or 2-d, return it as is.
    """
    if tensor.dim() <= 2:
        return tensor
    else:
        return tensor.view(-1, tensor.size(-1))


def bucket_values(
    distances: torch.Tensor, num_identity_buckets: int = 4, num_total_buckets: int = 10
) -> torch.Tensor:
    """
    Places the given values (designed for distances) into `num_total_buckets`semi-logscale
    buckets, with `num_identity_buckets` of these capturing single values.
    The default settings will bucket values into the following buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    # Parameters
    distances : `torch.Tensor`, required.
        A Tensor of any size, to be bucketed.
    num_identity_buckets: `int`, optional (default = `4`).
        The number of identity buckets (those only holding a single value).
    num_total_buckets : `int`, (default = `10`)
        The total number of buckets to bucket values into.
    # Returns
    `torch.Tensor`
        A tensor of the same shape as the input, containing the indices of the buckets
        the values were placed in.
    """
    # Chunk the values into semi-logscale buckets using .floor().
    # This is a semi-logscale bucketing because we divide by log(2) after taking the log.
    # We do this to make the buckets more granular in the initial range, where we expect
    # most values to fall. We then add (num_identity_buckets - 1) because we want these indices
    # to start _after_ the fixed number of buckets which we specified would only hold single values.
    logspace_index = (distances.float().log() / math.log(2)).floor().long() + (
        num_identity_buckets - 1
    )
    # create a mask for values which will go into single number buckets (i.e not a range).
    use_identity_mask = (distances <= num_identity_buckets).long()
    use_buckets_mask = 1 + (-1 * use_identity_mask)
    # Use the original values if they are less than num_identity_buckets, otherwise
    # use the logspace indices.
    combined_index = use_identity_mask * distances + use_buckets_mask * logspace_index
    # Clamp to put anything > num_total_buckets into the final bucket.
    return combined_index.clamp(0, num_total_buckets - 1)


def get_combined_dim(combination: str, tensor_dims: List[int]) -> int:
    """
    For use with [`combine_tensors`](./util.md#combine_tensors).
    This function computes the resultant dimension when calling `combine_tensors(combination, tensors)`,
    when the tensor dimension is known.  This is necessary for knowing the sizes of weight matrices
    when building models that use `combine_tensors`.
    # Parameters
    combination : `str`
        A comma-separated list of combination pieces, like `"1,2,1*2"`, specified identically to
        `combination` in `combine_tensors`.
    tensor_dims : `List[int]`
        A list of tensor dimensions, where each dimension is from the `last axis` of the tensors
        that will be input to `combine_tensors`.
    """
    if len(tensor_dims) > 9:
        raise Exception("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    return sum(_get_combination_dim(piece, tensor_dims) for piece in combination.split(","))


def _get_combination_dim(combination: str, tensor_dims: List[int]) -> int:
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        if len(combination) != 3:
            raise Exception("Invalid combination: " + combination)
        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        if first_tensor_dim != second_tensor_dim:
            raise Exception('Tensor dims must match for operation "{}"'.format(operation))
        return first_tensor_dim


def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.
    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](https://docs.allennlp.org/models/main/models/coref/models/coref/)
    model to select contextual word representations corresponding to the start and end indices of
    mentions.
    The key reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.
    # Returns
    selected_targets : `torch.Tensor`
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.
    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```
    # Parameters
    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.
    # Returns
    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise Exception(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def combine_tensors(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Combines a list of tensors using element-wise operations and concatenation, specified by a
    `combination` string.  The string refers to (1-indexed) positions in the input tensor list,
    and looks like `"1,2,1+2,3-1"`.
    We allow the following kinds of combinations : `x`, `x*y`, `x+y`, `x-y`, and `x/y`,
    where `x` and `y` are positive integers less than or equal to `len(tensors)`.  Each of
    the binary operations is performed elementwise.  You can give as many combinations as you want
    in the `combination` string.  For example, for the input string `"1,2,1*2"`, the result
    would be `[1;2;1*2]`, as you would expect, where `[;]` is concatenation along the last
    dimension.
    If you have a fixed, known way to combine tensors that you use in a model, you should probably
    just use something like `torch.cat([x_tensor, y_tensor, x_tensor * y_tensor])`.  This
    function adds some complexity that is only necessary if you want the specific combination used
    to be `configurable`.
    If you want to do any element-wise operations, the tensors involved in each element-wise
    operation must have the same shape.
    This function also accepts `x` and `y` in place of `1` and `2` in the combination
    string.
    """
    if len(tensors) > 9:
        raise Exception("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    to_concatenate = [_get_combination(piece, tensors) for piece in combination.split(",")]
    return torch.cat(to_concatenate, dim=-1)


def _get_combination(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        if len(combination) != 3:
            raise Exception("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == "*":
            return first_tensor * second_tensor
        elif operation == "/":
            return first_tensor / second_tensor
        elif operation == "+":
            return first_tensor + second_tensor
        elif operation == "-":
            return first_tensor - second_tensor
        else:
            raise Exception("Invalid operation: " + operation)


def uncombine_initial_dims(tensor: torch.Tensor, original_size: torch.Size) -> torch.Tensor:
    """
    Given a tensor of embeddings with shape
    (d1 * ... * dn, sequence_length, embedding_dim)
    and the original shape
    (d1, ..., dn, sequence_length),
    return the reshaped tensor of embeddings with shape
    (d1, ..., dn, sequence_length, embedding_dim).
    If original size is 1-d or 2-d, return it as is.
    """
    if len(original_size) <= 2:
        return tensor
    else:
        view_args = list(original_size) + [tensor.size(-1)]
        return tensor.view(*view_args)
