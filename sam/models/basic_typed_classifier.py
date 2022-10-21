import logging
import time
from collections import OrderedDict
from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, FBetaMeasure

from sam.utils import flatten_dict

logger = logging.getLogger(__name__)


@Model.register("basic_typed_classifier")
class BasicTypedClassifier(Model):
    """
    This `Model` implements a basic text classifier. After embedding the text into
    a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The
    resulting sequence is pooled using a `Seq2VecEncoder` and then passed to
    a linear classification layer, which projects into the label space. If a
    `Seq2SeqEncoder` is not provided, we will pass the embedded text directly to the
    `Seq2VecEncoder`.

    Registered as a `Model` with name "basic_classifier".

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2seq_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    feedforward : `FeedForward`, optional, (default = `None`)
        An optional feedforward layer to apply after the seq2vec_encoder.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    num_labels : `int`, optional (default = `None`)
        Number of labels to project to in classification layer. By default, the classification layer will
        project to the size of the vocabulary namespace corresponding to labels.
    label_namespace : `str`, optional (default = `"labels"`)
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    DEFAULT_TYPE_NAMESPACE = "type_tags"
    DEFAULT_TAG_NAMESPACE = "adu_tags"
    DEFAULT_NONE_LABEL = "NONE"
    MICRO_F1_NAME = "micro-f1"

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Seq2SeqEncoder = None,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        none_label: str = DEFAULT_NONE_LABEL,
        type_namespace: str = DEFAULT_TYPE_NAMESPACE,
        type_embedding_size: Optional[int] = None,
        tag_namespace: str = DEFAULT_TAG_NAMESPACE,
        tag_embedding_size: Optional[int] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        embedded_text_cache_max: int = 0,
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace
        self._namespace = namespace
        self._type_namespace = type_namespace

        self._num_types = vocab.get_vocab_size(namespace=self._type_namespace)
        if type_embedding_size is None:
            self._type_embedding_size = self._num_types
        elif type_embedding_size < 0:
            self._type_embedding_size = self._num_types + type_embedding_size
            assert self._type_embedding_size >= 0, \
                f'if a negative type_embedding_size is used, the num_types will be added. ' \
                f'But num_types [{self._num_types}] + type_embedding_size [{type_embedding_size}] = ' \
                f'{self._type_embedding_size} has to be a positive value which it s not.'
        else:
            self._type_embedding_size = type_embedding_size
        if self._type_embedding_size != 0:
            self._type_embedding = torch.nn.Embedding(num_embeddings=self._num_types,
                                                      embedding_dim=self._type_embedding_size)
            logger.info(f'use type embedding of size: {self._num_types} -> {self._type_embedding_size}'
                        f'(types: {list(vocab.get_token_to_index_vocabulary(namespace=self._type_namespace))})')
        else:
            self._type_embedding = None

        self._tag_namespace = tag_namespace
        self._num_tags = vocab.get_vocab_size(namespace=self._tag_namespace)
        if tag_embedding_size is None:
            self._tag_embedding_size = self._num_tags
        else:
            self._tag_embedding_size = tag_embedding_size
        if self._tag_embedding_size != 0:
            self._tag_embedding = torch.nn.Embedding(num_embeddings=self._num_tags,
                                                     embedding_dim=self._tag_embedding_size)
            logger.info(f'use tag embedding of size: {self._num_tags} -> {self._tag_embedding_size} '
                        f'(tags: {list(vocab.get_token_to_index_vocabulary(namespace=self._tag_namespace))})')
        else:
            self._tag_embedding = None

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)

        self._none_label = none_label
        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            BasicTypedClassifier.MICRO_F1_NAME: FBetaMeasure(
                beta=1, average='micro',
                # exclude NONE labels
                labels=[k for k, v in vocab.get_index_to_token_vocabulary(namespace=self._label_namespace).items()
                        if v != self._none_label]
            )
        }
        for k, v in vocab.get_index_to_token_vocabulary(namespace=self._label_namespace).items():
            self._metrics[f'{v}'] = F1Measure(k)
        self._loss = torch.nn.CrossEntropyLoss()
        self._embedded_text_cache = OrderedDict()
        self.embedded_text_cache_max = embedded_text_cache_max
        initializer(self)

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None,
            types: torch.LongTensor = None,
            tags: torch.LongTensor = None,
            metadata = None,
            **kwargs,  # to allow for a more general dataset reader that passes args we don't need
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`
            From a `TextField`
        label : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`

        # Returns

        An output dictionary consisting of:

            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """

        #self.eval()

        #embedded_text_list = []
        #tokens_split = None
        #for i, mdata in enumerate(metadata):
        #    text_id = mdata['text_id']
        #    ts = time.time()
        #    last_access, _embedded_text = self._embedded_text_cache.get(text_id, (0, None))
        #    #_embedded_text = None
        #    if _embedded_text is None:
        #        if tokens_split is None:
        #            # split to individual instances (un-batch)
        #            tokens_split = TensorDict(tokens).split()
        #            assert len(tokens_split) == len(metadata), \
        #                f'number of instances in tokens (batch size) [{len(tokens_split)}] does not match number of ' \
        #                f'entries in metadata [{len(metadata)}]'
        #        #with self.eval():
        #        _embedded_text = self._text_field_embedder(tokens_split[i].d)
        #        self._embedded_text_cache[text_id] = (ts, _embedded_text)
        #    else:
        #        self._embedded_text_cache[text_id] = (ts, _embedded_text)
        #    embedded_text_list.append(_embedded_text)
        #
        #    self._embedded_text_cache = OrderedDict(sorted(self._embedded_text_cache.items(), key=lambda kv: kv[1][0],
        #                                                   reverse=True))
        #    while len(self._embedded_text_cache) > self.embedded_text_cache_max:
        #        self._embedded_text_cache.popitem(last=True)
        #
        #embedded_text = TensorDict.merge(embedded_text_list).d

        #embedded_text_dep = self._text_field_embedder(tokens)
        #embedded_text_dep2 = self._text_field_embedder(tokens)
        #assert embedded_text.equal(embedded_text_dep), 'not equal'
        #embedded_text_dep2 = self._text_field_embedder(tokens)
        #assert embedded_text_dep.equal(embedded_text_dep2), 'not equal (output)'
        #tokens_multi = [TensorDict.merge([tokens_split[0]] * (i + 1)).d for i in range(10)]
        #embedded_text_multi = [self._text_field_embedder(tm) for tm in tokens_multi]
        #for i, e in enumerate(embedded_text_multi):
        #    for j, f in enumerate(embedded_text_multi):
        #        for k in range(min(len(e), len(f))):
        #            print(i, j, k, e[k].equal(f[k]))
        #embedded_text_dep3 = self._text_field_embedder(tokens)
        ##embedded_text = self._text_field_embedder(tokens)
        ## embedded_text = self._cache(self._text_field_embedder, tokens)
        ##embedded_text = embedded_text.d
        #assert embedded_text.equal(embedded_text_dep), 'not equal (output)'

        #asd
        #print("\nXXXXXXXXXXXXXXXXXXXXx\n")
        #for i, e in enumerate(tokens_multi):
        #    for j, f in enumerate(tokens_multi):
        #        for k in range(min(i, j)):
        #            print(i, j, k, TensorDict(e).split()[k] == TensorDict(f).split()[k])

        key = self.training, tuple([md['text_id'] for md in metadata])

        last_access, embedded_text = self._embedded_text_cache.get(key, (0, None))
        # _embedded_text = None
        ts = time.time()
        if embedded_text is None:
            # with self.eval():
            embedded_text = self._text_field_embedder(tokens)
            self._embedded_text_cache[key] = (ts, embedded_text)
            #print('PASS')
        else:
            self._embedded_text_cache[key] = (ts, embedded_text)
            #print('HIT')

        self._embedded_text_cache = OrderedDict(sorted(self._embedded_text_cache.items(),
                                                       key=lambda kv: kv[1][0],
                                                       reverse=True))
        while len(self._embedded_text_cache) > self.embedded_text_cache_max:
            self._embedded_text_cache.popitem(last=True)

        embedded_text_list = [embedded_text]
        if self._type_embedding is not None:
            embedded_text_list.append(self._type_embedding(types))
        if self._tag_embedding is not None:
            embedded_text_list.append(self._tag_embedding(tags))
        embedded_text = torch.cat(embedded_text_list, dim=-1)

        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            for m in self._metrics.values():
                m(logits, label)

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        """
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        """
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {k: v.get_metric(reset) for k, v in self._metrics.items()}
        # metrics may be nested
        #metrics = {'/'.join(k): v for k, v in flatten_dict(metrics).items()}
        metrics = flatten_dict(metrics)
        # calc macro measures
        for m in ['f1', 'precision', 'recall']:
            # calc mean over all entries (except micro-F1)
            l = [v for k, v in metrics.items() if m in k and BasicTypedClassifier.MICRO_F1_NAME not in k]
            metrics[(f'macro-f1', m)] = sum(l) / len(l)

        metrics = {'/'.join(k): v for k, v in metrics.items()}
        # do not print "recall" and "precision" to progress bar
        metrics = {f'_{k}' if "/f1" in k or "/precision" in k or "/recall" in k else k: v for k, v in metrics.items()}
        return metrics

    default_predictor = "text_classifier"
