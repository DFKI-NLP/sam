from typing import Dict

from allennlp.training.metrics.f1_measure import F1Measure
from allennlp.training.metrics.metric import Metric


@Metric.register("f1_with_counts")
class F1WithCounts(F1Measure):
    """
    Same as F1Measure but also returns tp, fp and fn.
    """

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        # Returns

        precision : `float`
        recall : `float`
        f1-measure : `float`
        tp : 'float'
        fp: 'float'
        fn: 'float'
        """

        metric = {'tp': self._true_positives,
                  'fp': self._false_positives,
                  'fn': self._false_negatives}
        metric.update(super().get_metric(reset=reset))

        return metric

    @property
    def _true_positives(self):
        # When this metric is never called, `self._true_positive_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_positive_sum is None:
            return 0.0
        else:
            return self._true_positive_sum[self._positive_label].item()

    @property
    def _true_negatives(self):
        # When this metric is never called, `self._true_negative_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_negative_sum is None:
            return 0.0
        else:
            return self._true_negative_sum[self._positive_label].item()

    @property
    def _false_positives(self):
        # When this metric is never called, `self._pred_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._pred_sum is None:
            return 0.0
        else:
            # `self._pred_sum` is the total number of instances under each _predicted_ class,
            # including true positives and false positives.
            return (self._pred_sum[self._positive_label] - self._true_positives).item()

    @property
    def _false_negatives(self):
        # When this metric is never called, `self._true_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_sum is None:
            return 0.0
        else:
            # `self._true_sum` is the total number of instances under each _true_ class,
            # including true positives and false negatives.
            return (self._true_sum[self._positive_label] - self._true_positives).item()
