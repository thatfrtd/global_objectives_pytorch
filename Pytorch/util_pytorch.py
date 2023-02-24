# Copyright 2018 The TensorFlow Global Objectives Authors. All Rights Reserved.
#
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
# ==============================================================================
"""Contains utility functions for the global objectives library."""

# Dependency imports
import torch


def weighted_sigmoid_cross_entropy_with_logits(labels,
                                               logits,
                                               positive_weights = 1.0,
                                               negative_weights = 1.0):
    """Computes a weighting of sigmoid cross entropy given `logits`.

    Measures the weighted probability error in discrete classification tasks in
    which classes are independent and not mutually exclusive.  For instance, one
    could perform multilabel classification where a picture can contain both an
    elephant and a dog at the same time. The class weight multiplies the
    different types of errors.
    For brevity, let `x = logits`, `z = labels`, `c = positive_weights`,
    `d = negative_weights`  The
    weighed logistic loss is

    ```
    c * z * -log(sigmoid(x)) + d * (1 - z) * -log(1 - sigmoid(x))
    = c * z * -log(1 / (1 + exp(-x))) - d * (1 - z) * log(exp(-x) / (1 + exp(-x)))
    = c * z * log(1 + exp(-x)) + d * (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
    = c * z * log(1 + exp(-x)) + d * (1 - z) * (x + log(1 + exp(-x)))
    = (1 - z) * x * d + (1 - z + c * z ) * log(1 + exp(-x))
    =  - d * x * z + d * x + (d - d * z + c * z ) * log(1 + exp(-x))
    ```

    To ensure stability and avoid overflow, the implementation uses the identity
        log(1 + exp(-x)) = max(0,-x) + log(1 + exp(-abs(x)))
    and the result is computed as

    ```
    = -d * x * z + d * x
        + (d - d * z + c * z ) * (max(0,-x) + log(1 + exp(-abs(x))))
    ```

    Note that the loss is NOT an upper bound on the 0-1 loss, unless it is divided
    by log(2).

    Args:
    labels: A `Tensor` of type `float32` or `float64`. `labels` can be a 2D
        tensor with shape [batch_size, num_labels] or a 3D tensor with shape
        [batch_size, num_labels, K].
    logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
        shape [batch_size, num_labels, K], the loss is computed separately on each
        slice [:, :, k] of `logits`.
    positive_weights: A `Tensor` that holds positive weights and has the
        following semantics according to its shape:
        scalar - A global positive weight.
        1D tensor - must be of size K, a weight for each 'attempt'
        2D tensor - of size [num_labels, K'] where K' is either K or 1.
        The `positive_weights` will be expanded to the left to match the
        dimensions of logits and labels.
    negative_weights: A `Tensor` that holds positive weight and has the
        semantics identical to positive_weights.

    Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
        weighted logistic losses.
    """

    labels, logits, positive_weights, negative_weights = prepare_loss_args(labels, logits, positive_weights, negative_weights)

    softplus_term = torch.add(torch.maximum(-logits, torch.tensor([0.0])), torch.log(1.0 + torch.exp(-torch.abs(logits))))
    weight_dependent_factor = (negative_weights + (positive_weights - negative_weights) * labels)

    return (negative_weights * (logits - labels * logits) + weight_dependent_factor * softplus_term)


def weighted_hinge_loss(labels,
                        logits,
                        positive_weights = 1.0,
                        negative_weights = 1.0):
    """Computes weighted hinge loss given logits `logits`.

    The loss applies to multi-label classification tasks where labels are
    independent and not mutually exclusive. See also
    `weighted_sigmoid_cross_entropy_with_logits`.

    Args:
    labels: A `Tensor` of type `float32` or `float64`. Each entry must be
        either 0 or 1. `labels` can be a 2D tensor with shape
        [batch_size, num_labels] or a 3D tensor with shape
        [batch_size, num_labels, K].
    logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
        shape [batch_size, num_labels, K], the loss is computed separately on each
        slice [:, :, k] of `logits`.
    positive_weights: A `Tensor` that holds positive weights and has the
        following semantics according to its shape:
        scalar - A global positive weight.
        1D tensor - must be of size K, a weight for each 'attempt'
        2D tensor - of size [num_labels, K'] where K' is either K or 1.
        The `positive_weights` will be expanded to the left to match the
        dimensions of logits and labels.
    negative_weights: A `Tensor` that holds positive weight and has the
        semantics identical to positive_weights.

    Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
        weighted hinge loss.
    """
    labels, logits, positive_weights, negative_weights = prepare_loss_args(labels, logits, positive_weights, negative_weights)

    positives_term = positive_weights * labels * torch.maximum(1.0 - logits, torch.tensor([0]))
    negatives_term = (negative_weights * (1.0 - labels) * torch.maximum(1.0 + logits, torch.tensor([0])))

    return positives_term + negatives_term


def weighted_surrogate_loss(labels,
                            logits,
                            surrogate_type = 'xent',
                            positive_weights = 1.0,
                            negative_weights = 1.0):
    """Returns either weighted cross-entropy or hinge loss.

    For example `surrogate_type` is 'xent' returns the weighted cross
    entropy loss.

    Args:
    labels: A `Tensor` of type `float32` or `float64`. Each entry must be
        between 0 and 1. `labels` can be a 2D tensor with shape
        [batch_size, num_labels] or a 3D tensor with shape
        [batch_size, num_labels, K].
    logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
        shape [batch_size, num_labels, K], each slice [:, :, k] represents an
        'attempt' to predict `labels` and the loss is computed per slice.
    surrogate_type: A string that determines which loss to return, supports
    'xent' for cross-entropy and 'hinge' for hinge loss.
    positive_weights: A `Tensor` that holds positive weights and has the
        following semantics according to its shape:
        scalar - A global positive weight.
        1D tensor - must be of size K, a weight for each 'attempt'
        2D tensor - of size [num_labels, K'] where K' is either K or 1.
        The `positive_weights` will be expanded to the left to match the
        dimensions of logits and labels.
    negative_weights: A `Tensor` that holds positive weight and has the
        semantics identical to positive_weights.

    Returns:
    The weigthed loss.

    Raises:
    ValueError: If value of `surrogate_type` is not supported.
    """

    if surrogate_type == 'xent':
        return weighted_sigmoid_cross_entropy_with_logits(
            logits = logits,
            labels = labels,
            positive_weights = positive_weights,
            negative_weights = negative_weights)
    elif surrogate_type == 'hinge':
        return weighted_hinge_loss(
            logits = logits,
            labels = labels,
            positive_weights = positive_weights,
            negative_weights = negative_weights)
    raise ValueError('surrogate_type %s not supported.' % surrogate_type)


def expand_outer(tensor, rank):
    """Expands the given `Tensor` outwards to a target rank.

    For example if rank = 3 and tensor.shape is [3, 4], this function will expand
    to such that the resulting shape will be  [1, 3, 4].

    Args:
    tensor: The tensor to expand.
    rank: The target dimension.

    Returns:
    The expanded tensor.

    Raises:
    ValueError: If rank of `tensor` is unknown, or if `rank` is smaller than
        the rank of `tensor`.
    """
    if tensor.ndim is None:
        raise ValueError('tensor dimension must be known.')
    if len(tensor.shape) > rank:
        raise ValueError(
            '`rank` must be at least the current tensor dimension: (%s vs %s).' %
            (rank, len(tensor.shape)))
    while len(tensor.shape) < rank:
        tensor = torch.unsqueeze(tensor, dim = 0)

    return tensor

class LabelPriors():
    """Creates an op to maintain and update label prior probabilities.

    For each label, the label priors are estimated as
        (P + sum_i w_i y_i) / (P + N + sum_i w_i),
    where y_i is the ith label, w_i is the ith weight, P is a pseudo-count of
    positive labels, and N is a pseudo-count of negative labels. The index i
    ranges over all labels observed during all evaluations of the returned op.

    Args:
    labels: A `Tensor` with shape [batch_size, num_labels]. Entries should be
        in [0, 1].
    weights: Coefficients representing the weight of each label. Must be either
        a Tensor of shape [batch_size, num_labels] or `None`, in which case each
        weight is treated as 1.0.
    positive_pseudocount: Number of positive labels used to initialize the label
        priors.
    negative_pseudocount: Number of negative labels used to initialize the label
        priors.

    Returns:
    label_priors: An op to update the weighted label_priors. Gives the
        current value of the label priors when evaluated.
    """
    def __init__(self, labels,
                       weights = None,
                       positive_pseudocount = 1.0,
                       negative_pseudocount = 1.0):
        self.positive_pseudocount = positive_pseudocount
        self.negative_psuedocount = negative_pseudocount

        # Initialize weighted label counts and weight sum
        dtype = labels.dtype
        num_labels = get_num_labels(labels)

        if weights is None:
            weights = torch.ones_like(labels)

        weighted_label_counts = torch.full((num_labels,), positive_pseudocount, dtype = dtype, requires_grad = False)
        self.weighted_label_counts = weighted_label_counts + torch.sum(weights * labels, dim = 0)
        weight_sum = torch.full((num_labels,), positive_pseudocount + negative_pseudocount, dtype = dtype, requires_grad = False)
        self.weight_sum = weight_sum  + torch.sum(weights, dim = 0)

        # Calculate Inital Priors
        self.label_priors = torch.div(self.weighted_label_counts, self.weight_sum)

    def update_label_priors(self, labels, weights = None):

        if weights is None:
            weights = torch.ones_like(labels)

        self.weighted_label_counts = self.weighted_label_counts + torch.sum(weights * labels, dim = 0)

        self.weight_sum = self.weight_sum + torch.sum(weights, dim = 0)

        self.label_priors = torch.div(self.weighted_label_counts, self.weight_sum)

def prepare_loss_args(labels, logits, positive_weights, negative_weights):
    """Prepare arguments for weighted loss functions.

    If needed, will convert given arguments to appropriate type and shape.

    Args:
    labels: labels or labels of the loss function.
    logits: Logits of the loss function.
    positive_weights: Weight on the positive examples.
    negative_weights: Weight on the negative examples.

    Returns:
    Converted labels, logits, positive_weights, negative_weights.
    """
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels, dtype = logits.dtype)
    if len(labels.shape) == 2 and len(logits.shape) == 3:
        labels = torch.unsqueeze(labels, dim = 2)

    positive_weights = torch.as_tensor(positive_weights, dtype = logits.dtype)
    positive_weights = expand_outer(positive_weights, logits.ndim)

    negative_weights = torch.as_tensor(negative_weights, dtype = logits.dtype)
    negative_weights = expand_outer(negative_weights, logits.ndim)

    return labels, logits, positive_weights, negative_weights


def get_num_labels(labels_or_logits):
    """Returns the number of labels inferred from labels_or_logits."""
    if labels_or_logits.ndim <= 1:
        return 1
    return labels_or_logits.shape[1]