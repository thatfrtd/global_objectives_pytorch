
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
"""Example for using global objectives.

Illustrate, using synthetic data, how using the precision_at_recall loss
significanly improves the performace of a linear classifier.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import torch
import loss_layers_pytorch as loss_layers
import matplotlib.pyplot as plt

# When optimizing using global_objectives, if set to True then the saddle point
# optimization steps are performed internally by the Tensorflow optimizer,
# otherwise by dedicated saddle-point steps as part of the optimization loop.
USE_GO_SADDLE_POINT_OPT = False

TARGET_PRECISION = 0.9
TRAIN_ITERATIONS = 240
LEARNING_RATE = 1
GO_DUAL_RATE_FACTOR = 15
NUM_CHECKPOINTS = 12

EXPERIMENT_DATA_CONFIG = {
    'positives_centers': [[0, 1.0], [1, -0.5]],
    'negatives_centers': [[0, -0.5], [1, 1.0]],
    'positives_variances': [0.15, 0.1],
    'negatives_variances': [0.15, 0.1],
    'positives_counts': [5000, 1000],
    'negatives_counts': [5000, 1000]
}


def create_training_and_eval_data_for_experiment(**data_config):
    """Creates train and eval data sets.

    Note: The synthesized binary-labeled data is a mixture of four Gaussians - two
    positives and two negatives. The centers, variances, and sizes for each of
    the two positives and negatives mixtures are passed in the respective keys
    of data_config:

    Args:
        **data_config: Dictionary with Array entries as follows:
        positives_centers - float [2,2] two centers of positives data sets.
        negatives_centers - float [2,2] two centers of negatives data sets.
        positives_variances - float [2] Variances for the positives sets.
        negatives_variances - float [2] Variances for the negatives sets.
        positives_counts - int [2] Counts for each of the two positives sets.
        negatives_counts - int [2] Counts for each of the two negatives sets.

    Returns:
    A dictionary with two shuffled data sets created - one for training and one
    for eval. The dictionary keys are 'train_data', 'train_labels', 'eval_data',
    and 'eval_labels'. The data points are two-dimentional floats, and the
    labels are in {0,1}.
    """
    def data_points(is_positives, index):
        variance = data_config['positives_variances'
                                if is_positives else 'negatives_variances'][index]
        center = data_config['positives_centers'
                                if is_positives else 'negatives_centers'][index]
        count = data_config['positives_counts'
                            if is_positives else 'negatives_counts'][index]
        return variance*np.random.randn(count, 2) + np.array([center])

    def create_data():
        return np.concatenate([data_points(False, 0),
                               data_points(True, 0),
                               data_points(True, 1),
                               data_points(False, 1)], axis=0)

    def create_labels():
        """Creates an array of 0.0 or 1.0 labels for the data_config batches."""
        return np.array([0.0]*data_config['negatives_counts'][0] +
                        [1.0]*data_config['positives_counts'][0] +
                        [1.0]*data_config['positives_counts'][1] +
                        [0.0]*data_config['negatives_counts'][1])

    permutation = np.random.permutation(
        sum(data_config['positives_counts'] + data_config['negatives_counts']))

    train_data = create_data()[permutation, :]
    eval_data = create_data()[permutation, :]
    train_labels = create_labels()[permutation]
    eval_labels = create_labels()[permutation]

    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'eval_data': eval_data,
        'eval_labels': eval_labels
    }


def train_model(data, use_global_objectives):
    """Trains a linear model for maximal accuracy or precision at given recall."""

    def recall_at_precision(scores, labels, target_precision):
        """Computes recall - at target precision - over data."""
        positive_scores = scores[labels == 1.0]
        threshold = np.percentile(positive_scores, 100 - target_precision*100)
        predicted = scores >= threshold
        return precision_score(labels, predicted)

    w = torch.tensor([-1.0, -1.0], dtype = torch.float32).reshape(2, 1).requires_grad_() # Weights
    b = torch.zeros([1], dtype = torch.float32, requires_grad = True) # Biases

    logits = torch.matmul(torch.tensor(data['train_data'], dtype = torch.float32), w) + b
    labels = torch.tensor(data['train_labels'], dtype = torch.float32).reshape(-1, 1)

    optimizer = torch.optim.SGD([w, b], lr = LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters = TRAIN_ITERATIONS, power = 1.0)

    if use_global_objectives:
        loss_function = loss_layers.RecallAtPrecisionLoss(TARGET_PRECISION,
                                                          dual_rate_factor = GO_DUAL_RATE_FACTOR,
                                                          surrogate_type = 'hinge')
    else:
        loss_function = torch.nn.BCEWithLogitsLoss(reduction = 'mean')

    # Training loop:
    checkpoint_step = TRAIN_ITERATIONS // NUM_CHECKPOINTS

    lambda_values = np.zeros(TRAIN_ITERATIONS, dtype = np.float32)

    for step in range(TRAIN_ITERATIONS):
        if (not use_global_objectives) or USE_GO_SADDLE_POINT_OPT:
            loss = loss_function.forward(labels, logits)

            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
            scheduler.step()
        else:
            for i in range(1):
                loss, other_outputs = loss_function.forward(labels, logits)
                loss = torch.mean(loss)

                lambdas = other_outputs['lambdas'].dual_variable

                lambda_optimizer = torch.optim.SGD([{'params': lambdas, 'lr': -other_outputs['lambdas'].dual_rate_factor}])

                lambda_optimizer.zero_grad()
                loss.backward(retain_graph = True)
                lambda_optimizer.step()

                other_outputs['lambdas'].nonnegativity_constraint()

                lambda_values[step] = lambdas.detach().float()

                if step > TRAIN_ITERATIONS / 3:
                    other_outputs['lambdas'].reset()

            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()

            scheduler.step()

        if step % checkpoint_step == 0:
            recall = recall_at_precision(
                np.dot(data['train_data'], w.detach().numpy()) + b.detach().numpy(),
                data['train_labels'], TARGET_PRECISION)

            print('Loss = %f Recall = %f' % (loss, recall))
            if use_global_objectives:
                other_outputs['lambdas'] = np.mean(other_outputs['lambdas'].dual_variable.detach().numpy())
                for i, output_name in enumerate(other_outputs.keys()):
                    print('\t%s = %f' % (output_name, other_outputs[output_name]))

    plt.plot(lambda_values)
    plt.show()

    return recall_at_precision(np.dot(data['eval_data'], w.detach().numpy()) + b.detach().numpy(),
                               data['eval_labels'],
                               TARGET_PRECISION)


def main():
    experiment_data = create_training_and_eval_data_for_experiment(**EXPERIMENT_DATA_CONFIG)

    global_objectives_loss_recall = train_model(experiment_data, True)
    print('global_objectives recall at requested precision is %f' % global_objectives_loss_recall)

    cross_entropy_loss_recall = train_model(experiment_data, False)
    print('cross_entropy recall at requested precision is %f' % cross_entropy_loss_recall)


if __name__ == '__main__':
    main()