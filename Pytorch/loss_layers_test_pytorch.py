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
"""Tests for global objectives loss layers."""

# Dependency imports
from absl.testing import parameterized
from absl.testing import absltest
import numpy as np
import torch

import loss_layers_pytorch as loss_layers
import util_pytorch as util


# TODO: Include weights in the lagrange multiplier update tests.
class PrecisionRecallAUCLossTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ('_xent', 'xent', 0.7),
        ('_hinge', 'hinge', 0.7),
        ('_hinge_2', 'hinge', 0.5)
    )
    def testSinglePointAUC(self, surrogate_type, target_precision):
        # Tests a case with only one anchor point, where the loss should equal
        # recall_at_precision_loss
        batch_shape = [10, 2]
        logits = torch.rand(batch_shape)
        labels = torch.greater(torch.rand(batch_shape), 0.4).float()
   
        auc_loss, _ = loss_layers.precision_recall_auc_loss(
            labels,
            logits,
            precision_range = (target_precision - 0.01, target_precision  + 0.01),
            num_anchors = 1,
            surrogate_type = surrogate_type)
        point_loss, _ = loss_layers.recall_at_precision_loss(
            labels, 
            logits, 
            target_precision = target_precision,
            surrogate_type = surrogate_type)

        torch.testing.assert_close(auc_loss, point_loss)

    def testThreePointAUC(self):
        # Tests a case with three anchor points against a weighted sum of recall
        # at precision losses.
        batch_shape = [11, 3]
        logits = torch.rand(batch_shape)
        labels = torch.greater(torch.rand(batch_shape), 0.4).float()
    
        # TODO: Place the hing/xent loss in a for loop.
        auc_loss, _ = loss_layers.precision_recall_auc_loss(labels, logits, num_anchors = 1)
        first_point_loss, _ = loss_layers.recall_at_precision_loss(labels, logits, target_precision = 0.25)
        second_point_loss, _ = loss_layers.recall_at_precision_loss(labels, logits, target_precision = 0.5)
        third_point_loss, _ = loss_layers.recall_at_precision_loss(labels, logits, target_precision = 0.75)
        expected_loss = (first_point_loss + second_point_loss + third_point_loss) / 3

        auc_loss_hinge, _ = loss_layers.precision_recall_auc_loss(labels, logits, num_anchors = 1, surrogate_type = 'hinge')
        first_point_hinge, _ = loss_layers.recall_at_precision_loss(labels, logits, target_precision = 0.25, surrogate_type = 'hinge')
        second_point_hinge, _ = loss_layers.recall_at_precision_loss(labels, logits, target_precision = 0.5, surrogate_type = 'hinge')
        third_point_hinge, _ = loss_layers.recall_at_precision_loss(labels, logits, target_precision = 0.75, surrogate_type = 'hinge')
        expected_hinge = (first_point_hinge + second_point_hinge + third_point_hinge) / 3

        torch.testing.assert_close(auc_loss, expected_loss)
        torch.testing.assert_close(auc_loss_hinge, expected_hinge)

    def testLagrangeMultiplierUpdateDirection(self):
        for target_precision in [0.35, 0.65]:
            precision_range = (target_precision - 0.01, target_precision + 0.01)

            for surrogate_type in ['xent', 'hinge']:
                kwargs = {'precision_range': precision_range,
                            'num_anchors': 1,
                            'surrogate_type': surrogate_type}

                run_lagrange_multiplier_test(
                    global_objective = loss_layers.precision_recall_auc_loss,
                    objective_kwargs = kwargs,
                    data_builder = _multilabel_data,
                    test_object = self)

                run_lagrange_multiplier_test(
                    global_objective = loss_layers.precision_recall_auc_loss,
                    objective_kwargs = kwargs,
                    data_builder = _other_multilabel_data(surrogate_type),
                    test_object = self)


class ROCAUCLossTest(parameterized.TestCase):

    def testSimpleScores(self):
        # Tests the loss on data with only one negative example with score zero.
        # In this case, the loss should equal the surrogate loss on the scores with
        # positive labels.
        num_positives = 10
        scores_positives = torch.as_tensor(3.0 * np.random.randn(num_positives))
        labels = torch.tensor([0.0] + [1.0] * num_positives).reshape(num_positives + 1, 1)
        scores = torch.cat([torch.tensor([0.0]), scores_positives], 0)

        loss = torch.sum(loss_layers.roc_auc_loss(labels, scores, surrogate_type = 'hinge')[0])
        expected_loss = torch.sum(torch.maximum(1.0 - scores_positives, torch.tensor([0]))) / (num_positives + 1)
    
        #torch.testing.assert_close(expected_loss, loss)

    def testRandomROCLoss(self):
        # Checks that random Bernoulli scores and labels has ~25% swaps.
        shape = [1000, 30]
        scores = torch.as_tensor(np.random.randint(0, 2, size = shape), dtype = torch.float32)
        labels = torch.as_tensor(np.random.randint(0, 2, size = shape), dtype = torch.float32)
        loss = torch.mean(loss_layers.roc_auc_loss(labels, scores, surrogate_type = 'hinge')[0])
    
        torch.testing.assert_close(torch.tensor(0.25), loss, rtol = 1e-2, atol = 1e-2)

    @parameterized.named_parameters(
        ('_zero_hinge', 'xent',
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [-5.0, -7.0, -9.0, 8.0, 10.0, 14.0],
        0.0),
        ('_zero_xent', 'hinge',
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [-0.2, 0, -0.1, 1.0, 1.1, 1.0],
        0.0),
        ('_xent', 'xent',
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, -17.0, -19.0, 1.0, 14.0, 14.0],
        np.log(1.0 + np.exp(-1.0)) / 6),
        ('_hinge', 'hinge',
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [-0.2, -0.05, 0.0, 0.95, 0.8, 1.0],
        0.4 / 6)
    )
    def testManualROCLoss(self, surrogate_type, labels, logits, expected_value):
        labels = torch.as_tensor(labels)
        logits = torch.as_tensor(logits)
        loss, _ = loss_layers.roc_auc_loss(labels = labels, logits = logits, surrogate_type = surrogate_type)

        torch.testing.assert_close(torch.tensor(expected_value), torch.sum(loss), check_dtype = False)

    def testMultiLabelROCLoss(self):
        # Tests the loss on multi-label data against manually computed loss.
        targets = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        scores = np.array([[0.1, 1.0, 1.1, 1.0], [1.0, 0.0, 1.3, 1.1]])
        class_1_auc = torch.sum(loss_layers.roc_auc_loss(targets[0], scores[0])[0])
        class_2_auc = torch.sum(loss_layers.roc_auc_loss(targets[1], scores[1])[0])
        total_auc = torch.sum(loss_layers.roc_auc_loss(targets.transpose(), scores.transpose())[0])

        torch.testing.assert_close(total_auc, class_1_auc + class_2_auc)

    def testWeights(self):
        # Test the loss with per-example weights.
        # The logits_negatives below are repeated, so that setting half their
        # weights to 2 and the other half to 0 should leave the loss unchanged.
        logits_positives = torch.tensor([2.54321, -0.26, 3.334334]).reshape(3, 1)
        logits_negatives = torch.tensor([-0.6, 1, -1.3, -1.3, -0.6, 1]).reshape(6, 1)
        logits = torch.cat([logits_positives, logits_negatives], 0)
        targets = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype = torch.float32).reshape(9, 1)
        weights = torch.tensor([1, 1, 1, 0, 0, 0, 2, 2, 2], dtype = torch.float32).reshape(9, 1)

        loss = torch.sum(loss_layers.roc_auc_loss(targets, logits)[0])
        weighted_loss = torch.sum(loss_layers.roc_auc_loss(targets, logits, weights)[0])

        torch.testing.assert_close(loss, weighted_loss)


class RecallAtPrecisionTest(parameterized.TestCase):

    def testEqualWeightLoss(self):
        # Tests a special case where the loss should equal cross entropy loss.
        target_precision = 1.0
        num_labels = 5
        batch_shape = [20, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.7).float()
        label_priors = torch.full((num_labels,), 0.34)

        loss, _ = loss_layers.recall_at_precision_loss(targets, logits, target_precision, label_priors = label_priors)
        expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction = 'none')

        torch.testing.assert_close(loss, expected_loss)

    def testEqualWeightLossWithMultiplePrecisions(self):
        """Tests a case where the loss equals xent loss with multiple precisions."""
        target_precision = [1.0, 1.0]
        num_labels = 2
        batch_size = 20
        target_shape = [batch_size, num_labels]
        logits = torch.rand(target_shape)
        targets = torch.greater(torch.rand(target_shape), 0.7).float()
        label_priors = torch.full((num_labels,), 0.34)

        loss, _ = loss_layers.recall_at_precision_loss(
            targets,
            logits,
            target_precision,
            label_priors = label_priors,
            surrogate_type = 'xent',
        )

        expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction = 'none')

        torch.testing.assert_close(loss, expected_loss)

    def testPositivesOnlyLoss(self):
        # Tests a special case where the loss should equal cross entropy loss
        # on the negatives only.
        target_precision = 1.0
        num_labels = 3
        batch_shape = [30, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.4).float()
        label_priors = torch.full((num_labels,), 0.45)

        loss, _ = loss_layers.recall_at_precision_loss(
            targets, logits, target_precision, label_priors = label_priors,
            lambdas_initializer = torch.nn.init.zeros_)
        expected_loss = util.weighted_sigmoid_cross_entropy_with_logits(
            targets,
            logits,
            positive_weights = 1.0,
            negative_weights = 0.0)

        torch.testing.assert_close(loss, expected_loss)

    def testEquivalenceBetweenSingleAndMultiplePrecisions(self):
        """Checks recall at precision with different precision values.

        Runs recall at precision with multiple precision values, and runs each label
        seperately with its own precision value as a scalar. Validates that the
        returned loss values are the same.
        """
        target_precision = [0.2, 0.9, 0.4]
        num_labels = 3
        batch_shape = [30, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.4).float()
        label_priors = torch.tensor([0.45, 0.8, 0.3])

        multi_label_loss, _ = loss_layers.recall_at_precision_loss(
            targets, logits, target_precision, label_priors = label_priors,
        )

        single_label_losses = [
            loss_layers.recall_at_precision_loss(
                torch.unsqueeze(targets[:, i], -1),
                torch.unsqueeze(logits[:, i], -1),
                target_precision[i],
                label_priors = label_priors[i])[0]
            for i in range(num_labels)
        ]

        single_label_losses = torch.cat(single_label_losses, 1)

        torch.testing.assert_close(multi_label_loss, single_label_losses)

    def testEquivalenceBetweenSingleAndEqualMultiplePrecisions(self):
        """Compares single and multiple target precisions with the same value.

        Checks that using a single target precision and multiple target precisions
        with the same value would result in the same loss value.
        """
        num_labels = 2
        target_shape = [20, num_labels]
        logits = torch.rand(target_shape)
        targets = torch.greater(torch.rand(target_shape), 0.7).float()
        label_priors = torch.full((num_labels,), 0.34)

        multi_precision_loss, _ = loss_layers.recall_at_precision_loss(
            targets,
            logits,
            [0.75, 0.75],
            label_priors = label_priors,
            surrogate_type = 'xent',
        )

        single_precision_loss, _ = loss_layers.recall_at_precision_loss(
            targets,
            logits,
            0.75,
            label_priors = label_priors,
            surrogate_type = 'xent',
        )

        torch.testing.assert_close(multi_precision_loss, single_precision_loss)

    def testLagrangeMultiplierUpdateDirection(self):
        for target_precision in [0.35, 0.65]:
            for surrogate_type in ['xent', 'hinge']:
                kwargs = {'target_precision': target_precision,
                            'surrogate_type': surrogate_type}

                run_lagrange_multiplier_test(
                    global_objective = loss_layers.recall_at_precision_loss,
                    objective_kwargs = kwargs,
                    data_builder = _multilabel_data,
                    test_object = self)

                run_lagrange_multiplier_test(
                    global_objective = loss_layers.recall_at_precision_loss,
                    objective_kwargs = kwargs,
                    data_builder = _other_multilabel_data(surrogate_type),
                    test_object = self)

    def testLagrangeMultiplierUpdateDirectionWithMultiplePrecisions(self):
        """Runs Lagrange multiplier test with multiple precision values."""
        target_precision = [0.65, 0.35]

        for surrogate_type in ['xent', 'hinge']:
            kwargs = {
                'target_precision': target_precision,
                'surrogate_type': surrogate_type,
            }

            run_lagrange_multiplier_test(
                global_objective = loss_layers.recall_at_precision_loss,
                objective_kwargs = kwargs,
                data_builder = _multilabel_data,
                test_object = self)

            run_lagrange_multiplier_test(
                global_objective = loss_layers.recall_at_precision_loss,
                objective_kwargs = kwargs,
                data_builder = _other_multilabel_data(surrogate_type),
                test_object = self)


class PrecisionAtRecallTest(parameterized.TestCase):

    def testCrossEntropyEquivalence(self):
        # Checks a special case where the loss should equal cross-entropy loss.
        target_recall = 1.0
        num_labels = 3
        batch_shape = [10, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.4).float()

        loss, _ = loss_layers.precision_at_recall_loss(
            targets, logits, target_recall,
            lambdas_initializer = torch.nn.init.ones_)
        expected_loss = util.weighted_sigmoid_cross_entropy_with_logits(
            targets, logits)
    
        torch.testing.assert_close(loss, expected_loss)

    def testNegativesOnlyLoss(self):
        # Checks a special case where the loss should equal the loss on
        # the negative examples only.
        target_recall = 0.61828
        num_labels = 4
        batch_shape = [8, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.6).float()

        loss, _ = loss_layers.precision_at_recall_loss(
            targets,
            logits,
            target_recall,
            surrogate_type = 'hinge',
            lambdas_initializer = torch.nn.init.zeros_)
        expected_loss = util.weighted_hinge_loss(
            targets, logits, positive_weights = 0.0, negative_weights = 1.0)

        torch.testing.assert_close(expected_loss, loss)

    def testLagrangeMultiplierUpdateDirection(self):
        for target_recall in [0.34, 0.66]:
            for surrogate_type in ['xent', 'hinge']:
                kwargs = {'target_recall': target_recall,
                            'dual_rate_factor': 1.0,
                            'surrogate_type': surrogate_type}

                run_lagrange_multiplier_test(
                    global_objective = loss_layers.precision_at_recall_loss,
                    objective_kwargs = kwargs,
                    data_builder = _multilabel_data,
                    test_object = self)

                run_lagrange_multiplier_test(
                    global_objective = loss_layers.precision_at_recall_loss,
                    objective_kwargs = kwargs,
                    data_builder = _other_multilabel_data(surrogate_type),
                    test_object = self)

    def testCrossEntropyEquivalenceWithMultipleRecalls(self):
        """Checks a case where the loss equals xent loss with multiple recalls."""
        num_labels = 3
        target_recall = [1.0] * num_labels
        batch_shape = [10, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.4).float()

        loss, _ = loss_layers.precision_at_recall_loss(
            targets, logits, target_recall,
            lambdas_initializer = torch.nn.init.ones_)
        expected_loss = util.weighted_sigmoid_cross_entropy_with_logits(
            targets, logits)

        torch.testing.assert_close(loss, expected_loss)

    def testNegativesOnlyLossWithMultipleRecalls(self):
        """Tests a case where the loss equals the loss on the negative examples.

        Checks this special case using multiple target recall values.
        """
        num_labels = 4
        target_recall = [0.61828] * num_labels
        batch_shape = [8, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.6).float()

        loss, _ = loss_layers.precision_at_recall_loss(
            targets,
            logits,
            target_recall,
            surrogate_type = 'hinge',
            lambdas_initializer = torch.nn.init.zeros_)
        expected_loss = util.weighted_hinge_loss(
            targets, logits, positive_weights = 0.0, negative_weights = 1.0)

        torch.testing.assert_close(expected_loss, loss)

    def testLagrangeMultiplierUpdateDirectionWithMultipleRecalls(self):
        """Runs Lagrange multiplier test with multiple recall values."""
        target_recall = [0.34, 0.66]

        for surrogate_type in ['xent', 'hinge']:
            kwargs = {'target_recall': target_recall,
                    'dual_rate_factor': 1.0,
                    'surrogate_type': surrogate_type}

            run_lagrange_multiplier_test(
                global_objective = loss_layers.precision_at_recall_loss,
                objective_kwargs = kwargs,
                data_builder = _multilabel_data,
                test_object = self)

            run_lagrange_multiplier_test(
                global_objective = loss_layers.precision_at_recall_loss,
                objective_kwargs = kwargs,
                data_builder = _other_multilabel_data(surrogate_type),
                test_object = self)

    def testEquivalenceBetweenSingleAndMultipleRecalls(self):
        """Checks precision at recall with multiple different recall values.

        Runs precision at recall with multiple recall values, and runs each label
        seperately with its own recall value as a scalar. Validates that the
        returned loss values are the same.
        """
        target_precision = [0.7, 0.9, 0.4]
        num_labels = 3
        batch_shape = [30, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.4).float()
        label_priors = torch.full((num_labels,), 0.45)

        multi_label_loss, _ = loss_layers.precision_at_recall_loss(
            targets, logits, target_precision, label_priors = label_priors
        )

        single_label_losses = [
            loss_layers.precision_at_recall_loss(
                torch.unsqueeze(targets[:, i], -1),
                torch.unsqueeze(logits[:, i], -1),
                target_precision[i],
                label_priors = label_priors[i])[0]
            for i in range(num_labels)
        ]

        single_label_losses = torch.cat(single_label_losses, 1)

        torch.testing.assert_close(multi_label_loss, single_label_losses)

    def testEquivalenceBetweenSingleAndEqualMultipleRecalls(self):
        """Compares single and multiple target recalls of the same value.

        Checks that using a single target recall and multiple recalls with the
        same value would result in the same loss value.
        """
        num_labels = 2
        target_shape = [20, num_labels]
        logits = torch.rand(target_shape)
        targets = torch.greater(torch.rand(target_shape), 0.7).float()
        label_priors = torch.full((num_labels,), 0.34)

        multi_precision_loss, _ = loss_layers.precision_at_recall_loss(
            targets,
            logits,
            [0.75, 0.75],
            label_priors = label_priors,
            surrogate_type = 'xent',
        )

        single_precision_loss, _ = loss_layers.precision_at_recall_loss(
            targets,
            logits,
            0.75,
            label_priors = label_priors,
            surrogate_type = 'xent',
        )

        torch.testing.assert_close(multi_precision_loss, single_precision_loss)


class FalsePositiveRateAtTruePositiveRateTest(parameterized.TestCase):

    def testNegativesOnlyLoss(self):
        # Checks a special case where the loss returned should be the loss on the
        # negative examples.
        target_recall = 0.6
        num_labels = 3
        batch_shape = [3, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.4).float()
        label_priors = torch.as_tensor(np.random.uniform(size = [num_labels]), dtype = torch.float32)

        xent_loss, _ = loss_layers.false_positive_rate_at_true_positive_rate_loss(
            targets, logits, target_recall, label_priors = label_priors,
            lambdas_initializer = torch.nn.init.zeros_)
        xent_expected = util.weighted_sigmoid_cross_entropy_with_logits(
            targets,
            logits,
            positive_weights = 0.0,
            negative_weights = 1.0)
        hinge_loss, _ = loss_layers.false_positive_rate_at_true_positive_rate_loss(
            targets, logits, target_recall, label_priors = label_priors,
            lambdas_initializer = torch.nn.init.zeros_,
            surrogate_type = 'hinge')
        hinge_expected = util.weighted_hinge_loss(
            targets,
            logits,
            positive_weights = 0.0,
            negative_weights = 1.0)

        torch.testing.assert_close(xent_loss, xent_expected)
        torch.testing.assert_close(hinge_loss, hinge_expected)

    def testPositivesOnlyLoss(self):
        # Checks a special case where the loss returned should be the loss on the
        # positive examples only.
        target_recall = 1.0
        num_labels = 5
        batch_shape = [5, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.ones_like(logits)
        label_priors = torch.as_tensor(np.random.uniform(size = [num_labels]), dtype = torch.float32)

        loss, _ = loss_layers.false_positive_rate_at_true_positive_rate_loss(
            targets, logits, target_recall, label_priors = label_priors)
        expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction = 'none')
        hinge_loss, _ = loss_layers.false_positive_rate_at_true_positive_rate_loss(
            targets, logits, target_recall, label_priors = label_priors,
            surrogate_type = 'hinge')
        expected_hinge = util.weighted_hinge_loss(
            targets, logits)
  
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(hinge_loss, expected_hinge)

    def testEqualWeightLoss(self):
        # Checks a special case where the loss returned should be proportional to
        # the ordinary loss.
        target_recall = 1.0
        num_labels = 4
        batch_shape = [40, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.6).float()
        label_priors = torch.full((num_labels,), 0.5)

        loss, _ = loss_layers.false_positive_rate_at_true_positive_rate_loss(
            targets, logits, target_recall, label_priors = label_priors)
        expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction = 'none')

        torch.testing.assert_close(loss, expected_loss)

    def testLagrangeMultiplierUpdateDirection(self):
        for target_rate in [0.35, 0.65]:
            for surrogate_type in ['xent', 'hinge']:
                kwargs = {'target_rate': target_rate,
                            'surrogate_type': surrogate_type}

                # True positive rate is a synonym for recall, so we use the
                # recall constraint data.
                run_lagrange_multiplier_test(
                    global_objective = (loss_layers.false_positive_rate_at_true_positive_rate_loss),
                    objective_kwargs = kwargs,
                    data_builder = _multilabel_data,
                    test_object = self)

                run_lagrange_multiplier_test(
                    global_objective = (loss_layers.false_positive_rate_at_true_positive_rate_loss),
                    objective_kwargs = kwargs,
                    data_builder = _other_multilabel_data(surrogate_type),
                    test_object = self)

    def testLagrangeMultiplierUpdateDirectionWithMultipleRates(self):
        """Runs Lagrange multiplier test with multiple target rates."""
        target_rate = [0.35, 0.65]

        for surrogate_type in ['xent', 'hinge']:
            kwargs = {'target_rate': target_rate,
                    'surrogate_type': surrogate_type}

            # True positive rate is a synonym for recall, so we use the
            # recall constraint data.
            run_lagrange_multiplier_test(
                global_objective = loss_layers.false_positive_rate_at_true_positive_rate_loss,
                objective_kwargs = kwargs,
                data_builder = _multilabel_data,
                test_object = self)

            run_lagrange_multiplier_test(
                global_objective = loss_layers.false_positive_rate_at_true_positive_rate_loss,
                objective_kwargs = kwargs,
                data_builder = _other_multilabel_data(surrogate_type),
                test_object = self)

    def testEquivalenceBetweenSingleAndEqualMultipleRates(self):
        """Compares single and multiple target rates of the same value.

        Checks that using a single target rate and multiple rates with the
        same value would result in the same loss value.
        """
        num_labels = 2
        target_shape = [20, num_labels]
        logits = torch.rand(target_shape)
        targets = torch.greater(torch.rand(target_shape), 0.7).float()
        label_priors = torch.full((num_labels,), 0.34)

        multi_label_loss, _ = (
            loss_layers.false_positive_rate_at_true_positive_rate_loss(
                targets, logits, [0.75, 0.75], label_priors = label_priors))

        single_label_loss, _ = (
            loss_layers.false_positive_rate_at_true_positive_rate_loss(
                targets, logits, 0.75, label_priors = label_priors))

        torch.testing.assert_close(multi_label_loss, single_label_loss)

    def testEquivalenceBetweenSingleAndMultipleRates(self):
        """Compares single and multiple target rates of different values.

        Runs false_positive_rate_at_true_positive_rate_loss with multiple target
        rates, and runs each label seperately with its own target rate as a
        scalar. Validates that the returned loss values are the same.
        """
        target_precision = [0.7, 0.9, 0.4]
        num_labels = 3
        batch_shape = [30, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.4).float()
        label_priors = torch.full((num_labels,), 0.45)

        multi_label_loss, _ = (
            loss_layers.false_positive_rate_at_true_positive_rate_loss(
                targets, logits, target_precision, label_priors = label_priors))

        single_label_losses = [
            loss_layers.false_positive_rate_at_true_positive_rate_loss(
                torch.unsqueeze(targets[:, i], -1),
                torch.unsqueeze(logits[:, i], -1),
                target_precision[i],
                label_priors = label_priors[i])[0]
            for i in range(num_labels)
        ]

        single_label_losses = torch.cat(single_label_losses, 1)

        torch.testing.assert_close(multi_label_loss, single_label_losses)


class TruePositiveRateAtFalsePositiveRateTest(parameterized.TestCase):

    def testPositivesOnlyLoss(self):
        # A special case where the loss should equal the loss on the positive
        # examples.
        target_rate = np.random.uniform()
        num_labels = 3
        batch_shape = [20, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.6).float()
        label_priors = torch.as_tensor(np.random.uniform(size = [num_labels]), dtype = torch.float32)

        xent_loss, _ = loss_layers.true_positive_rate_at_false_positive_rate_loss(
            targets, logits, target_rate, label_priors = label_priors,
            lambdas_initializer = torch.nn.init.zeros_)
        xent_expected = util.weighted_sigmoid_cross_entropy_with_logits(
            targets,
            logits,
            positive_weights = 1.0,
            negative_weights = 0.0)
        hinge_loss, _ = loss_layers.true_positive_rate_at_false_positive_rate_loss(
            targets, logits, target_rate, label_priors = label_priors,
            lambdas_initializer = torch.nn.init.zeros_,
            surrogate_type = 'hinge')
        hinge_expected = util.weighted_hinge_loss(
            targets,
            logits,
            positive_weights = 1.0,
            negative_weights = 0.0)

        torch.testing.assert_close(xent_expected, xent_loss)
        torch.testing.assert_close(hinge_expected, hinge_loss)

    def testNegativesOnlyLoss(self):
        # A special case where the loss should equal the loss on the negative
        # examples, minus target_rate * (1 - label_priors) * maybe_log2.
        target_rate = np.random.uniform()
        num_labels = 3
        batch_shape = [25, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.zeros_like(logits)
        label_priors = torch.as_tensor(np.random.uniform(size = [num_labels]),dtype = torch.float32)

        xent_loss, _ = loss_layers.true_positive_rate_at_false_positive_rate_loss(
            targets, logits, target_rate, label_priors = label_priors)
        xent_expected = torch.subtract(
            util.weighted_sigmoid_cross_entropy_with_logits(targets, logits, positive_weights = 0.0, negative_weights = 1.0),
            target_rate * (1.0 - label_priors) * np.log(2))
        hinge_loss, _ = loss_layers.true_positive_rate_at_false_positive_rate_loss(
            targets, logits, target_rate, label_priors = label_priors,
            surrogate_type = 'hinge')
        hinge_expected = util.weighted_hinge_loss(
            targets, logits) - target_rate * (1.0 - label_priors)

        torch.testing.assert_close(xent_expected, xent_loss)
        torch.testing.assert_close(hinge_expected, hinge_loss)

    def testLagrangeMultiplierUpdateDirection(self):
        for target_rate in [0.35, 0.65]:
            for surrogate_type in ['xent', 'hinge']:
                kwargs = {'target_rate': target_rate,
                            'surrogate_type': surrogate_type}

                run_lagrange_multiplier_test(
                    global_objective = loss_layers.true_positive_rate_at_false_positive_rate_loss,
                    objective_kwargs = kwargs,
                    data_builder = _multilabel_data,
                    test_object = self)

                run_lagrange_multiplier_test(
                    global_objective = loss_layers.true_positive_rate_at_false_positive_rate_loss,
                    objective_kwargs = kwargs,
                    data_builder = _other_multilabel_data(surrogate_type),
                    test_object = self)

    def testLagrangeMultiplierUpdateDirectionWithMultipleRates(self):
        """Runs Lagrange multiplier test with multiple target rates."""
        target_rate = [0.35, 0.65]

        for surrogate_type in ['xent', 'hinge']:
            kwargs = {'target_rate': target_rate,
                    'surrogate_type': surrogate_type}

            run_lagrange_multiplier_test(
                global_objective = loss_layers.true_positive_rate_at_false_positive_rate_loss,
                objective_kwargs = kwargs,
                data_builder = _multilabel_data,
                test_object = self)

            run_lagrange_multiplier_test(
                global_objective = loss_layers.true_positive_rate_at_false_positive_rate_loss,
                objective_kwargs = kwargs,
                data_builder = _other_multilabel_data(surrogate_type),
                test_object = self)

    def testEquivalenceBetweenSingleAndEqualMultipleRates(self):
        """Compares single and multiple target rates of the same value.

        Checks that using a single target rate and multiple rates with the
        same value would result in the same loss value.
        """
        num_labels = 2
        target_shape = [20, num_labels]
        logits = torch.rand(target_shape)
        targets = torch.greater(torch.rand(target_shape), 0.7).float()
        label_priors = torch.full((num_labels,), 0.34)

        multi_label_loss, _ = (
            loss_layers.true_positive_rate_at_false_positive_rate_loss(
                targets, logits, [0.75, 0.75], label_priors = label_priors))

        single_label_loss, _ = (
            loss_layers.true_positive_rate_at_false_positive_rate_loss(
                targets, logits, 0.75, label_priors = label_priors))

        torch.testing.assert_close(multi_label_loss, single_label_loss)

    def testEquivalenceBetweenSingleAndMultipleRates(self):
        """Compares single and multiple target rates of different values.

        Runs true_positive_rate_at_false_positive_rate_loss with multiple target
        rates, and runs each label seperately with its own target rate as a
        scalar. Validates that the returned loss values are the same.
        """
        target_precision = [0.7, 0.9, 0.4]
        num_labels = 3
        batch_shape = [30, num_labels]
        logits = torch.rand(batch_shape)
        targets = torch.greater(torch.rand(batch_shape), 0.4).float()
        label_priors = torch.full((num_labels,), 0.45)

        multi_label_loss, _ = (
            loss_layers.true_positive_rate_at_false_positive_rate_loss(
                targets, logits, target_precision, label_priors = label_priors))

        single_label_losses = [
            loss_layers.true_positive_rate_at_false_positive_rate_loss(
                torch.unsqueeze(targets[:, i], -1),
                torch.unsqueeze(logits[:, i], -1),
                target_precision[i],
                label_priors = label_priors[i])[0]
            for i in range(num_labels)
        ]

        single_label_losses = torch.cat(single_label_losses, 1)

        torch.testing.assert_close(multi_label_loss, single_label_losses)


class UtilityFunctionsTest(parameterized.TestCase):

    def testTrainableDualVariable(self):
        # Confirm correct behavior of a trainable dual variable.
        x = torch.tensor([2.0], dtype = torch.float32, requires_grad = True) # primal
        y_value, y = loss_layers._create_dual_variable(
            shape = (1,), dtype = torch.float32, initializer = torch.nn.init.ones_,
            trainable = True, dual_rate_factor = 0.3)
        optimizer = torch.optim.SGD([x, y_value], lr = 1.0)
        # Update parameters once(?)
        loss = 0.5 * torch.square(x - y_value);
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.testing.assert_close(torch.tensor([0.7]), y)

    def testUntrainableDualVariable(self):
        # Confirm correct behavior of dual variable which is not trainable.
        x = torch.tensor([-2.0], dtype = torch.float32, requires_grad = True) # primal
        y_value, y = loss_layers._create_dual_variable(
            shape = (1,), dtype = torch.float32, initializer = torch.nn.init.ones_,
            trainable = False, dual_rate_factor = 0.8)
        optimizer = torch.optim.SGD([x, y_value], lr = 1.0)
        # Update parameters once(?)
        loss = torch.square(x) * y_value + torch.exp(y_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.testing.assert_close(torch.tensor([1.0]), y)


class BoundTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ('_xent', 'xent', 1.0, [2.0, 1.0]),
        ('_xent_weighted', 'xent',
        np.array([0, 2, 0.5, 1, 2, 3]).reshape(6, 1), [2.5, 0]),
        ('_hinge', 'hinge', 1.0, [2.0, 1.0]),
        ('_hinge_weighted', 'hinge',
        np.array([1.0, 2, 3, 4, 5, 6]).reshape(6, 1), [5.0, 1]))
    def testLowerBoundMultilabel(self, surrogate_type, weights, expected):
        labels, logits, _ = _multilabel_data()
        lower_bound = loss_layers.true_positives_lower_bound(
            labels, logits, weights, surrogate_type)

        torch.testing.assert_close(lower_bound, torch.tensor(expected), check_dtype = False)

    @parameterized.named_parameters(
        ('_xent', 'xent'), ('_hinge', 'hinge'))
    def testLowerBoundOtherMultilabel(self, surrogate_type):
        labels, logits, _ = _other_multilabel_data(surrogate_type)()
        lower_bound = loss_layers.true_positives_lower_bound(
            labels, logits, 1.0, surrogate_type)

        torch.testing.assert_close(lower_bound, torch.tensor([4.0, 2.0]))

    @parameterized.named_parameters(
        ('_xent', 'xent', 1.0, [1.0, 2.0]),
        ('_xent_weighted', 'xent',
        np.array([3.0, 2, 1, 0, 1, 2]).reshape(6, 1), [2.0, 1.0]),
        ('_hinge', 'hinge', 1.0, [1.0, 2.0]),
        ('_hinge_weighted', 'hinge',
        np.array([13, 12, 11, 0.5, 0, 0.5]).reshape(6, 1), [0.5, 0.5]))
    def testUpperBoundMultilabel(self, surrogate_type, weights, expected):
        labels, logits, _ = _multilabel_data()
        upper_bound = loss_layers.false_positives_upper_bound(
            labels, logits, weights, surrogate_type)

        torch.testing.assert_close(upper_bound, torch.tensor(expected), check_dtype = False)

    @parameterized.named_parameters(
        ('_xent', 'xent'), ('_hinge', 'hinge'))
    def testUpperBoundOtherMultilabel(self, surrogate_type):
        labels, logits, _ = _other_multilabel_data(surrogate_type)()
        upper_bound = loss_layers.false_positives_upper_bound(
            labels, logits, 1.0, surrogate_type)

        torch.testing.assert_close(upper_bound, torch.tensor([2.0, 4.0]))

    @parameterized.named_parameters(
        ('_lower', 'lower'), ('_upper', 'upper'))
    def testThreeDimensionalLogits(self, bound):
        bound_function = loss_layers.false_positives_upper_bound
        if bound == 'lower':
            bound_function = loss_layers.true_positives_lower_bound
        random_labels = np.float32(np.random.uniform(size = [2, 3]) > 0.5)
        random_logits = np.float32(np.random.randn(2, 3, 2))
        first_slice_logits = random_logits[:, :, 0].reshape(2, 3)
        second_slice_logits = random_logits[:, :, 1].reshape(2, 3)

        full_bound = bound_function(
            torch.as_tensor(random_labels), torch.as_tensor(random_logits), 1.0, 'xent')
        first_slice_bound = bound_function(torch.as_tensor(random_labels),
                                            torch.as_tensor(first_slice_logits),
                                            1.0,
                                            'xent')
        second_slice_bound = bound_function(torch.as_tensor(random_labels),
                                            torch.as_tensor(second_slice_logits),
                                            1.0,
                                            'xent')
        stacked_bound = torch.stack([first_slice_bound, second_slice_bound], dim = 1)

        torch.testing.assert_close(full_bound, stacked_bound)


def run_lagrange_multiplier_test(global_objective,
                                    objective_kwargs,
                                    data_builder,
                                    test_object):
    """Runs a test for the Lagrange multiplier update of `global_objective`.

    The test checks that the constraint for `global_objective` is satisfied on
    the first label of the data produced by `data_builder` but not the second.

    Args:
    global_objective: One of the global objectives.
    objective_kwargs: A dictionary of keyword arguments to pass to
        `global_objective`. Must contain an entry for the constraint argument
        of `global_objective`, e.g. 'target_rate' or 'target_precision'.
    data_builder: A function  which returns tensors corresponding to labels,
        logits, and label priors.
    test_object: An instance of .
    """
    # Construct global objective kwargs from a copy of `objective_kwargs`.
    kwargs = dict(objective_kwargs)
    targets, logits, priors = data_builder()
    kwargs['labels'] = targets
    kwargs['logits'] = logits
    kwargs['label_priors'] = priors

    loss, output_dict = global_objective(**kwargs)
    lambdas = torch.squeeze(output_dict['lambdas'])
    # Save unoptimized lambdas
    lambdas_before = lambdas.clone().detach()

    optimizer = torch.optim.SGD([output_dict['lambdas']], lr = 1.0)
    # Update parameters once(?)
    loss = torch.sum(loss) ##????
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save optimized lambdas
    lambdas_after = lambdas

    torch.testing.assert_close(torch.less(lambdas_after[0], lambdas_before[0]), torch.tensor(True))
    torch.testing.assert_close(torch.greater(lambdas_after[1], lambdas_before[1]), torch.tensor(True))


class CrossFunctionTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ('_auc01xent', loss_layers.precision_recall_auc_loss, {
            'precision_range': (0.0, 1.0), 'surrogate_type': 'xent'
        }),
        ('_auc051xent', loss_layers.precision_recall_auc_loss, {
            'precision_range': (0.5, 1.0), 'surrogate_type': 'xent'
        }),
        ('_auc01)hinge', loss_layers.precision_recall_auc_loss, {
            'precision_range': (0.0, 1.0), 'surrogate_type': 'hinge'
        }),
        ('_ratp04', loss_layers.recall_at_precision_loss, {
            'target_precision': 0.4, 'surrogate_type': 'xent'
        }),
        ('_ratp066', loss_layers.recall_at_precision_loss, {
            'target_precision': 0.66, 'surrogate_type': 'xent'
        }),
        ('_ratp07_hinge', loss_layers.recall_at_precision_loss, {
            'target_precision': 0.7, 'surrogate_type': 'hinge'
        }),
        ('_fpattp066', loss_layers.false_positive_rate_at_true_positive_rate_loss,
        {'target_rate': 0.66, 'surrogate_type': 'xent'}),
        ('_fpattp046', loss_layers.false_positive_rate_at_true_positive_rate_loss,
        {
            'target_rate': 0.46, 'surrogate_type': 'xent'
        }),
        ('_fpattp076_hinge',
        loss_layers.false_positive_rate_at_true_positive_rate_loss, {
            'target_rate': 0.76, 'surrogate_type': 'hinge'
        }),
        ('_fpattp036_hinge',
        loss_layers.false_positive_rate_at_true_positive_rate_loss, {
            'target_rate': 0.36, 'surrogate_type': 'hinge'
        }),
    )
    def testWeigtedGlobalObjective(self,
                                    global_objective,
                                    objective_kwargs):
        """Runs a test of `global_objective` with per-example weights.

        Args:
            global_objective: One of the global objectives.
            objective_kwargs: A dictionary of keyword arguments to pass to
            `global_objective`. Must contain keys 'surrogate_type', and the keyword
            for the constraint argument of `global_objective`, e.g. 'target_rate' or
            'target_precision'.
        """
        logits_positives = torch.tensor([1, -0.5, 3]).reshape(3, 1)
        logits_negatives = torch.tensor([-0.5, 1, -1, -1, -0.5, 1]).reshape(6, 1)

        # Dummy tensor is used to compute the gradients.
        dummy = torch.tensor([1.0], requires_grad = True)
        logits = torch.cat([logits_positives, logits_negatives], 0)
        logits = torch.multiply(logits, dummy)
        targets = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype = torch.float32).reshape(9, 1)
        priors = torch.tensor([1.0/3.0])
        weights = torch.tensor([1, 1, 1, 0, 0, 0, 2, 2, 2], dtype = torch.float32).reshape(9, 1)

        # Construct global objective kwargs.
        objective_kwargs['labels'] = targets
        objective_kwargs['logits'] = logits
        objective_kwargs['label_priors'] = priors

        # Unweighted loss.
        raw_loss, update = global_objective(**objective_kwargs)
        loss = torch.sum(raw_loss)

        # Weighted loss.
        objective_kwargs['weights'] = weights
        raw_weighted_loss, weighted_update = global_objective(**objective_kwargs)
        weighted_loss = torch.sum(raw_weighted_loss)

        lambdas = update['lambdas']
        weighted_lambdas = weighted_update['lambdas']
        logits_gradient = torch.autograd.grad(loss, dummy, retain_graph = True)
        weighted_logits_gradient = torch.autograd.grad(weighted_loss, dummy)

        # Assertions - Update in correct order???
        torch.testing.assert_close(loss, weighted_loss)

        torch.testing.assert_close(logits_gradient, weighted_logits_gradient)

        torch.testing.assert_close(lambdas, weighted_lambdas)

    @parameterized.named_parameters(
        ('_prauc051xent', loss_layers.precision_recall_auc_loss, {
            'precision_range': (0.5, 1.0), 'surrogate_type': 'xent'
        }),
        ('_prauc01hinge', loss_layers.precision_recall_auc_loss, {
            'precision_range': (0.0, 1.0), 'surrogate_type': 'hinge'
        }),
        ('_rocxent', loss_layers.roc_auc_loss, {'surrogate_type': 'xent'}),
        ('_rochinge', loss_layers.roc_auc_loss, {'surrogate_type': 'xent'}),
        ('_ratp04', loss_layers.recall_at_precision_loss, {
            'target_precision': 0.4, 'surrogate_type': 'xent'
        }),
        ('_ratp07_hinge', loss_layers.recall_at_precision_loss, {
            'target_precision': 0.7, 'surrogate_type': 'hinge'
        }),
        ('_patr05', loss_layers.precision_at_recall_loss, {
            'target_recall': 0.4, 'surrogate_type': 'xent'
        }),
        ('_patr08_hinge', loss_layers.precision_at_recall_loss, {
            'target_recall': 0.7, 'surrogate_type': 'hinge'
        }),
        ('_fpattp046', loss_layers.false_positive_rate_at_true_positive_rate_loss,
        {
            'target_rate': 0.46, 'surrogate_type': 'xent'
        }),
        ('_fpattp036_hinge',
        loss_layers.false_positive_rate_at_true_positive_rate_loss, {
            'target_rate': 0.36, 'surrogate_type': 'hinge'
        }),
        ('_tpatfp076', loss_layers.true_positive_rate_at_false_positive_rate_loss,
        {
            'target_rate': 0.76, 'surrogate_type': 'xent'
        }),
        ('_tpatfp036_hinge',
        loss_layers.true_positive_rate_at_false_positive_rate_loss, {
            'target_rate': 0.36, 'surrogate_type': 'hinge'
        }),
    )
    def testVectorAndMatrixLabelEquivalence(self,
                                            global_objective,
                                            objective_kwargs):
        """Tests equivalence between label shape [batch_size] or [batch_size, 1]."""
        vector_labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        vector_logits = torch.tensor([1.0, 0.1, 0.1, -1.0])

        # Construct vector global objective kwargs and loss.
        vector_kwargs = objective_kwargs.copy()
        vector_kwargs['labels'] = vector_labels
        vector_kwargs['logits'] = vector_logits
        vector_loss, _ = global_objective(**vector_kwargs)
        vector_loss_sum = torch.sum(vector_loss)

        # Construct matrix global objective kwargs and loss.
        matrix_kwargs = objective_kwargs.copy()
        matrix_kwargs['labels'] = torch.unsqueeze(vector_labels, 1)
        matrix_kwargs['logits'] = torch.unsqueeze(vector_logits, 1)
        matrix_loss, _ = global_objective(**matrix_kwargs)
        matrix_loss_sum = torch.sum(matrix_loss)

        self.assertEqual(1, vector_loss.ndim)
        self.assertEqual(2, matrix_loss.ndim)

        torch.testing.assert_close(vector_loss_sum, matrix_loss_sum)


# Both sets of logits below are designed so that the surrogate precision and
# recall (true positive rate) of class 1 is ~ 2/3, and the same surrogates for
# class 2 are ~ 1/3. The false positive rate surrogates are ~ 1/3 and 2/3.
def _multilabel_data():
    targets = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).reshape(6, 1)
    targets = torch.cat([targets, targets], 1)
    logits_positives = torch.tensor([[0.0, 15],
                                    [16, 0.0],
                                    [14, 0.0]])
    logits_negatives = torch.tensor([[-17, 0.0],
                                    [-15, 0.0],
                                    [0.0, -101]])
    logits = torch.cat([logits_positives, logits_negatives], 0)
    priors = torch.full((2,), 0.5)

    return targets, logits, priors


def _other_multilabel_data(surrogate_type):
    targets = torch.tensor([1.0] * 6 + [0.0] * 6).reshape(12, 1)
    targets = torch.cat([targets, targets], 1)
    logits_positives = torch.tensor([[0.0, 13],
                                    [12, 0.0],
                                    [15, 0.0],
                                    [0.0, 30],
                                    [13, 0.0],
                                    [18, 0.0]])
    # A score of cost_2 incurs a loss of ~2.0.
    cost_2 = 1.0 if surrogate_type == 'hinge' else 1.09861229
    logits_negatives = torch.tensor([[-16, cost_2],
                                    [-15, cost_2],
                                    [cost_2, -111],
                                    [-133, -14,],
                                    [-14.0100101, -16,],
                                    [-19.888828882, -101]])
    logits = torch.cat([logits_positives, logits_negatives], 0)
    priors = torch.full((2,), 0.5)

    def builder():
        return targets, logits, priors

    return builder


if __name__ == '__main__':
    absltest.main()
