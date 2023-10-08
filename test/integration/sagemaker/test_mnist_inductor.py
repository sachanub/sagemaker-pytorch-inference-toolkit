# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import numpy as np
import pytest
import sagemaker
from sagemaker.pytorch import PyTorchModel

from integration import model_inductor_tar, mnist_inductor_script
from integration.sagemaker.timeout import timeout_and_delete_endpoint

SM_CPU_INSTANCE_TYPES = ["ml.c5.9xlarge"]
SM_SINGLE_GPU_INSTANCE_TYPES = ["ml.p3.2xlarge", "ml.g4dn.4xlarge", "ml.g5.4xlarge"]
SM_GRAVITON_INSTANCE_TYPES = ["ml.c7g.4xlarge"]


@pytest.mark.parametrize("instance_type", SM_CPU_INSTANCE_TYPES)
@pytest.mark.cpu_test
def test_mnist_cpu_inductor(sagemaker_session, image_uri, instance_type):
    if 'gpu' in image_uri or 'neuron' in image_uri:
        pytest.skip('Skipping because test will run on \'{}\' instance'.format(instance_type))
    _test_mnist_distributed(sagemaker_session, image_uri, instance_type, model_inductor_tar, mnist_inductor_script)


# @pytest.mark.parametrize("instance_type", SM_GRAVITON_INSTANCE_TYPES)
# def test_mnist_graviton_inductor(sagemaker_session, image_uri, instance_type):
#     if 'cpu' in image_uri or 'gpu' in image_uri:
#         pytest.skip('Skipping because test will run on \'{}\' instance'.format(instance_type))
#     _test_mnist_distributed(sagemaker_session, image_uri, instance_type, model_inductor_tar, mnist_inductor_script)


@pytest.mark.parametrize("instance_type", SM_SINGLE_GPU_INSTANCE_TYPES)
@pytest.mark.gpu_test
def test_mnist_gpu_inductor(sagemaker_session, image_uri, instance_type):
    if 'cpu' in image_uri or 'neuron' in image_uri:
        pytest.skip('Skipping because test will run on \'{}\' instance'.format(instance_type))
    _test_mnist_distributed(sagemaker_session, image_uri, instance_type, model_inductor_tar, mnist_inductor_script)


def _test_mnist_distributed(sagemaker_session, image_uri, instance_type, model_tar, mnist_script):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-pytorch-serving")

    model_data = sagemaker_session.upload_data(
        path=model_tar,
        key_prefix="sagemaker-pytorch-serving/models",
    )

    pytorch = PyTorchModel(model_data=model_data, role='SageMakerRole', entry_point=mnist_script,
                           image_uri=image_uri, sagemaker_session=sagemaker_session)
    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        predictor = pytorch.deploy(initial_instance_count=1, instance_type=instance_type,endpoint_name=endpoint_name)

        batch_size = 100
        data = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)
