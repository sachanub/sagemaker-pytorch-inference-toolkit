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

import os

from utils import file_utils

resources_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources'))
mnist_path = os.path.join(resources_path, 'mnist')
resnet18_path = os.path.join(resources_path, 'resnet18')
mme_path = os.path.join(resources_path, 'mme')
data_dir = os.path.join(mnist_path, 'data')
training_dir = os.path.join(data_dir, 'training')
cpu_sub_dir = 'model_cpu'
gpu_sub_dir = 'model_gpu'
inductor_sub_dir = 'model_inductor'
code_sub_dir = 'code'
default_sub_dir = 'default_model'
default_sub_traced_resnet_dir = 'default_traced_resnet'
resnet18_sub_dir = 'resnet18'
traced_resnet18_sub_dir = 'traced_resnet18'

model_cpu_dir = os.path.join(mnist_path, cpu_sub_dir)
mnist_cpu_script = os.path.join(model_cpu_dir, code_sub_dir, 'mnist.py')
model_cpu_tar = file_utils.make_tarfile(mnist_cpu_script,
                                        os.path.join(model_cpu_dir, "torch_model.pth"),
                                        model_cpu_dir,
                                        script_path="code")

model_cpu_1d_dir = os.path.join(model_cpu_dir, '1d')
mnist_1d_script = os.path.join(model_cpu_1d_dir, code_sub_dir, 'mnist_1d.py')
model_cpu_1d_tar = file_utils.make_tarfile(mnist_1d_script,
                                           os.path.join(model_cpu_1d_dir, "torch_model.pth"),
                                           model_cpu_1d_dir,
                                           script_path="code")

model_gpu_dir = os.path.join(mnist_path, gpu_sub_dir)
mnist_gpu_script = os.path.join(model_gpu_dir, code_sub_dir, 'mnist.py')
model_gpu_tar = file_utils.make_tarfile(mnist_gpu_script,
                                        os.path.join(model_gpu_dir, "torch_model.pth"),
                                        model_gpu_dir,
                                        script_path="code")

model_inductor_dir = os.path.join(mnist_path, inductor_sub_dir)
mnist_inductor_script = os.path.join(model_inductor_dir, code_sub_dir, 'mnist.py')
model_inductor_tar = file_utils.make_tarfile(mnist_inductor_script,
                                        os.path.join(model_inductor_dir, "torch_model.pth"),
                                        model_inductor_dir)

call_model_fn_once_script = os.path.join(model_cpu_dir, code_sub_dir, 'call_model_fn_once.py')
call_model_fn_once_tar = file_utils.make_tarfile(call_model_fn_once_script,
                                                 os.path.join(model_cpu_dir, "torch_model.pth"),
                                                 model_cpu_dir,
                                                 "model_call_model_fn_once.tar.gz",
                                                 script_path="code")

default_model_dir = os.path.join(resnet18_path, default_sub_dir)
default_model_script = os.path.join(default_model_dir, code_sub_dir, "resnet18.py")
default_model_tar = file_utils.make_tarfile(
    default_model_script, os.path.join(default_model_dir, "model.pt"), default_model_dir, script_path="code"
)

default_traced_resnet_dir = os.path.join(resnet18_path, default_sub_traced_resnet_dir)
default_traced_resnet_script = os.path.join(default_traced_resnet_dir, code_sub_dir, "resnet18.py")
default_model_traced_resnet18_tar = file_utils.make_tarfile(
    default_traced_resnet_script,
    os.path.join(default_traced_resnet_dir, "traced_resnet18.pt"),
    default_traced_resnet_dir,
    filename="traced_resnet18.tar.gz",
    script_path="code",
)

resnet18_model_dir = os.path.join(mme_path, resnet18_sub_dir)
resnet18_script = os.path.join(resnet18_model_dir, code_sub_dir, "inference.py")
resnet18_tar = file_utils.make_tarfile(
    resnet18_script,
    os.path.join(resnet18_model_dir, "model.pt"),
    resnet18_model_dir,
    filename="resnet18.tar.gz",
    script_path="code"
)

traced_resnet18_model_dir = os.path.join(mme_path, traced_resnet18_sub_dir)
traced_resnet18_script = os.path.join(traced_resnet18_model_dir, code_sub_dir, "inference.py")
traced_resnet18_tar = file_utils.make_tarfile(
    traced_resnet18_script,
    os.path.join(traced_resnet18_model_dir, "traced_resnet18.pt"),
    traced_resnet18_model_dir,
    filename="traced_resnet18.tar.gz",
    script_path="code"
)

ROLE = 'dummy/unused-role'
DEFAULT_TIMEOUT = 20
PYTHON3 = 'py3'

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources'))

# These regions have some p2 and p3 instances, but not enough for automated testing
NO_P2_REGIONS = ['ca-central-1', 'eu-central-1', 'eu-west-2', 'us-west-1', 'eu-west-3',
                 'eu-north-1', 'sa-east-1', 'ap-east-1']
NO_P3_REGIONS = ['ap-southeast-1', 'ap-southeast-2', 'ap-south-1', 'ca-central-1',
                 'eu-central-1', 'eu-west-2', 'us-west-1', 'eu-west-3', 'eu-north-1',
                 'sa-east-1', 'ap-east-1']
