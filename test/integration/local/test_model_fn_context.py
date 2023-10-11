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
import subprocess
import sys
import time

import pytest
import requests
import torch

from integration import model_gpu_context_dir

BASE_URL = "http://0.0.0.0:8080/"
PING_URL = BASE_URL + "ping"


@pytest.fixture(scope="module", autouse=True)
def container(image_uri):
    try:
        if 'cpu' in image_uri:
            pytest.skip("Skipping because tests running on CPU instance")

        command = (
            "docker run --gpus=all "
            "--name sagemaker-pytorch-inference-toolkit-context-test "
            "-v {}:/opt/ml/model "
            "{} serve"
        ).format(model_gpu_context_dir, image_uri)

        proc = subprocess.Popen(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

        attempts = 0
        while attempts < 10:
            time.sleep(3)
            try:
                requests.get(PING_URL)
                break
            except Exception:
                attempts += 1
                pass
        yield proc.pid

    finally:
        if 'cpu' in image_uri:
            pytest.skip("Skipping because tests running on CPU instance")
        subprocess.check_call("docker rm -f sagemaker-pytorch-inference-toolkit-context-test".split())


def test_context_all_device_ids():
    gpu_count = torch.cuda.device_count()

    gpu_ids_expected = [i for i in range(gpu_count)]
    gpu_ids_actual = []

    with open(os.path.join(model_gpu_context_dir, 'code', 'device_info.txt'), 'r') as f:
        for line in f:
            gpu_ids_actual.append(int(line))

    gpu_ids_actual = list(set(gpu_ids_actual))
    gpu_ids_actual.sort()

    assert gpu_ids_actual == gpu_ids_expected
