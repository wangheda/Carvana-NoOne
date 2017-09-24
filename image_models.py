# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
from tensorflow import flags
FLAGS = flags.FLAGS

import sys
from os.path import dirname
if dirname(__file__) not in sys.path:
  sys.path.append(dirname(__file__))
from all_image_models import *


flags.DEFINE_integer("scaled_unet_downsample_rate", 2, "Rate of downsampling in ScaledUNetModel.")
flags.DEFINE_string("mixed_scaled_unet_downsample_rate", "2,3,5", "Rate of downsampling in MixedScaledUNetModel.")
flags.DEFINE_string("stacked_scaled_unet_downsample_rate", "5,3,2", "Rate of downsampling in StackedScaledUNetModel.")
flags.DEFINE_boolean("stacked_scaled_unet_use_support_predictions", False, "Whether to treat predictions in the middle layers as support predictions in StackedScaledUNetModel.")
