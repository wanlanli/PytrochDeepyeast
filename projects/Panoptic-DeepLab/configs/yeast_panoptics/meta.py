# Copyright 2024 wlli
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

YEAST_CATEGORIES = [
    {"color": [0, 0, 0], "isthing": 0, "id": 0, "trainId": 0, "name": "background"},
    {"color": [253, 27, 27], "isthing": 1, "id": 1, "trainId": 1, "name": "cell"},
    {"color": [249, 127, 34], "isthing": 1, "id": 2, "trainId": 2, "name": "shmoo"},
    {"color": [246, 223, 107], "isthing": 1, "id": 3, "trainId": 3, "name": "zygote"},
    {"color": [42, 180, 112], "isthing": 1, "id": 4, "trainId": 4, "name": "tetrad"},
    {"color": [39, 69, 167], "isthing": 1, "id": 5, "trainId": 5, "name": "lysis"},
    {"color": [71, 20, 101], "isthing": 1, "id": 6, "trainId": 6, "name": "spore"},
    # trainId 0-6 are the meaningful phenotype classes. Everything below (mating-type
    # variants, paired, and the raw "unknown" annotation) is folded into a single
    # trainId=7 "unknown" class, so the model has 8 output channels in total.
    {"color": [252, 111, 177], "isthing": 1, "id": 7, "trainId": 7, "name": "cell_mt"},
    {"color": [255, 242, 161], "isthing": 1, "id": 8, "trainId": 7, "name": "shmoo_mt"},
    {"color": [189, 245, 169], "isthing": 1, "id": 9, "trainId": 7, "name": "fusion_mt"},
    {"color": [120, 224, 245], "isthing": 1, "id": 10, "trainId": 7, "name": "tetrads_mt"},
    {"color": [105, 150, 230], "isthing": 0, "id": 11, "trainId": 7, "name": "paired"},
    {"color": [120, 107, 207], "isthing": 0, "id": 12, "trainId": 7, "name": "unknown"},
]

# One entry per trainId (0-7), in trainId order. This must have exactly NUM_CLASSES
# entries: detectron2's SemSegEvaluator sets `self._num_classes = len(stuff_classes)`,
# so a mismatch here silently reports IoU under the wrong class names (or phantom
# all-nan classes) rather than failing loudly.
YEAST_SEM_SEG_CATEGORIES = [
    {"id": 0, "name": "background", "color": [0, 0, 0]},
    {"id": 1, "name": "cell", "color": [253, 27, 27]},
    {"id": 2, "name": "shmoo", "color": [249, 127, 34]},
    {"id": 3, "name": "zygote", "color": [246, 223, 107]},
    {"id": 4, "name": "tetrad", "color": [42, 180, 112]},
    {"id": 5, "name": "lysis", "color": [39, 69, 167]},
    {"id": 6, "name": "spore", "color": [71, 20, 101]},
    {"id": 7, "name": "unknown", "color": [252, 111, 177]},
]