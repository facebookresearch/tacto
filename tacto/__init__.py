# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .renderer import Renderer  # noqa: F401

from .sensor import (  # noqa: F401; noqa: F401
    Sensor,
    get_digit_config_path,
    get_digit_shadow_config_path,
    get_omnitact_config_path,
)

__author__ = "Wang, Shaoxiong and Lambeta, Mike and Chou, Po-Wei and Calandra, Roberto"
__contact__ = "sxwang@fb.com, lambetam@fb.com, poweic@fb.com, rcalandra@fb.com"
__version__ = "0.0.3"  # Source of truth for tacto's version

_exported_dunders = {
    "__version__",
}

# Export symbols except those start with "_", unless they are whitelisted above.
__all__ = [s for s in dir() if s in _exported_dunders or not s.startswith("_")]
