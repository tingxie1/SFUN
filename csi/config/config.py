

import functools
import inspect
import logging
from fvcore.common.config import CfgNode as _CfgNode

from csi.utils.file_io import PathManager

class CfgNode(_CfgNode):

    @classmethod
    def _open_cfg(cls, filename):
        return PathManager.open(filename, "r")

    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        assert PathManager.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        from .defaults import _C

        logger = logging.getLogger(__name__)

        self.merge_from_other_cfg(loaded_cfg)
        

    def dump(self, *args, **kwargs):
        return super().dump(*args, **kwargs)
    

def get_cfg() -> CfgNode:
    from .defaults import _C

    return _C.clone()