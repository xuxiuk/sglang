from unittest.mock import Mock

import torch

from sglang.srt.speculative.dspark_components.dspark_tp import DsparkTpSync


def test_dspark_tp_sync_is_noop_for_single_rank():
    group = Mock(world_size=1)
    tensor = torch.tensor([1, 2, 3])

    assert DsparkTpSync(group).sync(tensor) is tensor
    group.broadcast.assert_not_called()


def test_dspark_tp_sync_broadcasts_from_rank_zero():
    group = Mock(world_size=2, pynccl_comm=None)
    tensor = torch.tensor([1, 2, 3])

    assert DsparkTpSync(group).sync(tensor) is tensor
    group.broadcast.assert_called_once_with(tensor, src=0)
