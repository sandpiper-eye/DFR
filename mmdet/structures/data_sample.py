# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing.reduction import ForkingPickler
from typing import Union, Sequence

import numpy as np
import torch
from mmengine.structures import BaseDataElement

from .utils import LABEL_TYPE, SCORE_TYPE, format_label, format_score

class DataSample(BaseDataElement):

    def set_gt_label(self, value: LABEL_TYPE) -> 'DataSample':
        """Set ``gt_label``."""
        self.set_field(format_label(value), 'gt_label', dtype=torch.Tensor)
        return self

    def set_gt_score(self, value: SCORE_TYPE) -> 'DataSample':
        """Set ``gt_score``."""
        score = format_score(value)
        self.set_field(score, 'gt_score', dtype=torch.Tensor)
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
        return self

    def set_pred_label(self, value: LABEL_TYPE) -> 'DataSample':
        """Set ``pred_label``."""
        self.set_field(format_label(value), 'pred_label', dtype=torch.Tensor)
        return self

    def set_pred_score(self, value: SCORE_TYPE):
        """Set ``pred_label``."""
        score = format_score(value)
        self.set_field(score, 'pred_score', dtype=torch.Tensor)
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
        return self

    def set_mask(self, value: Union[torch.Tensor, np.ndarray]):
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        elif not isinstance(value, torch.Tensor):
            raise TypeError(f'Invalid mask type {type(value)}')
        self.set_field(value, 'mask', dtype=torch.Tensor)
        return self

    def __repr__(self) -> str:
        """Represent the object."""

        def dump_items(items, prefix=''):
            return '\n'.join(f'{prefix}{k}: {v}' for k, v in items)

        repr_ = ''
        if len(self._metainfo_fields) > 0:
            repr_ += '\n\nMETA INFORMATION\n'
            repr_ += dump_items(self.metainfo_items(), prefix=' ' * 4)
        if len(self._data_fields) > 0:
            repr_ += '\n\nDATA FIELDS\n'
            repr_ += dump_items(self.items(), prefix=' ' * 4)

        repr_ = f'<{self.__class__.__name__}({repr_}\n\n) at {hex(id(self))}>'
        return repr_


def _reduce_datasample(data_sample):
    """reduce DataSample."""
    attr_dict = data_sample.__dict__
    convert_keys = []
    for k, v in attr_dict.items():
        if isinstance(v, torch.Tensor):
            attr_dict[k] = v.numpy()
            convert_keys.append(k)
    return _rebuild_datasample, (attr_dict, convert_keys)


def _rebuild_datasample(attr_dict, convert_keys):
    """rebuild DataSample."""
    data_sample = DataSample()
    for k in convert_keys:
        attr_dict[k] = torch.from_numpy(attr_dict[k])
    data_sample.__dict__ = attr_dict
    return data_sample


# Due to the multi-processing strategy of PyTorch, DataSample may consume many
# file descriptors because it contains multiple tensors. Here we overwrite the
# reduce function of DataSample in ForkingPickler and convert these tensors to
# np.ndarray during pickling. It may slightly influence the performance of
# dataloader.
ForkingPickler.register(DataSample, _reduce_datasample)