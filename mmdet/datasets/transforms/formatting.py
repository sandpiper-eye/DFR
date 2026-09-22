# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Dict

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample, ReIDDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes
from mmdet.structures import DataSample

import torch
import torchvision.transforms.functional as F
from PIL import Image

@TRANSFORMS.register_module()
class ToTensorCHW(BaseTransform):
    """
    Convert image from numpy.ndarray (H, W, C) to torch.Tensor (C, H, W).
    """
    def __init__(self, keys: Sequence[str]):
        self.keys = keys

    def transform(self, results: Dict) -> Dict:
        """
        Args:
            results (dict): The result dict from the data pipeline.

        Returns:
            dict: The result dict with specified keys converted to CHW tensor.
        """
        for key in self.keys:
            if key in results:
                img = results[key]
                if isinstance(img, np.ndarray):
                    # 1. 确保是 HWC 格式
                    if img.ndim < 3:
                        img = np.expand_dims(img, -1)
                    # 2. HWC to CHW
                    img_chw = img.transpose(2, 0, 1)
                    # 3. 转换为 Tensor
                    results[key] = torch.from_numpy(np.ascontiguousarray(img_chw))
                elif isinstance(img, torch.Tensor):
                    # 如果已经是 Tensor，确保它是 CHW
                    if img.ndim == 3 and img.shape[2] in [1, 3]: # 看起来像 HWC
                        results[key] = img.permute(2, 0, 1)
                else:
                    raise TypeError(f"Unsupported type {type(img)} for ToTensorCHW")
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(keys={self.keys})'

@TRANSFORMS.register_module()
class Rename(BaseTransform):
    """Rename the keys in data pipeline.

    Args:
        mapping (dict): A dictionary mapping old keys to new keys.
                        Example: {'old_key': 'new_key'}
    """

    def __init__(self, mapping: Dict[str, str]) -> None:
        if not isinstance(mapping, dict):
            raise TypeError(
                f'mapping must be a dict, but got {type(mapping)} instead.')
        self.mapping = mapping

    def transform(self, results: Dict) -> Dict:
        """The transform function to rename keys in results.

        Args:
            results (dict): The result dict from the data pipeline.

        Returns:
            dict: The result dict with renamed keys.
        """
        for old_key, new_key in self.mapping.items():
            if old_key in results:
                results[new_key] = results.pop(old_key)
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mapping={self.mapping})'

@TRANSFORMS.register_module()
class DebugHook(BaseTransform):
    """
    A transform to print the shape, dtype, and other info of keys in the results dict.

    This hook can be inserted anywhere in a pipeline to inspect the data flow.
    """

    def __init__(self,
                 label: str = "Debug Point",
                 keys: Optional[Sequence[str]] = None):
        """
        Args:
            label (str): A custom label to identify this debug point in logs.
            keys (list[str], optional): Specific keys to inspect.
                                        If None, inspects all keys.
        """
        self.label = label
        self.keys_to_inspect = keys

    def transform(self, results: Dict) -> Dict:
        """
        Prints the info and returns the results dict unchanged.
        """
        print(f"\n--- Debug Hook: {self.label} ---")

        keys_to_show = self.keys_to_inspect if self.keys_to_inspect else results.keys()

        for key in keys_to_show:
            if key in results:
                value = results[key]
                info = f"  Key: '{key}' | Type: {type(value)}"
                if hasattr(value, 'shape'):
                    info += f" | Shape: {value.shape}"
                if hasattr(value, 'dtype'):
                    info += f" | Dtype: {value.dtype}"
                if isinstance(value, torch.Tensor):
                    info += f" | Min: {value.min():.2f}, Max: {value.max():.2f}"
                elif isinstance(value, np.ndarray):
                    info += f" | Min: {value.min():.2f}, Max: {value.max():.2f}"

                print(info)
            else:
                print(f"  Key: '{key}' (Not Found)")

        print("--- End Debug Hook ---\n")
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(label="{self.label}")'

@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    """Pack the inputs data.

    **Required Keys:**

    - ``input_key``
    - ``*algorithm_keys``
    - ``*meta_keys``

    **Deleted Keys:**

    All other keys in the dict.

    **Added Keys:**

    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~mmpretrain.structures.DataSample`): The
      annotation info of the sample.

    Args:
        input_key (str): The key of element to feed into the model forwarding.
            Defaults to 'img'.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
        meta_keys (Sequence[str]): The keys of meta information to be saved in
            the data sample. Defaults to :attr:`PackInputs.DEFAULT_META_KEYS`.

    .. admonition:: Default algorithm keys

        Besides the specified ``algorithm_keys``, we will set some default keys
        into the output data sample and do some formatting. Therefore, you
        don't need to set these keys in the ``algorithm_keys``.

        - ``gt_label``: The ground-truth label. The value will be converted
          into a 1-D tensor.
        - ``gt_score``: The ground-truth score. The value will be converted
          into a 1-D tensor.
        - ``mask``: The mask for some self-supervise tasks. The value will
          be converted into a tensor.

    .. admonition:: Default meta keys

        - ``sample_idx``: The id of the image sample.
        - ``img_path``: The path to the image file.
        - ``ori_shape``: The original shape of the image as a tuple (H, W).
        - ``img_shape``: The shape of the image after the pipeline as a
          tuple (H, W).
        - ``scale_factor``: The scale factor between the resized image and
          the original image.
        - ``flip``: A boolean indicating if image flip transform was used.
        - ``flip_direction``: The flipping direction.
    """

    DEFAULT_META_KEYS = ('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction')

    def __init__(self,
                 input_key='img',
                 algorithm_keys=(),
                 meta_keys=DEFAULT_META_KEYS):
        self.input_key = input_key
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [PackInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            if input_.ndim == 2:  # For grayscale image.
                input_ = np.expand_dims(input_, -1)
            if input_.ndim == 3 and not input_.flags.c_contiguous:
                input_ = np.ascontiguousarray(input_.transpose(2, 0, 1))
                input_ = to_tensor(input_)
            elif input_.ndim == 3:
                # convert to tensor first to accelerate, see
                # https://github.com/open-mmlab/mmdetection/pull/9533
                input_ = to_tensor(input_).permute(2, 0, 1).contiguous()
            else:
                # convert input with other shape to tensor without permute,
                # like video input (num_crops, C, T, H, W).
                input_ = to_tensor(input_)
        elif isinstance(input_, Image.Image):
            input_ = F.pil_to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f'Unsupported input type {type(input_)}.')

        return input_

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = self.format_input(input_)

        data_sample = DataSample()

        # Set default keys
        if 'gt_label' in results:
            data_sample.set_gt_label(results['gt_label'])
        if 'gt_score' in results:
            data_sample.set_gt_score(results['gt_score'])
        if 'mask' in results:
            data_sample.set_mask(results['mask'])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='metainfo')

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_key='{self.input_key}', "
        repr_str += f'algorithm_keys={self.algorithm_keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str

@TRANSFORMS.register_module()
class Collect(BaseTransform):
    """Collect and only reserve the specified fields.

    **Required Keys:**

    - ``*keys``

    **Deleted Keys:**

    All keys except those in the argument ``*keys``.

    Args:
        keys (Sequence[str]): The keys of the fields to be collected.
    """

    def __init__(self, keys):
        self.keys = keys

    def transform(self, results):
        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'




@TRANSFORMS.register_module()
class PackDetInputs(BaseTransform):
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = img

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class ToTensor:
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class ImageToTensor:
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and permuted to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img).permute(2, 0, 1).contiguous()

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class Transpose:
    """Transpose some results by given keys.

    Args:
        keys (Sequence[str]): Keys of results to be transposed.
        order (Sequence[int]): Order of transpose.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """Call function to transpose the channel order of data in results.

        Args:
            results (dict): Result dict contains the data to transpose.

        Returns:
            dict: The result dict contains the data transposed to \
                ``self.order``.
        """
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, order={self.order})'


@TRANSFORMS.register_module()
class WrapFieldsToLists:
    """Wrap fields of the data dictionary into lists for evaluation.

    This class can be used as a last step of a test or validation
    pipeline for single image evaluation or inference.

    Example:
        >>> test_pipeline = [
        >>>    dict(type='LoadImageFromFile'),
        >>>    dict(type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
        >>>    dict(type='Pad', size_divisor=32),
        >>>    dict(type='ImageToTensor', keys=['img']),
        >>>    dict(type='Collect', keys=['img']),
        >>>    dict(type='WrapFieldsToLists')
        >>> ]
    """

    def __call__(self, results):
        """Call function to wrap fields into lists.

        Args:
            results (dict): Result dict contains the data to wrap.

        Returns:
            dict: The result dict where value of ``self.keys`` are wrapped \
                into list.
        """

        # Wrap dict fields into lists
        for key, val in results.items():
            results[key] = [val]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@TRANSFORMS.register_module()
class PackTrackInputs(BaseTransform):
    """Pack the inputs data for the multi object tracking and video instance
    segmentation. All the information of images are packed to ``inputs``. All
    the information except images are packed to ``data_samples``. In order to
    get the original annotaiton and meta info, we add `instances` key into meta
    keys.

    Args:
        meta_keys (Sequence[str]): Meta keys to be collected in
            ``data_sample.metainfo``. Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('img_id',
            'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'frame_id', 'is_video_data',
            'video_id', 'video_length', 'instances').
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_instances_ids': 'instances_ids'
    }

    def __init__(self,
                 meta_keys: Optional[dict] = None,
                 default_meta_keys: tuple = ('img_id', 'img_path', 'ori_shape',
                                             'img_shape', 'scale_factor',
                                             'flip', 'flip_direction',
                                             'frame_id', 'video_id',
                                             'video_length',
                                             'ori_video_length', 'instances')):
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.
        Args:
            results (dict): Result dict from the data pipeline.
        Returns:
            dict:
            - 'inputs' (dict[Tensor]): The forward data of models.
            - 'data_samples' (obj:`TrackDataSample`): The annotation info of
                the samples.
        """
        packed_results = dict()
        packed_results['inputs'] = dict()

        # 1. Pack images
        if 'img' in results:
            imgs = results['img']
            imgs = np.stack(imgs, axis=0)
            imgs = imgs.transpose(0, 3, 1, 2)
            packed_results['inputs'] = to_tensor(imgs)

        # 2. Pack InstanceData
        if 'gt_ignore_flags' in results:
            gt_ignore_flags_list = results['gt_ignore_flags']
            valid_idx_list, ignore_idx_list = [], []
            for gt_ignore_flags in gt_ignore_flags_list:
                valid_idx = np.where(gt_ignore_flags == 0)[0]
                ignore_idx = np.where(gt_ignore_flags == 1)[0]
                valid_idx_list.append(valid_idx)
                ignore_idx_list.append(ignore_idx)

        assert 'img_id' in results, "'img_id' must contained in the results "
        'for counting the number of images'

        num_imgs = len(results['img_id'])
        instance_data_list = [InstanceData() for _ in range(num_imgs)]
        ignore_instance_data_list = [InstanceData() for _ in range(num_imgs)]

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks':
                mapped_key = self.mapping_table[key]
                gt_masks_list = results[key]
                if 'gt_ignore_flags' in results:
                    for i, gt_mask in enumerate(gt_masks_list):
                        valid_idx, ignore_idx = valid_idx_list[
                            i], ignore_idx_list[i]
                        instance_data_list[i][mapped_key] = gt_mask[valid_idx]
                        ignore_instance_data_list[i][mapped_key] = gt_mask[
                            ignore_idx]

                else:
                    for i, gt_mask in enumerate(gt_masks_list):
                        instance_data_list[i][mapped_key] = gt_mask

            else:
                anns_list = results[key]
                if 'gt_ignore_flags' in results:
                    for i, ann in enumerate(anns_list):
                        valid_idx, ignore_idx = valid_idx_list[
                            i], ignore_idx_list[i]
                        instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(
                                ann[valid_idx])
                        ignore_instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(
                                ann[ignore_idx])
                else:
                    for i, ann in enumerate(anns_list):
                        instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(ann)

        det_data_samples_list = []
        for i in range(num_imgs):
            det_data_sample = DetDataSample()
            det_data_sample.gt_instances = instance_data_list[i]
            det_data_sample.ignored_instances = ignore_instance_data_list[i]
            det_data_samples_list.append(det_data_sample)

        # 3. Pack metainfo
        for key in self.meta_keys:
            if key not in results:
                continue
            img_metas_list = results[key]
            for i, img_meta in enumerate(img_metas_list):
                det_data_samples_list[i].set_metainfo({f'{key}': img_meta})

        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = det_data_samples_list
        if 'key_frame_flags' in results:
            key_frame_flags = np.asarray(results['key_frame_flags'])
            key_frames_inds = np.where(key_frame_flags)[0].tolist()
            ref_frames_inds = np.where(~key_frame_flags)[0].tolist()
            track_data_sample.set_metainfo(
                dict(key_frames_inds=key_frames_inds))
            track_data_sample.set_metainfo(
                dict(ref_frames_inds=ref_frames_inds))

        packed_results['data_samples'] = track_data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'meta_keys={self.meta_keys}, '
        repr_str += f'default_meta_keys={self.default_meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackReIDInputs(BaseTransform):
    """Pack the inputs data for the ReID. The ``meta_info`` item is always
    populated. The contents of the ``meta_info`` dictionary depends on
    ``meta_keys``. By default this includes:

        - ``img_path``: path to the image file.
        - ``ori_shape``: original shape of the image as a tuple (H, W).
        - ``img_shape``: shape of the image input to the network as a tuple
            (H, W). Note that images may be zero padded on the bottom/right
          if the batch tensor is larger than this shape.
        - ``scale``: scale of the image as a tuple (W, H).
        - ``scale_factor``: a float indicating the pre-processing scale.
        -  ``flip``: a boolean indicating if image flip transform was used.
        - ``flip_direction``: the flipping direction.
    Args:
        meta_keys (Sequence[str], optional): The meta keys to saved in the
            ``metainfo`` of the packed ``data_sample``.
    """
    default_meta_keys = ('img_path', 'ori_shape', 'img_shape', 'scale',
                         'scale_factor')

    def __init__(self, meta_keys: Sequence[str] = ()) -> None:
        self.meta_keys = self.default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple.'
            self.meta_keys += meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.
        Args:
            results (dict): Result dict from the data pipeline.
        Returns:
            dict:
            - 'inputs' (dict[Tensor]): The forward data of models.
            - 'data_samples' (obj:`ReIDDataSample`): The meta info of the
                sample.
        """
        packed_results = dict(inputs=dict(), data_samples=None)
        assert 'img' in results, 'Missing the key ``img``.'
        _type = type(results['img'])
        label = results['gt_label']

        if _type == list:
            img = results['img']
            label = np.stack(label, axis=0)  # (N,)
            assert all([type(v) == _type for v in results.values()]), \
                'All items in the results must have the same type.'
        else:
            img = [results['img']]

        img = np.stack(img, axis=3)  # (H, W, C, N)
        img = img.transpose(3, 2, 0, 1)  # (N, C, H, W)
        img = np.ascontiguousarray(img)

        packed_results['inputs'] = to_tensor(img)

        data_sample = ReIDDataSample()
        data_sample.set_gt_label(label)

        meta_info = dict()
        for key in self.meta_keys:
            meta_info[key] = results[key]
        data_sample.set_metainfo(meta_info)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
