import math
from typing import Optional, Sequence

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class RatioSampler(Sampler[int]):
    """Distributed fixed-ratio sampler for ConcatDataset.

    Every local GPU batch contains a fixed number of samples from each
    sub-dataset. Smaller datasets are cyclically re-sampled if necessary.

    Important:
        - This sampler yields individual indices.
        - DataLoader batch_size MUST equal sampler batch_size.
        - num_batches_per_epoch can be used to decouple epoch length from
          the size of the largest dataset.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        source_ratio: Sequence[float],
        shuffle: bool = True,
        seed: Optional[int] = None,
        num_batches_per_epoch: Optional[int] = None,
    ):
        if not hasattr(dataset, 'datasets'):
            raise TypeError(
                'RatioSampler requires a ConcatDataset with a `datasets` '
                'attribute.'
            )

        if not hasattr(dataset, 'cumulative_sizes'):
            raise TypeError(
                'RatioSampler requires a ConcatDataset with '
                '`cumulative_sizes`.'
            )

        if len(source_ratio) != len(dataset.datasets):
            raise ValueError(
                f'Got {len(source_ratio)} ratios for '
                f'{len(dataset.datasets)} datasets.'
            )

        if batch_size <= 0:
            raise ValueError('batch_size must be positive.')

        if any(r <= 0 for r in source_ratio):
            raise ValueError('Every source ratio must be positive.')

        if num_batches_per_epoch is not None and num_batches_per_epoch <= 0:
            raise ValueError(
                'num_batches_per_epoch must be positive when specified.'
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.source_ratio = list(source_ratio)
        self.shuffle = bool(shuffle)
        self.epoch = 0

        self.rank, self.world_size = get_dist_info()
        self.seed = (
            int(sync_random_seed())
            if seed is None else int(seed)
        )

        self.dataset_sizes = [
            len(sub_dataset) for sub_dataset in dataset.datasets
        ]

        if any(size == 0 for size in self.dataset_sizes):
            raise ValueError(
                'Every source dataset must be non-empty, but got '
                f'{self.dataset_sizes}.'
            )

        # Global start offset of every child dataset in ConcatDataset.
        self.dataset_offsets = [0] + list(dataset.cumulative_sizes[:-1])

        self.num_per_source = self._allocate_batch_counts(
            batch_size=self.batch_size,
            ratios=self.source_ratio,
        )

        if any(count == 0 for count in self.num_per_source):
            raise ValueError(
                'The selected batch size and ratios yield zero samples '
                f'for at least one domain: {self.num_per_source}.'
            )

        # Original behavior: epoch length is controlled by the largest domain.
        natural_num_batches = max(
            math.ceil(
                dataset_size / (samples_per_rank * self.world_size)
            )
            for dataset_size, samples_per_rank in zip(
                self.dataset_sizes,
                self.num_per_source,
            )
        )

        # New behavior: fixed epoch length if explicitly specified.
        self.num_batches = (
            int(num_batches_per_epoch)
            if num_batches_per_epoch is not None
            else natural_num_batches
        )

        self.num_samples = self.num_batches * self.batch_size

    @staticmethod
    def _allocate_batch_counts(batch_size, ratios):
        """Convert float ratios to integer counts summing to batch_size."""
        ratio_tensor = torch.tensor(ratios, dtype=torch.float64)
        raw_counts = ratio_tensor / ratio_tensor.sum() * batch_size

        counts = torch.floor(raw_counts).to(torch.long)
        remaining = batch_size - int(counts.sum())

        if remaining > 0:
            fractions = raw_counts - counts
            order = torch.argsort(fractions, descending=True)

            for index in order[:remaining]:
                counts[index] += 1

        return counts.tolist()

    def _build_source_stream(
        self,
        dataset_size: int,
        required_size: int,
        generator: torch.Generator,
    ):
        """Create an index stream, repeatedly cycling a small dataset."""
        stream = []

        while len(stream) < required_size:
            if self.shuffle:
                indices = torch.randperm(
                    dataset_size,
                    generator=generator,
                ).tolist()
            else:
                indices = list(range(dataset_size))

            stream.extend(indices)

        return stream[:required_size]

    def __iter__(self):
        source_streams = []

        # Build a global stream per source domain first.
        # Afterwards every DDP rank consumes a different contiguous chunk.
        for source_id, dataset_size in enumerate(self.dataset_sizes):
            count_per_local_batch = self.num_per_source[source_id]

            required_global_samples = (
                self.num_batches
                * self.world_size
                * count_per_local_batch
            )

            source_generator = torch.Generator()
            source_generator.manual_seed(
                self.seed + self.epoch * 1009 + source_id
            )

            stream = self._build_source_stream(
                dataset_size=dataset_size,
                required_size=required_global_samples,
                generator=source_generator,
            )

            source_streams.append(stream)

        local_generator = torch.Generator()
        local_generator.manual_seed(
            self.seed + self.epoch * 1009 + self.rank + 100_000
        )

        for batch_id in range(self.num_batches):
            batch = []

            for source_id, count in enumerate(self.num_per_source):
                global_batch_position = (
                    batch_id * self.world_size + self.rank
                )

                start = global_batch_position * count
                end = start + count

                source_indices = source_streams[source_id][start:end]
                offset = self.dataset_offsets[source_id]

                batch.extend(
                    local_index + offset
                    for local_index in source_indices
                )

            if len(batch) != self.batch_size:
                raise RuntimeError(
                    'RatioSampler generated an incomplete batch: '
                    f'{len(batch)} instead of {self.batch_size}.'
                )

            # Keep fixed ratio but randomize sample ordering inside one batch.
            if self.shuffle:
                permutation = torch.randperm(
                    len(batch),
                    generator=local_generator,
                ).tolist()

                batch = [batch[index] for index in permutation]

            yield from batch

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
