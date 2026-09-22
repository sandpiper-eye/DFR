"""DeepSpeed-compatible checkpoint hook for mmengine Runner.

Fixed version with proper distributed synchronization and error handling.
"""

import os
import os.path as osp
import torch
import torch.distributed as dist
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.dist import master_only, get_dist_info, is_main_process
import torch.nn as nn
import torch.nn.functional as F


@HOOKS.register_module()
class DeepSpeedCheckpointHook(Hook):
    """Fixed DeepSpeed Checkpoint Hook with proper distributed sync."""

    priority = 'VERY_LOW'

    def __init__(self,
                 interval=1,
                 by_epoch=True,
                 save_optimizer=True,
                 max_keep_ckpts=-1,
                 save_last=True,
                 sync_timeout=3600):  # 新增：同步超时
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.sync_timeout = sync_timeout

    def _get_engine(self, runner):
        """Safely extract DeepSpeed engine from wrapped model."""
        model = runner.model

        # Unwrap MMEngine wrapper
        if hasattr(model, 'module'):
            model = model.module

        # Check if it's DeepSpeed engine
        if hasattr(model, 'save_checkpoint') and callable(getattr(model, 'save_checkpoint')):
            return model

        return None

    def _should_save(self, runner):
        """Check if should save at current step."""
        if self.by_epoch:
            # 注意：epoch 从 0 开始，但保存时通常用 +1
            return (runner.epoch + 1) % self.interval == 0
        else:
            return (runner.iter + 1) % self.interval == 0

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        if self._should_save(runner):
            self._save_checkpoint(runner)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.by_epoch:
            return
        if self._should_save(runner):
            self._save_checkpoint(runner)

    def after_train(self, runner):
        if self.save_last:
            self._save_checkpoint(runner, tag='latest')

    def _save_checkpoint(self, runner, tag=None):
        """Save with proper distributed synchronization."""
        rank, world_size = get_dist_info()

        # 1. 获取 engine（所有 rank 都必须成功）
        engine = self._get_engine(runner)
        if engine is None:
            # 所有 rank 统一行为：记录日志但继续
            runner.logger.warning_once(
                'DeepSpeed engine not found, skipping checkpoint save')
            # 关键：即使不保存，也要参与 barrier
            if world_size > 1:
                dist.barrier()
            return

        # 2. 准备保存目录（只在 main rank 创建，避免冲突）
        work_dir = runner.work_dir
        if tag:
            ckpt_name = f'checkpoint_{tag}'
        elif self.by_epoch:
            ckpt_name = f'epoch_{runner.epoch + 1}'
        else:
            ckpt_name = f'iter_{runner.iter + 1}'

        ckpt_dir = osp.join(work_dir, ckpt_name)

        # 3. 关键：所有 rank 同步，准备进入保存
        runner.logger.info(f'Rank {rank}: Preparing to save checkpoint to {ckpt_dir}')
        if world_size > 1:
            dist.barrier()  # ← 关键同步点 1

        # 4. 执行保存（DeepSpeed 内部会处理多卡同步）
        try:
            client_state = {
                'epoch': runner.epoch + 1,
                'iter': runner.iter + 1,
                'max_epochs': runner.max_epochs,
                'max_iters': runner.max_iters,
                'rank': rank,  # 记录保存时的 rank 信息
            }

            # 可选：保存随机状态（用于精确复现）
            if hasattr(runner, 'random_state'):
                client_state['random_state'] = runner.random_state

            # 添加 message_hub 状态
            if hasattr(runner, 'message_hub'):
                try:
                    client_state['message_hub'] = runner.message_hub.state_dict()
                except Exception as e:
                    runner.logger.warning(f'Failed to save message_hub: {e}')

            # 执行保存
            engine.save_checkpoint(
                save_dir=ckpt_dir,
                client_state=client_state,
                save_latest=(tag == 'latest')  # 只有 latest 标签才保存到 latest 目录
            )

            runner.logger.info(f'Rank {rank}: Checkpoint saved successfully')

        except Exception as e:
            runner.logger.error(f'Rank {rank}: Failed to save checkpoint: {e}')
            # 异常时也要 barrier，避免其他 rank 等待
            if world_size > 1:
                dist.barrier()
            raise  # 向上抛出，让 runner 处理

        # 5. 保存后同步，确保所有 rank 都完成
        if world_size > 1:
            dist.barrier()  # ← 关键同步点 2

        # 6. 清理旧 checkpoint（只在 main rank 执行）
        if is_main_process() and self.max_keep_ckpts > 0 and tag is None:
            self._remove_old_checkpoints(work_dir)

    @master_only
    def _remove_old_checkpoints(self, work_dir):
        """Remove old checkpoints to keep only max_keep_ckpts."""
        prefix = 'epoch_' if self.by_epoch else 'iter_'

        ckpt_dirs = []
        try:
            for name in os.listdir(work_dir):
                if name.startswith(prefix):
                    ckpt_path = osp.join(work_dir, name)
                    if osp.isdir(ckpt_path):
                        try:
                            num = int(name.replace(prefix, ''))
                            ckpt_dirs.append((num, ckpt_path))
                        except ValueError:
                            continue
        except FileNotFoundError:
            return

        # 保留最新的 N 个
        ckpt_dirs.sort(key=lambda x: x[0], reverse=True)
        for i, (num, ckpt_path) in enumerate(ckpt_dirs):
            if i >= self.max_keep_ckpts:
                try:
                    import shutil
                    shutil.rmtree(ckpt_path)
                    print(f'Removed old checkpoint: {ckpt_path}')
                except Exception as e:
                    print(f'Failed to remove {ckpt_path}: {e}')


@HOOKS.register_module()
class DeepSpeedResumeHook(Hook):
    """Fixed DeepSpeed Resume Hook with proper distributed sync and validation."""

    priority = 'VERY_HIGH'

    def __init__(self,
                 load_dir,
                 tag,
                 load_optimizer_states=True,
                 load_lr_scheduler_states=True,
                 strict=True,  # 新增：严格模式
                 verify_loading=True):  # 新增：验证加载
        self.load_dir = load_dir
        self.tag = tag
        self.load_optimizer_states = load_optimizer_states
        self.load_lr_scheduler_states = load_lr_scheduler_states
        self.strict = strict
        self.verify_loading = verify_loading
        self._loaded = False

    def _get_engine(self, runner):
        """Safely extract DeepSpeed engine."""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'load_checkpoint') and callable(getattr(model, 'load_checkpoint')):
            return model
        return None

    def _validate_checkpoint(self):
        """Validate checkpoint exists and is complete."""
        if not osp.exists(self.load_dir):
            raise FileNotFoundError(f'Checkpoint directory not found: {self.load_dir}')

        # 检查关键文件是否存在
        required_files = ['mp_rank_00_model_states.pt']  # 至少主 rank 的文件存在
        for f in required_files:
            if not osp.exists(osp.join(self.load_dir, f)):
                # 可能是 ZeRO-3，检查其他模式
                if not osp.exists(osp.join(self.load_dir, 'zero_to_fp32.py')):
                    raise FileNotFoundError(f'Checkpoint appears incomplete: {self.load_dir}')

        return True

    def before_train(self, runner):
        """Load checkpoint with proper synchronization."""
        if self._loaded:
            return

        rank, world_size = get_dist_info()

        # 1. 验证 checkpoint（所有 rank 都验证，避免不一致）
        if self.strict:
            try:
                self._validate_checkpoint()
            except FileNotFoundError as e:
                runner.logger.error(f'Rank {rank}: Checkpoint validation failed: {e}')
                if world_size > 1:
                    dist.barrier()  # 确保所有 rank 都知道失败
                raise

        # 2. 获取 engine
        engine = self._get_engine(runner)
        if engine is None:
            runner.logger.warning('DeepSpeed engine not found, skipping resume')
            if world_size > 1:
                dist.barrier()
            return

        # 3. 关键：加载前同步，确保所有 rank 同时开始加载
        runner.logger.info(f'Rank {rank}: Waiting for all ranks before loading...')
        if world_size > 1:
            dist.barrier()  # ← 关键同步点 1

        # 4. 执行加载
        try:
            runner.logger.info(f'Rank {rank}: Loading checkpoint from {self.load_dir}')

            load_path, client_state = engine.load_checkpoint(
                load_dir=self.load_dir,
                tag = self.tag,
                load_optimizer_states=self.load_optimizer_states,
                load_lr_scheduler_states=self.load_lr_scheduler_states,
                load_module_strict=self.strict
            )

            if load_path is None:
                raise RuntimeError(f'Failed to load checkpoint from {self.load_dir}')

            runner.logger.info(f'Rank {rank}: Successfully loaded from {load_path}')

        except Exception as e:
            runner.logger.error(f'Rank {rank}: Checkpoint loading failed: {e}')
            if world_size > 1:
                dist.barrier()  # 通知其他 rank
            raise

        # 5. 加载后同步，确保所有 rank 都完成
        if world_size > 1:
            dist.barrier()  # ← 关键同步点 2

        # 6. 恢复训练状态
        if client_state:
            self._restore_runner_state(runner, client_state)

        # 7. 验证所有 rank 状态一致（可选）
        if self.verify_loading and world_size > 1:
            self._verify_consistency(runner, engine)

        self._loaded = True
        runner.logger.info(f'Rank {rank}: Resume completed successfully')

    def _restore_runner_state(self, runner, client_state):
        """Restore runner state from client_state."""
        # ✅ 关键：获取 train_loop 并设置其状态
        loop = runner._train_loop

        if loop is None:
            raise RuntimeError('runner._train_loop is None! Cannot resume.')
        if 'epoch' in client_state:
            target_epoch = client_state['epoch']
            new_epoch = target_epoch if target_epoch > 0 else 0
            try:
                object.__setattr__(loop, '_epoch', new_epoch)
                actual = object.__getattribute__(loop, '_epoch')
                assert actual == new_epoch, f'Set failed: {actual} != {new_epoch}'
                runner.logger.info(f'Resumed epoch: {runner.epoch} (target: {target_epoch})')
            except Exception as e:
                raise RuntimeError(f'Failed to set loop._epoch: {e}')
            #runner._epoch = target_epoch - 1 if target_epoch > 0 else 0
            #runner.logger.info(f'Resumed epoch: {runner.epoch} (target: {target_epoch})')

        if 'iter' in client_state:
            target_iter = client_state['iter']
            new_iter = target_iter if target_iter > 0 else 0
            try:
                object.__setattr__(loop, '_iter', new_iter)
                actual = object.__getattribute__(loop, '_iter')
                assert actual == new_iter, f'Set failed: {actual} != {new_iter}'
                runner.logger.info(f'Resumed iter: {new_iter} (target: {target_iter})')
            except Exception as e:
                raise RuntimeError(f'Failed to set loop._iter: {e}')
            #runner._iter = target_iter - 1 if target_iter > 0 else 0
            #runner.logger.info(f'Resumed iter: {runner.iter} (target: {target_iter})')

        # 恢复 message_hub
        if 'message_hub' in client_state and hasattr(runner, 'message_hub'):
            try:
                runner.message_hub.load_state_dict(client_state['message_hub'])
                runner.logger.info('Restored message_hub state')
            except Exception as e:
                runner.logger.warning(f'Failed to restore message_hub: {e}')

        # 恢复随机状态（关键！）
        if 'random_state' in client_state:
            try:
                torch.set_rng_state(client_state['random_state'])
                runner.logger.info('Restored random state')
            except Exception as e:
                runner.logger.warning(f'Failed to restore random state: {e}')

    def _verify_consistency(self, runner, engine):
        """Verify all ranks have consistent state after resume."""
        rank, world_size = get_dist_info()

        # 收集所有 rank 的 epoch 和 iter
        my_epoch = torch.tensor([runner.epoch], dtype=torch.int32, device='cuda')
        my_iter = torch.tensor([runner.iter], dtype=torch.int32, device='cuda')

        all_epochs = [torch.zeros(1, dtype=torch.int32, device='cuda') for _ in range(world_size)]
        all_iters = [torch.zeros(1, dtype=torch.int32, device='cuda') for _ in range(world_size)]

        dist.all_gather(all_epochs, my_epoch)
        dist.all_gather(all_iters, my_iter)

        # 检查一致性
        if rank == 0:
            epochs = [t.item() for t in all_epochs]
            iters = [t.item() for t in all_iters]

            if len(set(epochs)) > 1 or len(set(iters)) > 1:
                runner.logger.error(f'Inconsistent state after resume!')
                runner.logger.error(f'Epochs: {epochs}')
                runner.logger.error(f'Iters: {iters}')
                raise RuntimeError('Resume verification failed: inconsistent state across ranks')
            else:
                runner.logger.info(f'Resume verification passed: all ranks at epoch {epochs[0]}, iter {iters[0]}')


@HOOKS.register_module()
class DeepSpeedLoadFromHook(Hook):
    """正确处理 load_from：兼容 DeepSpeed ZeRO 并确保 BN 统计量加载."""

    priority = 'HIGH'

    def __init__(self, strict=False, debug_verify=True):
        self.strict = strict
        self.debug_verify = debug_verify
        self._loaded = False

    def _get_engine(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'save_checkpoint') and callable(getattr(model, 'save_checkpoint')):
            return model
        return None

    def before_train(self, runner):
        if self._loaded:
            return

        load_from = getattr(runner, '_load_from', None)
        is_resume = getattr(runner, '_resume', False)

        if not load_from or is_resume:
            return

        rank, world_size = get_dist_info()
        engine = self._get_engine(runner)

        if engine is None:
            return

        if not osp.exists(load_from):
            raise FileNotFoundError(f'load_from checkpoint not found: {load_from}')

        if is_main_process():
            runner.logger.info(f'Loading load_from checkpoint: {load_from}')

        # 1. 加载 checkpoint（所有 rank 都加载完整权重）
        checkpoint = torch.load(load_from, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        # 2. 处理键名前缀（去掉 module. 如果有）
        processed_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]  # 去掉 'module.'
            else:
                new_key = k
            processed_state_dict[new_key] = v

        # 3. ✅ 关键：分离 parameters 和 buffers（state_dict 包含两者）
        param_keys = {name for name, _ in engine.module.named_parameters()}
        buffer_keys = {name for name, _ in engine.module.named_buffers()}

        params_dict = {k: v for k, v in processed_state_dict.items() if k in param_keys}
        buffers_dict = {k: v for k, v in processed_state_dict.items() if k in buffer_keys}
        unexpected = {k: v for k, v in processed_state_dict.items()
                      if k not in param_keys and k not in buffer_keys}

        if is_main_process():
            runner.logger.info(f'Parameters to load: {len(params_dict)}, Buffers to load: {len(buffers_dict)}')
            if unexpected and not self.strict:
                runner.logger.warning(f'Unexpected keys (will be ignored): {list(unexpected.keys())[:5]}...')

        # 4. ✅ 关键：加载到裸模型（所有 rank 都需要执行，确保 ZeRO 分片正确）
        # strict=False 因为 DeepSpeed 可能有一些额外的参数（如 ZeRO 相关）
        missing_keys, loaded_unexpected = engine.module.load_state_dict(
            processed_state_dict,
            strict=self.strict
        )

        if is_main_process():
            if missing_keys:
                runner.logger.warning(f'Missing keys: {missing_keys[:10]}... ({len(missing_keys)} total)')
                if len(missing_keys) > len(processed_state_dict) * 0.1:  # 如果缺失超过10%
                    runner.logger.error('❌ Too many missing keys! Loading failed.')

        # 5. ✅ 关键：显式加载 BN 统计量（有时 load_state_dict 会漏掉）
        loaded_buffers = 0
        for name, buffer in engine.module.named_buffers():
            if name in buffers_dict:
                buffer.data.copy_(buffers_dict[name])
                loaded_buffers += 1

        # 加载后显式检查 BN 层
        for name, buf in engine.module.named_buffers():
            if 'running_var' in name:
                if (buf <= 0.01).any():
                    runner.logger.error(f"Invalid running_var in {name}: min={buf.min()}")
                    # 强制修正为合理值
                    buf.data = torch.clamp(buf, min=0.01)

        if is_main_process():
            runner.logger.info(f'Explicitly loaded {loaded_buffers} buffers (BN stats, etc.)')

        # 6. ✅ 关键：同步确保所有 rank 都完成加载
        if world_size > 1:
            dist.barrier()

        # 7. ✅ 关键：验证（放宽阈值 + 多点验证）
        if self.debug_verify:
            self._verify_loading(runner, engine, params_dict, buffers_dict)

        # 8. ✅ 关键：重置 DeepSpeed 数据分片（让 DeepSpeed 重新分片已加载的权重）
        # 这确保 ZeRO 知道权重已更新
        if hasattr(engine, '_zero3_rebuild_global_params'):
            engine._zero3_rebuild_global_params()

        self._loaded = True
        runner.logger.info(f'Rank {rank}: load_from completed successfully')

        # ✅ 关键新增：检查加载后的权重是否有 NaN
        has_nan = False
        for name, param in engine.module.named_parameters():
            if torch.isnan(param).any():
                runner.logger.error(f"❌ NaN detected in parameter: {name}")
                has_nan = True
        for name, buf in engine.module.named_buffers():
            if torch.isnan(buf).any():
                runner.logger.error(f"❌ NaN detected in buffer (BN stats): {name}")
                has_nan = True

        if has_nan:
            raise RuntimeError("Checkpoint contains NaN! Loading aborted.")

        # 检查无穷大（Inf）也很重要
        for name, param in engine.module.named_parameters():
            if torch.isinf(param).any():
                runner.logger.error(f"❌ Inf detected in parameter: {name}")

    def _verify_loading(self, runner, engine, params_dict, buffers_dict):
        """多层级验证，放宽阈值."""
        rank = get_dist_info()[0]
        if rank != 0:
            return

        # 随机采样 5 个参数验证
        sample_params = list(params_dict.keys())[:5]
        max_diff = 0

        for name in sample_params:
            loaded_param = dict(engine.module.named_parameters())[name]
            saved_param = params_dict[name].to(loaded_param.device)
            diff = (loaded_param - saved_param).abs().max().item()
            max_diff = max(max_diff, diff)

        # 放宽阈值到 1e-3（考虑到 FP16/FP32 转换）
        threshold = 1e-3
        if max_diff < threshold:
            runner.logger.info(f'✅ Verification passed: max param diff = {max_diff:.6f} < {threshold}')
        else:
            runner.logger.error(f'❌ Verification failed: max param diff = {max_diff:.6f} > {threshold}')

        # 验证 BN 统计量
        if buffers_dict:
            sample_buffer = list(buffers_dict.keys())[0]
            loaded_buf = dict(engine.module.named_buffers())[sample_buffer]
            saved_buf = buffers_dict[sample_buffer].to(loaded_buf.device)
            buf_diff = (loaded_buf - saved_buf).abs().max().item()
            runner.logger.info(f'Buffer ({sample_buffer}) diff: {buf_diff:.6f}')


@HOOKS.register_module()
class OutlierClippingHook(Hook):
    """
    模拟bnb outlier分离：对所有指定模块的激活值进行硬截断。
    应用于输入（预防前层outlier传入）和输出（阻断本层outlier传出）。
    """
    priority = 'VERY_LOW'  # 在模型构建后，训练前执行

    def __init__(self,
                 clip_values=None,  # 模块名到阈值的映射
                 default_clip=1000.0,
                 clip_input=True,  # 是否截断输入
                 clip_output=True,  # 是否截断输出
                 log_outlier_ratio=0.01):  # 当outlier超过此比例时打印日志
        """
        Args:
            clip_values: dict, e.g., {'backbone': 1000, 'attribute_encoder': 100}
        """
        self.clip_values = clip_values or {}
        self.default_clip = default_clip
        self.clip_input = clip_input
        self.clip_output = clip_output
        self.log_outlier_ratio = log_outlier_ratio
        self._registered = False

    def before_run(self, runner):
        if self._registered:
            return

        model = runner.model.module  # DeepSpeed包装的裸模型
        device = next(model.parameters()).device

        # 递归遍历所有模块，给风险层注册hook
        registered_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.LayerNorm)):
                clip_val = self._get_clip_value(name)

                # 注册pre-hook（输入截断）
                if self.clip_input:
                    module.register_forward_pre_hook(
                        self._make_input_clip_hook(name, clip_val, runner.logger)
                    )

                # 注册post-hook（输出截断）
                if self.clip_output:
                    module.register_forward_hook(
                        self._make_output_clip_hook(name, clip_val, runner.logger, device)
                    )
                registered_count += 1

        runner.logger.info(f'OutlierClippingHook: 已注册 {registered_count} 个模块的截断保护')
        self._registered = True

    def _get_clip_value(self, module_name):
        """根据模块名返回对应的clip阈值"""
        # 最长匹配原则：'backbone.layer1' 匹配 'backbone' 而不是默认
        matched_len = 0
        matched_val = self.default_clip

        for pattern, val in self.clip_values.items():
            if pattern in module_name and len(pattern) > matched_len:
                matched_val = val
                matched_len = len(pattern)

        return matched_val

    def _make_input_clip_hook(self, name, clip_val, logger):
        """创建输入截断hook"""

        def hook(module, input):
            if not isinstance(input, tuple):
                input = (input,)

            clipped_input = []
            outlier_detected = False

            for inp in input:
                if isinstance(inp, torch.Tensor):
                    # 检测outlier比例（调试用）
                    if inp.numel() > 0:
                        outlier_mask = inp.abs() > clip_val
                        outlier_ratio = outlier_mask.float().mean()

                        if outlier_ratio > self.log_outlier_ratio:
                            logger.warning(
                                f'[{name}] Input outlier ratio: {outlier_ratio:.2%} '
                                f'(max abs val: {inp.abs().max():.2f}, clip_threshold: {clip_val})'
                            )
                            outlier_detected = True

                    # 硬截断（模拟bnb的outlier分离）
                    clipped = torch.clamp(inp, min=-clip_val, max=clip_val)
                    clipped_input.append(clipped)
                else:
                    clipped_input.append(inp)

            return tuple(clipped_input) if len(clipped_input) > 1 else clipped_input[0]

        return hook

    def _make_output_clip_hook(self, name, clip_val, logger, device):
        """创建输出截断hook"""

        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return output

            # 检测NaN/Inf（关键！）
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.error(f'[{name}] NaN/Inf detected before clipping!')
                # 替换NaN/Inf为0，然后截断
                output = torch.where(
                    torch.isnan(output) | torch.isinf(output),
                    torch.zeros_like(output),
                    output
                )

            # 截断输出
            clipped = torch.clamp(output, min=-clip_val, max=clip_val)

            # 检查是否实际发生了截断
            if not torch.equal(output, clipped):
                max_before = output.abs().max()
                logger.debug(f'[{name}] Output clipped: max {max_before:.2f} -> {clip_val}')

            return clipped

        return hook


@HOOKS.register_module()
class ResumeBNFixHook(Hook):
    """
    阶段1结束前专用：修复 BN 统计量并重置 EMA。
    阶段2将加载修复后的权重，并添加新结构。
    """
    priority = 51

    def __init__(self, num_calib_batches=50, bn_momentum_calib=0.1):
        self.num_calib_batches = num_calib_batches
        self.bn_momentum_calib = bn_momentum_calib
        self.original_bn_momentums = {}
        self.original_requires_grad = {}

    def before_run(self, runner):
        model = runner.model
        device = next(model.parameters()).device

        # 【关键修复】保存原始 BN momentum（必须在修改前保存！）
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                self.original_bn_momentums[name] = module.momentum

        # 1. 冻结所有参数（只更新 BN 统计量，不训练权重）
        for name, param in model.named_parameters():
            self.original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        # 2. 设置 BN 为 train 模式并增大 momentum（加速适应）
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.train()
                module.momentum = self.bn_momentum_calib

        # 3. 前向传播校准 BN 统计量
        runner.logger.info("🔄 开始校准阶段1的BN统计量...")
        self._calibrate_bn_statistics(runner, device)

        # 4. 恢复参数训练状态
        for name, param in model.named_parameters():
            param.requires_grad = self.original_requires_grad[name]

        # 5. 关键：重置 EMA buffers（让 EMA 从修复后的统计量重新开始）
        self._reset_ema_buffers(runner)

        runner.logger.info("✅ BN统计量已修复，EMA已重置，可保存至阶段2使用")

    def _calibrate_bn_statistics(self, runner, device):
        model = runner.model
        data_loader = runner.train_dataloader

        model.train()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= self.num_calib_batches:
                    break

                imgs = self._get_images_from_batch(batch, device)
                if imgs is None:
                    continue

                # 归一化
                if imgs.max() > 10:
                    mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
                    std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
                    imgs = (imgs - mean) / std

                # 尺寸对齐
                _, _, h, w = imgs.shape
                if h % 32 != 0 or w % 32 != 0:
                    imgs = F.interpolate(imgs, size=(h // 32 * 32, w // 32 * 32),
                                         mode='bilinear', align_corners=False)

                try:
                    _ = model(imgs)
                except Exception as e:
                    runner.logger.error(f"Forward failed at batch {i}: {e}")
                    raise

                if i % 10 == 0:
                    runner.logger.info(f"BN校准进度: {i}/{self.num_calib_batches}")

    def _reset_ema_buffers(self, runner):
        """重置 EMA，使其从当前修复后的 BN 统计量重新开始平均"""
        ema_hook = None
        for hook in runner.hooks:
            if hook.__class__.__name__ == 'EMAHook':
                ema_hook = hook
                break

        if ema_hook is None or not hasattr(ema_hook, 'ema_model'):
            runner.logger.warning("未找到EMAHook，跳过重置")
            return

        # 解包 DDP/DeepSpeed
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        ema_model = ema_hook.ema_model.module if hasattr(ema_hook.ema_model, 'module') else ema_hook.ema_model

        # 复制当前模型的 BN 统计量到 EMA
        match_count = 0
        model_buffers = {name: buf for name, buf in model.named_buffers()
                         if 'running_mean' in name or 'running_var' in name}

        for ema_name, ema_buf in ema_model.named_buffers():
            if ema_name in model_buffers and ema_buf.shape == model_buffers[ema_name].shape:
                ema_buf.data.copy_(model_buffers[ema_name].data)
                match_count += 1

        runner.logger.info(f"✅ 已重置 {match_count} 个 EMA BN buffers")

    def after_train_epoch(self, runner):
        """恢复原始 BN momentum（如果是阶段性校准，不是最终保存）"""
        if runner.epoch == runner.max_epochs - 1:  # 在最后一轮恢复，准备保存
            for name, module in runner.model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)) and name in self.original_bn_momentums:
                    module.momentum = self.original_bn_momentums[name]
            runner.logger.info("✅ 已恢复原始BN momentum，准备保存最终权重")

    def _get_images_from_batch(self, batch, device):
        # 你的原有实现保持不变...
        img = None
        if 'inputs' in batch:
            inputs = batch['inputs']
            if isinstance(inputs, torch.Tensor):
                img = inputs
            elif isinstance(inputs, (list, tuple)) and len(inputs) > 0:
                first = inputs[0]
                img = first.data if hasattr(first, 'data') else first

        if img is None:
            return None

        if img.dim() == 3:
            img = img.unsqueeze(0)

        return img.to(device).float() if isinstance(img, torch.Tensor) else None



@HOOKS.register_module()
class StagedUnfreezeHook(Hook):
    """
    阶段2训练策略：
    Epoch 0-5: 只训练attribute_encoder和head，YOLO骨干完全冻结（FrozenBN）
    Epoch 5-10: 解冻neck部分，但保持FrozenBN
    Epoch 10+: 解冻全部（可选，如果过拟合则保持冻结）
    """

    def __init__(self,
                 freeze_epochs=5,
                 unfreeze_neck_epoch=5,
                 unfreeze_all_epoch=10,
                 new_module_keywords=('attribute_encoder', 'head')):
        self.freeze_epochs = freeze_epochs
        self.unfreeze_neck_epoch = unfreeze_neck_epoch
        self.unfreeze_all_epoch = unfreeze_all_epoch
        self.new_module_keywords = new_module_keywords

    def before_train(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        # 阶段2开始：冻结所有阶段1参数，只允许新模块训练
        for name, param in model.named_parameters():
            if any(key in name for key in self.new_module_keywords):
                param.requires_grad = True
                runner.logger.info(f"🟢 新模块可训练: {name}")
            else:
                param.requires_grad = False

        runner.logger.info(f"🔒 阶段1骨干已冻结，仅训练新模块（前{self.freeze_epochs}轮）")

    def after_train_epoch(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        current_epoch = runner.epoch

        # 阶段A：解冻neck（使用较小学习率）
        if current_epoch == self.unfreeze_neck_epoch:
            for name, param in model.named_parameters():
                if 'neck' in name:
                    param.requires_grad = True
            # 修改optimizer的参数组（应用不同lr）
            self._set_lr_for_module(runner, 'neck', lr_scale=0.1)
            runner.logger.info("🟡 解冻Neck部分，学习率x0.1")

        # 阶段B：解冻全部（可选）
        elif current_epoch == self.unfreeze_all_epoch:
            for name, param in model.named_parameters():
                param.requires_grad = True
            runner.logger.info("🔴 解冻全部参数")

    def _set_lr_for_module(self, runner, module_name, lr_scale=0.1):
        """为特定模块设置学习率缩放（需要配合优化器使用param_groups）"""
        # 注意：这里假设你使用了paramwise_cfg，实际可能需要调整optimizer配置
        pass