import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import yaml
import numpy as np 
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm
import argparse
import wandb
from datetime import datetime
from time import time 

from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch_pipelining.stage import PipelineStage
from torch_pipelining.schedules import ScheduleInterleaved1F1B, ScheduleGPipe
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.deepseek_v3.model import (
    Transformer, ModelArgs, tensor_parallelize_model, 
    pipeline_parallelize_model, materialize_meta_module
)
from models.deepseek_v3.utils import precompute_freqs_cis
from models.deepseek_v3.load_export_weights import (
    load_pipeline_weights, export_tp_pp_stage_to_hf_format_with_experts, merge_all_jsons_into_index
)
from train_utils import clear_cache_and_report, clip_grad_norm_, clip_grad_norm_no_tp_, Config, is_gradient_accumulation_boundary, take_optimizer_step
import math 

from data import create_mmlu_dataloader, create_dataloader

from mp_adamw import MPAdamW


def setup_model_and_optimizer(config: Config, device_mesh: DeviceMesh, device: torch.device):
    """Setup model, optimizer, and training components"""
    
    dp_mesh = device_mesh["dp"]
    pp_mesh = device_mesh["pp"]
    tp_mesh = device_mesh["tp"] # also ep_mesh is same as tp_mesh - both using 8 ranks within a node! 
    
    pp_rank = pp_mesh.get_local_rank()
    tp_rank = tp_mesh.get_local_rank()

    rank = dist.get_rank()
    
    # Load model config
    with open(config.model['config_path']) as f:
        model_args = ModelArgs(**json.load(f))
    model_args.tp_mesh = tp_mesh
    
    # Create model on meta device
    torch.set_default_dtype(getattr(torch, config.model['dtype']))
    with torch.device("meta"):
        model = Transformer(model_args)
    
    # Pipeline parallelize
    pipeline_parallelize_model(model, num_stages=config.distributed['pp_size'], stage_idx=pp_rank)
    
    # Materialize on device
    materialize_meta_module(model, device)
    model.freqs_cis = precompute_freqs_cis(model_args)
    model = model.to(device)

    # Convert to FP8 layers BEFORE loading weights (if enabled)
    use_fp8_training = config.model.get('fp8_training', {}).get('enabled', False)
    if use_fp8_training:
        from models.deepseek_v3.fp8_layers_triton import convert_linear_to_fp8, print_fp8_conversion_summary
        if rank == 0:
            print(f"Converting model to FP8 training...")
        model = convert_linear_to_fp8(model, fp8_enabled=True)
        
        # Print conversion summary
        print_fp8_conversion_summary(model, rank)
    else:
        if rank == 0:
            print(f"Using default {torch.get_default_dtype()} for training (not FP8)")
    
    # Load weights (now FP8 Linear layers will be properly identified)
    load_pipeline_weights(
        model=model,
        model_dir=config.model['model_dir'],
        pp_rank=pp_rank,
        pp_size=config.distributed['pp_size'],
        device=device,
        n_layers=model_args.n_layers,
        use_fp8_training=use_fp8_training
    )
    
    # Tensor parallelize - we can apply this for non-fp8 training layers later on!
    if not use_fp8_training:
        model = tensor_parallelize_model(
            model, 
            tp_mesh=tp_mesh, 
            no_tp_shard_on_experts=True
        )
        model.args.use_tp = True 
    else:
        model.args.use_tp = False
    
    # Enable gradient checkpointing if specified
    if config.training['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
    
    # Create optimizer
    if config.training['optimizer']['type'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.training['learning_rate']),
            betas=config.training['optimizer']['betas'],
            weight_decay=config.training['optimizer']['weight_decay'],
            eps=float(config.training['optimizer']['eps']),
            foreach=config.training['optimizer']['foreach']
        )
    elif config.training['optimizer']['type'].lower() == 'mpadamw':
        optimizer = MPAdamW(
            model.parameters(),
            lr=float(config.training['learning_rate']),
            betas=config.training['optimizer']['betas'],
            weight_decay=config.training['optimizer']['weight_decay'],
            eps=float(config.training['optimizer']['eps']),
            foreach=config.training['optimizer']['foreach']
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.training['optimizer']['type']}")
    
    return model, optimizer, pp_rank, pp_mesh, tp_mesh


def do_step(
    batch: Dict[str, torch.Tensor], 
    schedule, 
    pp_rank: int, 
    pp_size: int, 
    is_eval: bool = False, 
    eval_logits: bool = False # if False will return val loss; otherwise, return logits
):
    """Execute a single training step with proper synchronization"""
    
    attention_mask = batch['attention_mask']

    # Prepare batch for pipeline parallel
    pp_batch = {'attention_mask': attention_mask}
    pp_batch['batch_seqlen'] = attention_mask.shape[1]
    
    if pp_rank == 0:
        pp_batch['input_ids'] = batch['input_ids']
    
    if pp_rank == pp_size - 1:
        if not is_eval: # train
            losses = [] # the list will be updated inside the schedule!
            pp_batch['losses'] = losses
            pp_batch['target'] = batch['labels']
        else: # eval 
            if not eval_logits:
                pp_batch['labels'] = batch['labels']
            else:
                pp_batch['logits_to_keep'] = attention_mask.sum(dim=1) - 1
    
    # Execute pipeline schedule
    output = schedule.step(**pp_batch)
    
    # Return loss from last stage
    if pp_rank == pp_size - 1:
        if is_eval:
            return output
        else:
            return sum(losses) / len(losses) if losses else None
    
    return None


def validate(
    model,
    val_dataloader: DataLoader,
    val_schedule,
    pp_rank: int,
    pp_size: int,
    device: torch.device,
    config: Config,
    max_batches: Optional[int] = None
) -> float:
    """Run validation and return average loss"""
    model.eval()
    losses = []
    total_nan_counters = 0
    total_all_counters = 0
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Validation", disable=pp_rank != 0)):
            if max_batches and batch_idx >= max_batches:
                break
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Run validation step
            loss = do_step(batch, val_schedule, pp_rank, pp_size, is_eval=True)
            
            if pp_rank == pp_size - 1:
                if loss is not None:
                    # Handle NaN values like in the reference
                    nan_mask = torch.isnan(loss)
                    num_nans = nan_mask.sum().item()
                    total_nan_counters += num_nans
                    total_all_counters += loss.numel()
                    
                    # Filter out NaNs
                    valid_loss = loss[~nan_mask]
                    
                    # Append only non-NaN values
                    if valid_loss.numel() > 0:
                        losses.append(valid_loss.clone())

    # Log NaN statistics
    if pp_rank == pp_size - 1:
        print(f"[Validation] Total NaN entries encountered: {total_nan_counters} / {total_all_counters}")
        loss = torch.cat(losses).float().mean()#.detach().item()
        eval_perplexity = math.exp(loss)
        metrics = {"val/loss": float(loss), "val/perplexity": float(eval_perplexity)}
        dist.broadcast_object_list([metrics], src=dist.get_world_size() - 1)
    else:
        obj_list = [None]
        dist.broadcast_object_list(obj_list, src=dist.get_world_size() - 1)
        metrics = obj_list[0]

    model.train()
    return metrics


def validate_mmlu(
    model,
    mmlu_dataloader: DataLoader,
    val_schedule,
    pp_rank: int,
    pp_size: int,
    device: torch.device,
    tokenizer,
    config: Config,
) -> Dict[str, float]:
    """Evaluate model on MMLU benchmark"""
    model.eval()
    
    # Track predictions and answers by subject
    all_predictions = []
    all_answers = []
    all_subjects = []
    
    # Token IDs for answer choices
    answer_token_ids = torch.tensor([
        tokenizer.encode("A", add_special_tokens=False)[0],
        tokenizer.encode("B", add_special_tokens=False)[0],
        tokenizer.encode("C", add_special_tokens=False)[0],
        tokenizer.encode("D", add_special_tokens=False)[0],
    ], device=device)
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(mmlu_dataloader, desc="MMLU Evaluation", disable=pp_rank != 0)):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Run validation step
            attention_mask = batch['attention_mask']
            output = do_step(batch, val_schedule, pp_rank, pp_size, is_eval=True, eval_logits=True)
            
            # Process predictions on last stage
            if pp_rank == pp_size - 1 and output is not None:

                logits = output.squeeze()
                batch_size = logits.shape[0]
                
                predictions = []
                for i in range(batch_size):
                    # pred
                    last_token_logits = logits[i]
                    answer_logits = last_token_logits[answer_token_ids]

                    # Check answer_logits for NaN/Inf as well
                    if torch.isnan(answer_logits).any() or torch.isinf(answer_logits).any():
                        print(f"[MMLU] Skipping sample {i} due to NaN/Inf in answer_logits")
                        continue  # Skip this entire sample
                    
                    pred_idx = answer_logits.argmax().item()
                    
                    # Get correct answer
                    correct_token_id = batch['answer_token_id'][i].item()
                    correct_idx = (answer_token_ids == correct_token_id).nonzero(as_tuple=True)[0].item()
                    
                    # Only append if we reach here (no NaN/Inf issues)
                    #predictions.append(pred_idx)
                    all_predictions.append(pred_idx)
                    all_answers.append(correct_idx)
                    all_subjects.append(batch['subject'][i])
                    
                assert len(all_predictions) == len(all_answers) == len(all_subjects), \
                    f"Length mismatch: {len(all_predictions)} predictions, {len(all_answers)} answers, {len(all_subjects)} subjects"
                

    # Calculate accuracy by subject
    results = {}
    
    if pp_rank == pp_size - 1:
        # Convert to numpy for easier manipulation
        predictions = np.array(all_predictions)
        answers = np.array(all_answers)
        subjects = np.array(all_subjects)
        
        # Overall accuracy
        overall_correct = (predictions == answers).sum()
        overall_total = len(predictions)
        results['overall_accuracy'] = overall_correct / overall_total if overall_total > 0 else 0.0
        
        # Per-subject accuracy
        unique_subjects = np.unique(subjects)
        subject_scores = {}
        
        for subject in unique_subjects:
            mask = subjects == subject
            subject_preds = predictions[mask]
            subject_answers = answers[mask]
            
            correct = (subject_preds == subject_answers).sum()
            total = len(subject_preds)
            accuracy = correct / total if total > 0 else 0.0
            
            subject_scores[subject] = accuracy
        
        results['subject_scores'] = subject_scores
            
    # Broadcast results to all ranks
    if pp_rank == pp_size - 1:
        dist.broadcast_object_list([results], src=dist.get_world_size() - 1)
    else:
        obj_list = [None]
        dist.broadcast_object_list(obj_list, src=dist.get_world_size() - 1)
        results = obj_list[0]
    
    model.train()
    return results


def setup_wandb(config: Config, rank: int, pp_rank: int, tp_rank: int):
    """Initialize WandB for rank 0 only"""
    if rank == 0:
        # Create run name with timestamp
        run_name = f"deepseek_syncgrad_{config.distributed['pp_size']}pp_{config.training['optimizer']['type']}optm_{config.distributed['tp_ep_size']}tp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_name = config.logging['wandb'].get('run_name', run_name)

        # Initialize WandB
        wandb.init(
            project=config.logging['wandb'].get('project', 'deepseek-v3-training'),
            entity=config.logging['wandb'].get('entity'),
            name=run_name,
            config={
                'model': config.model,
                'training': config.training,
                'dataset': config.dataset,
                'distributed': config.distributed,
                'world_size': dist.get_world_size(),
                'pp_size': config.distributed['pp_size'],
                'tp_size': config.distributed['tp_ep_size'],
                'dp_size': config.distributed['dp_size'],
            },
            tags=[
                f"pp{config.distributed['pp_size']}",
                f"tp{config.distributed['tp_ep_size']}",
                "deepseek-v3"
            ]
        )
        return True
    return False


def log_metrics(
    rank: int,
    metrics: Dict[str, float],
    step: int,
    use_wandb: bool = True
):
    """Log metrics to console and WandB"""
    if rank == 0:
        # Log to console
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Step {step} | {metric_str}")
        
        # Log to WandB
        if use_wandb and wandb.run is not None:
            wandb.log(metrics, step=step)


def main():
    parser = argparse.ArgumentParser(description='Train DeepSeek with Pipeline Parallelism')
    parser.add_argument('--config', type=str, default='configs/train_example_formal.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Initialize distributed
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    pp_size = config.distributed['pp_size']
    tp_ep_size = config.distributed['tp_ep_size'] # same as ep (expert parallel) size!
    dp_size = config.distributed['dp_size']
    
    device_mesh = init_device_mesh(
        "cuda", 
        (dp_size, pp_size, tp_ep_size), 
        mesh_dim_names=("dp", "pp", "tp")
    )

    pp_rank = device_mesh["pp"].get_local_rank()
    tp_rank = device_mesh["tp"].get_local_rank()
    
    # Setup WandB (only on rank 0)
    use_wandb = config.logging['wandb'].get('enabled', False)
    if use_wandb:
        wandb_initialized = setup_wandb(config, rank, pp_rank, tp_rank)
    else:
        wandb_initialized = False

    # Initialize tokenizer with proper synchronization
    if rank == 0:
        # Only rank 0 downloads the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model['name'],
            token=os.environ.get("HF_TOKEN", ""),
            cache_dir=config.model['model_dir'], 
            local_files_only=False
        )
        print(f"Rank 0: Successfully downloaded tokenizer to {config.model['model_dir']}")

    # Synchronize all ranks
    dist.barrier()

    # Now all other ranks can safely load from cache
    if rank != 0:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model['name'],
            token=os.environ.get("HF_TOKEN", ""),
            cache_dir=config.model['model_dir'], 
            local_files_only=True  # This prevents downloading, only loads from cache
        )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Create dataloaders
    train_dataloader = create_dataloader(config, tokenizer, is_train=True)
    val_dataloader = create_dataloader(config, tokenizer, is_train=False)
    mmlu_dataloader = create_mmlu_dataloader(config, tokenizer) if config.dataset.get('use_mmlu_for_eval', False) else None
    
    # Setup model and optimizer
    model, optimizer, pp_rank, pp_mesh, tp_ep_mesh = setup_model_and_optimizer(config, device_mesh, device)

    # Create pipeline stages
    train_stage = PipelineStage(
        model,
        pp_rank,
        pp_size,
        device,
        group=pp_mesh.get_group(),
        return_outputs=False
    )

    val_stage = PipelineStage(
        model,
        pp_rank,
        pp_size,
        device,
        group=pp_mesh.get_group(),
        return_outputs=True
    )

    if config.dataset.get('use_mmlu_for_eval', False):
        val_stage_logits = PipelineStage(
            model,
            pp_rank,
            pp_size,
            device,
            group=pp_mesh.get_group(),
            return_outputs=True
    )

    # Create schedules
    if config.distributed['schedule_type'] == 'interleaved_1f1b':
        train_schedule = ScheduleInterleaved1F1B(
            [train_stage], 
            config.distributed['num_microbatches'], 
            loss_fn=model.loss_function, 
            scale_grads=config.training["gradient_accumulation_steps"] == 1
        )
    else:
        train_schedule = ScheduleGPipe(
            train_stage, 
            config.distributed['num_microbatches'], 
            loss_fn=model.loss_function
        )

    val_schedule = ScheduleGPipe(
        val_stage,
        config.distributed['num_microbatches']
    )

    if config.dataset.get('use_mmlu_for_eval', False):
        val_schedule_logits = ScheduleGPipe(
            val_stage_logits,
            config.distributed['num_microbatches']
        )

    # Training loop
    model.train()
    global_step = 0

    # Log initial configuration
    if rank == 0:
        print(f"Starting training with configuration:")
        print(f"  World size: {world_size}")
        print(f"  PP size: {pp_size}, TP(EP) size: {tp_ep_size}, DP size: {dp_size}")
        print(f"  Batch size: {config.training['total_batch_size']}")
        print(f"  Microbatches: {config.distributed['num_microbatches']}")
        print(f"  Learning rate: {config.training['learning_rate']}")
        print(f"  Training steps: {config.training['num_training_steps']}")
    # Training metrics tracking
    train_losses = []

    if config.training.get("eval_only", False):

        if rank == 0:
            print(f"\nDoing evaluation only! ")
        
        clear_cache_and_report("Running validation...")

        val_metrics = validate(
            model, 
            val_dataloader, 
            val_schedule, 
            pp_rank, 
            pp_size, 
            device, 
            config,
            max_batches=config.training.get('max_eval_batches', 50)  # Limit validation batches
        )

        if config.dataset.get('use_mmlu_for_eval', False):
            mmlu_results = validate_mmlu(
                model,
                mmlu_dataloader,
                val_schedule_logits,
                pp_rank,
                pp_size,
                device,
                tokenizer,
                config
            )
            
            # Log MMLU metrics
            mmlu_metrics = {
                'mmlu/overall_accuracy': mmlu_results['overall_accuracy'],
            }
            
            # Add category scores
            for category, score in mmlu_results.get('subject_scores', {}).items():
                mmlu_metrics[f'mmlu/{category}_accuracy'] = score

            val_metrics.update(mmlu_metrics)    

        log_metrics(rank, val_metrics, global_step, use_wandb=wandb_initialized)

        print(f"Eval only results: {val_metrics}")

        clear_cache_and_report("Cleanup validation...")

        # Cleanup
        dist.barrier()
        
        if rank == 0:
            print(f"\nEvaluation completed!")
            
            if wandb_initialized:
                wandb.finish()
        return 0 

    for epoch in range(config.training['num_epochs']):
        # Set the same seed across all ranks to ensure same shuffle order
        torch.manual_seed(42 + epoch)

        progress_bar = tqdm(train_dataloader, disable=rank != 0, desc=f"Epoch {epoch}")
        for step, batch in enumerate(progress_bar):
            # Move batch to device - no broadcasting needed!

            global_step += 1

            now = time() 

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}

            is_accum_boundary = is_gradient_accumulation_boundary(
               step=step, gas_amount=config.training["gradient_accumulation_steps"]
            )

            # Training step
            loss = do_step(batch, train_schedule, pp_rank, pp_size)
            if pp_rank == pp_size - 1 and loss is not None:
                train_losses.append(loss.item())
            
            grad_norm = take_optimizer_step(
                model = model,
                optimizer = optimizer,
                is_gas_boundary=is_accum_boundary,
                grad_clip_max_norm=config.training.get('gradient_clip_norm', 1.0),
                tp_mesh=tp_ep_mesh,
                pp_mesh=pp_mesh,
                config=config.training,
                fp8_training = config.model.get('fp8_training', {}).get('enabled', False)
            )
            
            # Logging
            if is_accum_boundary and global_step % config.logging['log_interval'] == 0:
                # Calculate average training loss
                if pp_rank == pp_size - 1 and train_losses:
                    avg_train_loss = sum(train_losses) / len(train_losses)
                    train_losses = []  # Reset
                else:
                    avg_train_loss = 0.0
                
                # Broadcast loss from last PP rank
                batch_time = time() - now

                avg_train_loss_tensor = torch.tensor(avg_train_loss, device=device)
                dist.broadcast(avg_train_loss_tensor, src=world_size - 1)  # Last global rank
                
                avg_train_loss = avg_train_loss_tensor.item()
                
                # Log metrics
                metrics = {
                    'train/loss': avg_train_loss,
                    'train/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                    'train/batch_time': batch_time,
                }
                
                log_metrics(rank, metrics, global_step, use_wandb=wandb_initialized)
            
            # Validation
            if is_accum_boundary and global_step % config.training['eval_steps'] == 0 and global_step > 0:
                if rank == 0:
                    print(f"\nRunning validation at step {global_step}...")
                
                clear_cache_and_report("Running validation...")

                val_metrics = validate(
                    model, 
                    val_dataloader, 
                    val_schedule, 
                    pp_rank, 
                    pp_size, 
                    device, 
                    config,
                    max_batches=config.training.get('max_eval_batches', 50)  # Limit validation batches
                )

                if config.dataset.get('use_mmlu_for_eval', False):
                    mmlu_results = validate_mmlu(
                        model,
                        mmlu_dataloader,
                        val_schedule_logits,
                        pp_rank,
                        pp_size,
                        device,
                        tokenizer,
                        config
                    )
                    
                    # Log MMLU metrics
                    mmlu_metrics = {
                        'mmlu/overall_accuracy': mmlu_results['overall_accuracy'],
                    }
                    
                    # Add category scores
                    for category, score in mmlu_results.get('subject_scores', {}).items():
                        mmlu_metrics[f'mmlu/{category}_accuracy'] = score

                    val_metrics.update(mmlu_metrics)    

                log_metrics(rank, val_metrics, global_step, use_wandb=wandb_initialized)

                clear_cache_and_report("Cleanup validation...")

            # Save checkpoint
            if is_accum_boundary and global_step % config.checkpoint['save_interval'] == 0 and global_step > 0:
                if rank == 0:
                    print(f"Saving checkpoint at step {global_step}")
                
                checkpoint_dir = export_tp_pp_stage_to_hf_format_with_experts(
                    model=model,
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                    tp_rank=device_mesh["tp"].get_local_rank(),
                    tp_size=config.distributed['tp_ep_size'],
                    tp_mesh=device_mesh["tp"],
                    temporal_dir=config.checkpoint.get("local_output_dir", "/mnt/localdisk/temp_local_checkpointing"),
                    using_expert_parallel=True,
                    output_dir=f"{config.checkpoint['save_dir']}/step_{global_step:06d}",
                    use_fp8_quantization=config.checkpoint.get('use_fp8_quantization', False), 
                    fp8_mp=config.model.get('fp8_training', {}).get('enabled', False)
                )

                dist.barrier()

                if rank == 0:
                    merge_all_jsons_into_index(
                        json_dir=checkpoint_dir,
                        temporal_dir=config.checkpoint.get("local_output_dir", "/mnt/localdisk/temp_local_checkpointing"),
                        json_pattern='partial_index*.json', 
                        output_dir=f"{config.checkpoint['save_dir']}/step_{global_step:06d}", 
                        delete_after_processing=True
                    )
                
                dist.barrier()
            
            # Update progress bar
            if rank == 0 and progress_bar is not None:
                progress_bar.set_postfix({
                    'loss': f"{avg_train_loss:.4f}" if 'avg_train_loss' in locals() else "N/A",
                    'grad_norm': f"{grad_norm:.4f}" if isinstance(grad_norm, (int, float)) or (isinstance(grad_norm, torch.Tensor) and grad_norm.numel() == 1) else "N/A"
                })
            
            if global_step >= config.training['num_training_steps']:
                break
        
        if global_step >= config.training['num_training_steps']:
            break
    
    if rank == 0:
        print(f"Saving checkpoint at step final")

    checkpoint_dir = export_tp_pp_stage_to_hf_format_with_experts(
        model=model,
        pp_rank=pp_rank,
        pp_size=pp_size,
        tp_rank=device_mesh["tp"].get_local_rank(),
        tp_size=config.distributed['tp_ep_size'],
        tp_mesh=device_mesh["tp"],
        temporal_dir=config.checkpoint.get("local_output_dir", "/mnt/localdisk/temp_local_checkpointing"),
        using_expert_parallel=True,
        output_dir=f"{config.checkpoint['save_dir']}/step_final", 
        use_fp8_quantization=config.checkpoint.get('use_fp8_quantization', False), 
        fp8_mp=config.model.get('fp8_training', {}).get('enabled', False)
    )
    
    dist.barrier()

    if rank == 0:
        merge_all_jsons_into_index(
            json_dir=checkpoint_dir,
            temporal_dir=config.checkpoint.get("local_output_dir", "/mnt/localdisk/temp_local_checkpointing"),
            json_pattern='partial_index*.json', 
            output_dir=f"{config.checkpoint['save_dir']}/step_final",
            delete_after_processing=True
        )
                
    # Cleanup
    dist.barrier()
    
    if rank == 0:
        print(f"\nTraining completed!")
        
        if wandb_initialized:
            wandb.finish()
    
    clear_cache_and_report("Training completed")



if __name__ == "__main__":
    main()
