# mlx_ppo_trainer_enhanced.py

import json
import logging
import math
import random
import os
import sys
import time
import gc
import re
import signal  # For graceful shutdown
import contextlib  # For conditional progress bar context
from typing import Tuple, Dict, Any, List, Optional, Generator, Set, Union
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

# Import MLX components
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map

# Numpy for shuffling indices and other utilities
import numpy as np

# Import rich for progress bars and logging
# Define console and RICH_AVAILABLE robustly before other imports/classes might use them
console: Union["Console", "DummyConsole"]  # Forward reference for type hint
RICH_AVAILABLE = False


# Define dummy Console first
class DummyConsole:
    def print(self, *args, **kwargs):
        print(*args)


console = DummyConsole()  # Default to dummy

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TaskID,
    )
    from rich.logging import RichHandler
    from rich.console import Console  # Import real Console
    from rich.text import Text

    # Try to instantiate the real console
    console = Console(force_terminal=True)  # Overwrite dummy if successful
    RICH_AVAILABLE = True
    # Logger not configured yet, print basic message
    print("[DEBUG] Rich library found and initialized.")
except ImportError:
    print(
        "[WARNING] Rich library not found. Progress bars and rich logging will be basic."
    )
    # console remains DummyConsole
except Exception as e:
    print(
        f"[ERROR] Error initializing Rich console: {e}. Falling back to basic console."
    )
    console = DummyConsole()  # Fallback just in case instantiation fails


# Command-line interface
import click

# MLX LM specific imports
try:
    from mlx_lm.utils import (
        load as load_mlx_model_and_tokenizer,
        save_weights,
        get_model_path,
        load_config,
    )
    from mlx_lm.tuner.utils import linear_to_lora_layers  # For LoRA support

    # Assuming Llama model structure is needed for type hints/accessing args
    from mlx_lm.models.llama import Model as LlamaModel

    MLX_LM_AVAILABLE = True
except ImportError as e:
    print(
        f"Error importing from mlx_lm: {e}. Ensure mlx_lm is installed.",
        file=sys.stderr,
    )
    print("You can typically install it via: pip install mlx-lm", file=sys.stderr)
    MLX_LM_AVAILABLE = False
    sys.exit(1)

# Local reward function import
try:
    # Use the version WITHOUT dataset scaling
    from reward import SimpleRewardFunction, RewardConfig

    REWARD_FUNC_AVAILABLE = True
except ImportError as e:
    print(
        f"Error importing reward function: {e}. Ensure reward.py is accessible.",
        file=sys.stderr,
    )
    REWARD_FUNC_AVAILABLE = False
    sys.exit(1)

# === Global Variables ===
# Logger instance configured in cli_main after basicConfig is called
logger = logging.getLogger(__name__)
shutdown_requested = False  # Flag for graceful shutdown


# === Signal Handler ===
def handle_signal(signum, frame):
    """Sets the shutdown flag when SIGINT (CTRL+C) is received."""
    global shutdown_requested
    if not shutdown_requested:
        msg = "[bold yellow]Shutdown requested (SIGINT received). Finishing current update/rollout and saving...[/]"
        console.print(msg)  # Use console for immediate feedback
        shutdown_requested = True
    else:
        msg = "[bold red]Multiple shutdown signals received. Forcing exit.[/]"
        console.print(msg)
        sys.exit(1)  # Force exit on second signal


# Register signal handler
signal.signal(signal.SIGINT, handle_signal)


# === Text Processing & Validation Utilities ===
def _extract_block_content(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Safely extracts content within the first occurrence of start_tag...end_tag block."""
    if not text or not isinstance(text, str):
        return None
    try:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    except Exception as e:
        logging.getLogger(__name__).debug(
            f"Regex error extracting {start_tag}...{end_tag}: {e}"
        )
        return None


def validate_text_format(
    generated_text: str, config: RewardConfig
) -> Tuple[bool, List[str]]:
    """Validates format <thinking>...</thinking>\n<answer>...</answer> using tags from RewardConfig."""
    # (Implementation remains the same as previous version)
    warnings = []
    if not isinstance(generated_text, str) or not generated_text.strip():
        return False, ["Input text is empty or not a string."]
    tags = config.special_tokens
    think_start, think_end = tags["think_start"], tags["think_end"]
    answer_start, answer_end = tags["answer_start"], tags["answer_end"]
    required_tags = [think_start, think_end, answer_start, answer_end]
    tag_positions = {}
    missing, multiple = [], []
    for tag in required_tags:
        try:
            indices = [
                m.start()
                for m in re.finditer(re.escape(tag), generated_text, re.IGNORECASE)
            ]
            count = len(indices)
            if count == 0:
                missing.append(f"'{tag}'")
                tag_positions[tag] = -1
            elif count > 1:
                multiple.append(f"'{tag}'")
                tag_positions[tag] = indices[0]
            else:
                tag_positions[tag] = indices[0]
        except Exception as e:
            warnings.append(f"Regex error finding {tag}: {e}")
            return False, warnings
    if missing:
        warnings.append(f"Missing tags: {', '.join(missing)}")
    if multiple:
        warnings.append(f"Multiple tags: {', '.join(multiple)}")
    if missing or multiple:
        return False, warnings
    p0, p1, p2, p3 = (
        tag_positions[think_start],
        tag_positions[think_end],
        tag_positions[answer_start],
        tag_positions[answer_end],
    )
    if not (p0 < p1 < p2 < p3):
        warnings.append(f"Tag order incorrect")
        return False, warnings
    try:
        separator = generated_text[p1 + len(think_end) : p2]
        if separator.strip():
            warnings.append(f"Non-whitespace separator")
            return False, warnings
    except IndexError:
        warnings.append("Separator check error")
        return False, warnings
    return True, []


# === Custom Iterable Dataset ===
class JsonlPromptDataset:
    """Reads and yields prompts from a JSONL file, handling potential errors."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.logger = logging.getLogger(__name__)  # Get logger instance
        self.prompts: List[str] = []
        self.raw_data: List[Dict] = []  # Store full data dicts

        if not self.file_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
        self._load_data()

    def _load_data(self):
        self.logger.info(f"Loading dataset: {self.file_path}")
        skipped = 0
        loaded_count = 0
        # Use Rich Progress if available for loading
        progress_context = (
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
                transient=True,
            )
            if RICH_AVAILABLE
            else None
        )
        try:
            # Estimate length for progress bar
            try:
                total_lines = sum(
                    1 for _ in open(self.file_path, "r", encoding="utf-8")
                )
            except Exception as e:
                self.logger.warning(f"Could not estimate dataset length: {e}")
                total_lines = None

            with open(
                self.file_path, "r", encoding="utf-8"
            ) as f, progress_context if progress_context else contextlib.nullcontext() as progress:
                task_id = None
                iterable_f = f  # Use the file handle directly if no progress bar

                if progress:
                    if total_lines:
                        task_id = progress.add_task(
                            f"Loading {self.file_path.name}", total=total_lines
                        )
                    else:
                        task_id = progress.add_task(
                            f"Loading {self.file_path.name}"
                        )  # Indeterminate
                    iterable_f = f  # Still iterate over the original file handle

                for i, line in enumerate(iterable_f):
                    line_num = i + 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        prompt = data.get("prompt")
                        if isinstance(prompt, str) and prompt:
                            # Also require 'completion' for reference in environment step
                            if (
                                isinstance(data.get("completion"), str)
                                and data["completion"]
                            ):
                                self.prompts.append(prompt)
                                self.raw_data.append(data)  # Store original data
                                loaded_count += 1
                            else:
                                self.logger.debug(
                                    f"Line {line_num}: Skipping, missing/invalid completion."
                                )
                                skipped += 1
                        else:
                            self.logger.debug(
                                f"Line {line_num}: Skipping, missing/invalid prompt."
                            )
                            skipped += 1
                    except json.JSONDecodeError:
                        self.logger.warning(f"Line {line_num}: Skipping invalid JSON.")
                        skipped += 1
                    except Exception as e:
                        self.logger.warning(f"Line {line_num}: Error processing: {e}")
                        skipped += 1
                    finally:
                        if progress and task_id is not None:
                            progress.update(task_id, advance=1)

                if progress and task_id is not None:
                    progress.update(
                        task_id,
                        completed=loaded_count + skipped,
                        total=loaded_count + skipped,
                    )  # Final update

            self.logger.info(
                f"Loaded {loaded_count} valid samples (skipped {skipped}) from {self.file_path}"
            )
            if not self.prompts:
                self.logger.error("No valid samples (prompt+completion) loaded!")

        except FileNotFoundError:
            self.logger.error(f"Dataset file not found during load: {self.file_path}")
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to load prompts from {self.file_path}: {e}", exc_info=True
            )

    def __len__(self):
        return len(self.prompts)  # Length based on valid loaded prompts

    def get_raw_sample(self, idx):
        return self.raw_data[idx % len(self.raw_data)]  # Allow wrap around

    def __getitem__(self, idx):
        return self.prompts[idx % len(self.prompts)]  # Allow wrap around


# === MLX Environment ===
class LLMEnv:
    """Environment for LLM PPO training using MLX."""

    def __init__(
        self,
        dataset: JsonlPromptDataset,
        tokenizer: Any,
        reward_function: SimpleRewardFunction,
        max_prompt_len: int = 512,
        env_id: str = "train",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.reward_fn = reward_function
        self.max_prompt_len = max_prompt_len
        self.env_id = env_id
        self.reward_config = reward_function.config
        self.logger = logging.getLogger(__name__)
        self.data_indices = list(
            range(len(self.dataset))
        )  # Use length of loaded prompts/data
        if not self.data_indices:
            raise ValueError(f"Env '{self.env_id}' dataset empty.")
        random.shuffle(self.data_indices)
        self.data_iterator_idx = 0
        self.current_prompt_text: Optional[str] = None
        self.current_reference_completion: Optional[str] = None
        self.current_sample_data: Optional[Dict] = None
        self.logger.info(
            f"LLMEnv '{self.env_id}' initialized with {len(self.data_indices)} samples."
        )

    def _get_next_sample_data(self) -> Optional[Dict]:
        """Gets the next full sample data dictionary using the shuffled index iterator."""
        if self.data_iterator_idx >= len(self.data_indices):
            self.logger.debug(
                f"Dataset '{self.env_id}' iterator exhausted. Resetting & shuffling."
            )
            random.shuffle(self.data_indices)
            self.data_iterator_idx = 0
            if not self.data_indices:
                return None
        current_internal_idx = self.data_indices[self.data_iterator_idx]
        self.data_iterator_idx += 1
        try:
            # Access raw data using the internal index
            raw_sample = self.dataset.get_raw_sample(current_internal_idx)
            # Basic check again, although dataset loading should have filtered
            if (
                isinstance(raw_sample.get("prompt"), str)
                and raw_sample["prompt"]
                and isinstance(raw_sample.get("completion"), str)
                and raw_sample["completion"]
            ):
                return raw_sample
            else:
                self.logger.warning(
                    f"Invalid data content retrieved at index {current_internal_idx}."
                )
                return self._get_next_sample_data()
        except IndexError:
            self.logger.error(
                f"Index {current_internal_idx} out of bounds for raw_data."
            )  # Should not happen
        except Exception as e:
            self.logger.error(
                f"Error fetching sample at index {current_internal_idx}: {e}"
            )
        return self._get_next_sample_data()  # Retry

    def reset(self) -> Optional[mx.array]:
        """Resets environment, returns tokenized prompt (state)."""
        attempts = 0
        max_attempts = max(10, len(self.dataset) // 10)
        while attempts < max_attempts:
            attempts += 1
            self.current_sample_data = self._get_next_sample_data()
            if self.current_sample_data is None:
                continue
            self.current_prompt_text = self.current_sample_data.get("prompt")
            self.current_reference_completion = self.current_sample_data.get(
                "completion"
            )
            if not self.current_prompt_text or not self.current_reference_completion:
                continue
            try:
                tokenized_ids = self.tokenizer.encode(self.current_prompt_text)
                if not tokenized_ids:
                    continue
                if len(tokenized_ids) > self.max_prompt_len:
                    tokenized_ids = tokenized_ids[: self.max_prompt_len]
                prompt_mx = mx.array(tokenized_ids, dtype=mx.int32).reshape(1, -1)
                mx.eval(prompt_mx)
                # self.reward_fn.reset(); # No reset needed
                return prompt_mx
            except Exception as e:
                self.logger.warning(f"Tokenization/Reset fail: {e}")
            self.current_prompt_text = None
            self.current_reference_completion = None
            self.current_sample_data = None
        self.logger.error(f"Failed to reset env after {attempts} attempts.")
        return None

    def step(self, generated_text: str) -> Tuple[float, Dict[str, Any]]:
        """Calculates reward using the stored reference completion."""
        if self.current_reference_completion is None:
            min_reward = getattr(self.reward_config, "min_reward_clip", 0.0)
            return min_reward, {"error": "Environment not reset or reset failed"}
        try:
            is_valid, fmt_warnings = validate_text_format(
                generated_text, self.reward_config
            )
            reward_metrics = {}
            if not is_valid:
                self.logger.debug(
                    f"Generated text failed format validation: {fmt_warnings}"
                )
                reward_val = getattr(self.reward_config, "min_reward_clip", -1.0)
                reward_metrics["format_error"] = fmt_warnings
                reward_metrics["format_score"] = 0.0
            else:
                reward_val, reward_metrics = self.reward_fn.calculate_reward(
                    generated_text=str(generated_text),
                    reference_text=str(self.current_reference_completion),
                )
            reward_metrics["generation_valid_format"] = 1.0 if is_valid else 0.0
            # Add per-sample bounds from dataset to info if they exist
            if self.current_sample_data:
                if "min_reward" in self.current_sample_data:
                    reward_metrics["sample_min_reward"] = self.current_sample_data[
                        "min_reward"
                    ]
                if "max_reward" in self.current_sample_data:
                    reward_metrics["sample_max_reward"] = self.current_sample_data[
                        "max_reward"
                    ]
            return float(reward_val), reward_metrics
        except Exception as e:
            self.logger.error(f"Reward calculation error: {e}", exc_info=True)
            min_reward = getattr(self.reward_config, "min_reward_clip", 0.0)
            return min_reward, {"reward_error": str(e), "generation_valid_format": 0.0}

    def __len__(self):
        return len(self.dataset)


# === MLX Critic Network ===
class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_factor: int = 2):
        super().__init__()
        hidden_dim = input_dim * hidden_dim_factor
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.logger = logging.getLogger(__name__)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Forward pass for the critic network.
        Args:
            hidden_states: The input hidden states, expected shape (batch, seq_len, hidden_dim)
                           or (batch, hidden_dim) if already pooled.
        Returns:
            The predicted value, shape (batch, 1).
        """
        try:
            # Pool if sequence dimension exists
            if hidden_states.ndim == 3:
                # Use mean pooling over the sequence length dimension
                pooled_state = mx.mean(hidden_states, axis=1)
            elif hidden_states.ndim == 2:
                # Assume already pooled or single state representation
                pooled_state = hidden_states
            else:
                raise ValueError(
                    f"Unsupported input ndim for CriticNetwork: {hidden_states.ndim}"
                )

            # Pass through linear layers
            x = nn.gelu(self.fc1(pooled_state))
            value = self.fc2(x)
            return value
        except Exception as e:
            self.logger.error(f"Critic forward pass failed: {e}", exc_info=True)
            # Return zeros with the expected batch dimension
            batch_dim = hidden_states.shape[0]
            return mx.zeros((batch_dim, 1))


# === MLX Rollout Buffer ===
@dataclass
class RolloutBuffer:
    # (Implementation remains the same as previous version)
    prompts: List[mx.array] = field(default_factory=list)
    generations: List[mx.array] = field(default_factory=list)
    actions_text: List[str] = field(default_factory=list)
    log_probs: List[mx.array] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[mx.array] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    advantages: Optional[mx.array] = None
    returns: Optional[mx.array] = None
    logger = logging.getLogger(__name__)

    def add(
        self,
        prompt: mx.array,
        gen_ids: mx.array,
        action_text: str,
        log_prob: mx.array,
        reward: float,
        done: bool,
        value: mx.array,
    ):
        if not all(isinstance(x, mx.array) for x in [prompt, gen_ids, log_prob, value]):
            self.logger.error("Invalid type to RolloutBuffer.add")
            return
        self.prompts.append(prompt.squeeze(0))
        self.generations.append(gen_ids.squeeze(0))
        self.actions_text.append(action_text)
        self.log_probs.append(log_prob.squeeze())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.squeeze())

    def _pad_sequences(
        self, sequences: List[mx.array], pad_value: int
    ) -> Tuple[mx.array, mx.array]:
        if not sequences:
            return mx.array([[]], dtype=mx.int32), mx.array([[]], dtype=mx.bool_)
        lengths = [s.size for s in sequences]
        max_len = max(lengths) if lengths else 0
        padded_seqs, masks = [], []
        for i, s in enumerate(sequences):
            pad_len = max_len - lengths[i]
            padding = mx.full((pad_len,), pad_value, dtype=s.dtype)
            mask = mx.concatenate(
                [mx.ones(lengths[i], dtype=mx.bool_), mx.zeros(pad_len, dtype=mx.bool_)]
            )
            padded_seqs.append(mx.concatenate([s, padding]))
            masks.append(mask)
        return mx.stack(padded_seqs, axis=0), mx.stack(masks, axis=0)

    def compute_advantages_and_returns(
        self, last_value: mx.array, gamma: float, gae_lambda: float
    ):
        num_steps = len(self.rewards)
        if num_steps == 0:
            self.advantages = mx.array([])
            self.returns = mx.array([])
            return
        # Ensure last_value is a scalar or has a compatible shape
        last_value_scalar = (
            last_value.item() if last_value.size == 1 else 0.0
        )  # Default if shape mismatch

        values_np = np.array(
            [v.item() for v in self.values] + [last_value_scalar], dtype=np.float32
        )
        rewards_np = np.array(self.rewards, dtype=np.float32)
        advantages_np = np.zeros_like(rewards_np)
        last_gae_lam = 0.0
        for t in reversed(range(num_steps)):
            # Assuming dones[t] is always True in this setup (end of episode per step)
            # If not, the GAE calculation needs adjustment: delta + gamma * gae_lambda * (1.0 - dones[t+1]) * last_gae_lam
            # For LLM generation, each step is effectively the end of that "action sequence", so done=True is reasonable.
            delta = rewards_np[t] + gamma * values_np[t + 1] - values_np[t]
            last_gae_lam = (
                delta + gamma * gae_lambda * last_gae_lam
            )  # Simplified assuming done=True
            advantages_np[t] = last_gae_lam

        self.advantages = mx.array(advantages_np)
        self.returns = self.advantages + mx.array(
            values_np[:-1]
        )  # Returns = Advantages + Values
        mx.eval(self.advantages, self.returns)
        self.logger.debug(
            f"Computed GAE: Adv shape {self.advantages.shape}, Ret shape {self.returns.shape}"
        )

    def get_batch_generator(
        self, batch_size: int, tokenizer: Any
    ) -> Generator[Optional[Dict[str, mx.array]], None, None]:
        num_samples = len(self.prompts)
        if num_samples == 0 or self.advantages is None or self.returns is None:
            self.logger.warning(
                "Cannot generate batches, buffer empty or advantages not computed."
            )
            yield None
            return
        indices = np.random.permutation(num_samples)
        pad_token_id = getattr(
            tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)
        )
        if pad_token_id is None:
            self.logger.error("Tokenizer needs pad_token_id or eos_token_id.")
            yield None
            return
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            if len(batch_indices) == 0:
                continue
            try:
                batch_prompts_list = [self.prompts[i] for i in batch_indices]
                batch_gens_list = [self.generations[i] for i in batch_indices]
                prompts_padded, prompt_mask = self._pad_sequences(
                    batch_prompts_list, pad_token_id
                )
                generations_padded, gen_mask = self._pad_sequences(
                    batch_gens_list, pad_token_id
                )
                batch_log_probs_old = mx.stack(
                    [self.log_probs[i] for i in batch_indices]
                )
                batch_advantages = self.advantages[batch_indices]
                batch_returns = self.returns[batch_indices]
                batch_values_old = mx.stack([self.values[i] for i in batch_indices])
                yield {
                    "prompts_padded": prompts_padded,
                    "prompt_mask": prompt_mask,
                    "generations_padded": generations_padded,
                    "generation_mask": gen_mask,
                    "log_probs_old": batch_log_probs_old,
                    "advantages": batch_advantages,
                    "returns": batch_returns,
                    "values_old": batch_values_old,
                }
            except Exception as e:
                self.logger.error(
                    f"Error creating batch starting at index {start_idx}: {e}",
                    exc_info=True,
                )
                yield None

    def clear(self):
        self.prompts.clear()
        self.generations.clear()
        self.actions_text.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None
        gc.collect()

    def __len__(self) -> int:
        return len(self.rewards)


# === Metrics Logger ===
class MetricsLogger:
    # (Implementation remains the same as previous version)
    def __init__(self, log_file: Union[str, Path]):
        self.log_path = Path(log_file)
        self.logger = logger
        is_new_file = not self.log_path.exists()
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Metrics will be logged to: {self.log_path}")
        except OSError as e:
            self.logger.error(f"Failed prepare metrics log file {self.log_path}: {e}")
            self.log_path = None

    def log(self, data: Dict[str, Any]):
        if self.log_path is None:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                serializable_data = {}
                for k, v in data.items():
                    if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                        serializable_data[k] = v
                    elif isinstance(v, (mx.array, np.number, np.bool_)):
                        serializable_data[k] = v.item()
                    elif isinstance(v, np.ndarray):
                        serializable_data[k] = v.tolist()
                    else:
                        serializable_data[k] = str(v)
                f.write(json.dumps(serializable_data) + "\n")
        except Exception as e:
            self.logger.error(f"Failed write metrics: {e} | Data: {data}")


# === PPO Agent (Functional Updates, Enhanced & Fixed) ===
class PPOAgent:
    def __init__(
        self,
        model: nn.Module,
        critic: CriticNetwork,
        tokenizer: Any,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        value_loss_coef: float,
        entropy_coef: float,
        max_gen_len: int,
        ppo_epochs: int,  # Added missing PPO epochs
        ppo_batch_size: int,  # Added missing PPO batch size
        grad_clip_norm: Optional[float] = 1.0,
    ):
        self.actor = model
        self.critic = critic
        self.tokenizer = tokenizer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_gen_len = max_gen_len
        self.ppo_epochs = ppo_epochs  # Store PPO epochs
        self.ppo_batch_size = ppo_batch_size  # Store PPO batch size
        self.grad_clip_norm = grad_clip_norm
        self.logger = logger
        try:
            # Attempt to get hidden_size from model.args or model.config
            if hasattr(model, "args") and hasattr(model.args, "hidden_size"):
                self.hidden_size = model.args.hidden_size
            elif (
                hasattr(model, "config")
                and isinstance(model.config, dict)
                and "hidden_size" in model.config
            ):
                self.hidden_size = model.config["hidden_size"]
            elif (
                hasattr(model, "model")
                and hasattr(model.model, "config")
                and isinstance(model.model.config, dict)
                and "hidden_size" in model.model.config
            ):
                self.hidden_size = model.model.config[
                    "hidden_size"
                ]  # Handle nested model attribute
            else:
                # Fallback: try inspecting a parameter shape (less reliable)
                params = model.parameters()
                found_dim = None
                for p in tree_flatten(params)[0]:
                    if isinstance(p, mx.array) and p.ndim > 0:
                        found_dim = p.shape[-1]  # Assume last dim is hidden size
                        break
                if found_dim:
                    self.hidden_size = found_dim
                    self.logger.warning(
                        f"Guessed hidden_size={self.hidden_size} from parameter shapes."
                    )
                else:
                    raise ValueError("Cannot determine actor hidden size.")
        except AttributeError as e:
            raise ValueError(f"Cannot determine actor hidden size: {e}")

        # Use model.parameters() which includes trainable and non-trainable
        # Ensure we capture the correct parameters based on LoRA
        self.actor_params = tree_map(lambda p: p, self.actor.parameters())
        self.critic_params = tree_map(lambda p: p, self.critic.parameters())

        # Initialize optimizer state correctly based on trainable parameters
        actor_trainable_params = self.actor.trainable_parameters()
        self.actor_opt_state = (
            self.actor_optimizer.state
        )  # tree_map(lambda _: None, actor_trainable_params)
        self.critic_opt_state = (
            self.critic_optimizer.state
        )  # tree_map(lambda _: None, self.critic_params)

        mx.eval(
            self.actor_params,
            self.critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
        )
        # Corrected param count using index [0] for leaves
        param_count = sum(
            p.size
            for p in tree_flatten(self.actor_params)[0]
            if isinstance(p, mx.array)
        )  # Robust count
        trainable_param_count = sum(
            p.size
            for p in tree_flatten(actor_trainable_params)[0]
            if isinstance(p, mx.array)
        )
        self.logger.info(
            f"Agent Initialized. Actor Total Params: {param_count / 1e6:.2f}M, Trainable: {trainable_param_count / 1e6:.3f}M"
        )

    def _get_state_embedding(
        self, actor_model: nn.Module, prompt_ids: mx.array
    ) -> mx.array:
        """Gets the embedding from the actor model without passing state."""
        try:
            # Call the model directly without the 'state' argument
            # The model will use its internal parameters (captured by the outer function)
            if hasattr(actor_model, "model") and isinstance(
                actor_model.model, nn.Module
            ):
                # Handle cases where the main logic is in a sub-module like model.model
                hidden_states = actor_model.model(prompt_ids)
            else:
                hidden_states = actor_model(prompt_ids)

            # Handle potential tuple output (e.g., (logits, cache))
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[
                    0
                ]  # Assume first element is hidden states/logits

            # Pool the hidden states (e.g., mean pooling over sequence length)
            if hidden_states.ndim == 3 and hidden_states.shape[1] > 0:
                # Mean pooling over the sequence length (axis 1)
                return mx.mean(hidden_states, axis=1)
            elif hidden_states.ndim == 2:
                # If already pooled or shape is (batch, hidden_dim)
                return hidden_states
            else:
                raise ValueError(
                    f"Unexpected state embedding shape after model call: {hidden_states.shape}"
                )
        except Exception as e:
            self.logger.error(f"Failed state embedding: {e}", exc_info=True)
            # Return zeros with the expected batch dimension and hidden size
            batch_dim = prompt_ids.shape[0]
            return mx.zeros((batch_dim, self.hidden_size))

    def get_value(self, prompt_ids: mx.array) -> mx.array:
        """Gets the value prediction from the critic for given prompt_ids."""
        # We don't need gradients w.r.t value calculation during rollouts
        self.actor.eval()
        self.critic.eval()
        value = mx.zeros((prompt_ids.shape[0],))  # Default value
        try:
            if prompt_ids.ndim == 1:
                prompt_ids = prompt_ids[None, :]  # Add batch dimension if missing

            # Get embedding using the current actor parameters (no gradients needed here)
            # Use the actual actor instance which holds the current parameters
            state_embedding = self._get_state_embedding(self.actor, prompt_ids)
            # state_embedding = mx.stop_gradient(state_embedding) # Stop gradient flow from actor

            # Get value using the current critic parameters (no gradients needed here)
            # Use the actual critic instance
            value_output = self.critic(state_embedding)  # Call critic without state=...
            value = value_output.squeeze(
                -1
            )  # Remove the last dimension (shape becomes (batch,))

            value = mx.stop_gradient(
                value
            )  # Ensure no gradients flow back from value calculation
            mx.eval(value)  # Evaluate the result

        except Exception as e:
            self.logger.error(f"Error getting value: {e}", exc_info=True)
            value = mx.zeros((prompt_ids.shape[0],))  # Return zeros on error

        # Restore train mode (though not strictly necessary if only eval used here)
        # self.actor.train()
        # self.critic.train()
        return value

    def _sample_token(self, logits: mx.array, temp: float) -> Tuple[mx.array, mx.array]:
        """Samples a token from logits and calculates its log probability."""
        if temp == 0:
            # Greedy sampling
            token = mx.argmax(logits, axis=-1)
        else:
            # Temperature-based sampling
            token = mx.random.categorical(logits * (1 / temp))

        # Calculate log probability of the sampled token
        log_probs_all = nn.log_softmax(logits, axis=-1)
        log_prob_sampled = mx.take_along_axis(
            log_probs_all,
            token[:, None],
            axis=-1,  # Ensure token has a dim for take_along_axis
        ).squeeze(
            -1
        )  # Remove the added dimension

        return token, log_prob_sampled

    def generate_action(
        self, prompt_ids: mx.array, temp: float = 0.7
    ) -> Tuple[str, mx.array, mx.array, mx.array, mx.array]:
        """Generates an action (sequence of tokens) from the actor."""
        self.actor.eval()  # Set actor to evaluation mode
        self.critic.eval()  # Set critic to evaluation mode (for get_value)

        generated_tokens_list = []
        log_probs_list = []
        current_ids = prompt_ids
        batch_size = prompt_ids.shape[0]

        if batch_size != 1:
            self.logger.warning(
                f"generate_action currently expects batch size 1, got {batch_size}. Using first element."
            )
            current_ids = prompt_ids[0:1]  # Adjust to batch size 1 if needed

        # Get the initial value estimate based on the prompt
        # This uses the current parameters held by the agent's critic instance
        initial_value = self.get_value(current_ids)

        # Generation loop (no gradients needed for the generation process itself)
        kv_cache = None  # KV caching can be added here for efficiency if needed
        try:
            current_gen_ids = current_ids
            for i in range(self.max_gen_len):
                # Call the actor model directly (uses internal parameters)
                # Add cache handling if implemented in the model
                if kv_cache is not None:
                    outputs = self.actor(current_gen_ids[:, -1:], cache=kv_cache)
                else:
                    outputs = self.actor(current_gen_ids)

                # Extract logits and potentially update cache
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    kv_cache = outputs[1]  # Assumes (logits, cache) output
                else:
                    logits = outputs  # Assumes only logits output

                next_token_logits = logits[:, -1, :]  # Get logits for the last token

                # Sample the next token and get its log probability
                token, log_prob = self._sample_token(next_token_logits, temp)

                # Store generated token and its log probability
                generated_tokens_list.append(token)
                log_probs_list.append(log_prob)

                # Append the new token for the next iteration
                current_gen_ids = mx.concatenate(
                    [current_gen_ids, token[:, None]], axis=1
                )

                # Check for EOS token
                if self.tokenizer.eos_token_id is not None and mx.all(
                    token == self.tokenizer.eos_token_id
                ):
                    break

            if not generated_tokens_list:
                # Handle case where no tokens were generated (e.g., max_gen_len=0)
                generation_ids = mx.array([[]], dtype=mx.int32)
                sequence_log_prob = mx.array(0.0)
            else:
                # Concatenate generated tokens and calculate total sequence log probability
                generation_ids = mx.concatenate(generated_tokens_list, axis=0)[
                    None, :
                ]  # Add batch dim back
                sequence_log_prob = mx.sum(mx.stack(log_probs_list))

            # Create the full sequence (prompt + generation)
            full_sequence_ids = mx.concatenate([current_ids, generation_ids], axis=1)

            # Decode the generated part into text
            generated_text = self.tokenizer.decode(generation_ids[0].tolist())

            # Ensure all MLX arrays are evaluated
            mx.eval(sequence_log_prob, initial_value, generation_ids, full_sequence_ids)

        except Exception as e:
            self.logger.error(f"Error during generate_action: {e}", exc_info=True)
            # Fallback values in case of error
            generated_text = "[Generation Error]"
            sequence_log_prob = mx.array(0.0)
            initial_value = mx.array(0.0)  # Ensure consistent shape
            generation_ids = mx.array([[]], dtype=mx.int32)
            full_sequence_ids = current_ids  # Fallback to prompt

        # Restore train mode (important if updates happen later)
        self.actor.train()
        self.critic.train()

        return (
            generated_text,
            sequence_log_prob,
            initial_value,
            generation_ids,
            full_sequence_ids,
        )

    def evaluate_actions(
        self,
        actor_model: nn.Module,  # Pass the model instance
        critic_model: nn.Module,  # Pass the model instance
        prompts_padded: mx.array,
        generations_padded: mx.array,
        generation_mask: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Evaluate actions (generations) using specific model instances.
        This is typically called within the loss function computation.
        """
        # Ensure models are in train mode for gradient calculation if needed
        # actor_model.train()
        # critic_model.train()

        # Concatenate prompt and generation for full sequence input
        full_sequence = mx.concatenate([prompts_padded, generations_padded], axis=1)
        prompt_len = prompts_padded.shape[1]
        gen_len = generations_padded.shape[1]

        # --- Actor Evaluation ---
        # Get logits from the actor model (without state=...)
        logits = actor_model(full_sequence)
        if isinstance(logits, tuple):  # Handle potential cache output
            logits = logits[0]

        # Select logits corresponding to the generated tokens
        # Logits for predicting token t are output at sequence position t-1
        action_logits = logits[:, prompt_len - 1 : -1, :]

        # Validate shape consistency
        if action_logits.shape[1] != gen_len:
            # Adjust if logits shape is off by one (common indexing issue)
            action_logits = logits[:, prompt_len - 1 : prompt_len + gen_len - 1, :]
            if action_logits.shape[1] != gen_len:
                raise ValueError(
                    f"Logit/Generation length mismatch in evaluate_actions: "
                    f"Logits shape {logits.shape}, Action Logits shape {action_logits.shape}, "
                    f"Gen length {gen_len}, Prompt length {prompt_len}"
                )

        # Calculate log probabilities and entropy from the distribution
        log_probs_dist = nn.log_softmax(action_logits, axis=-1)
        probs_dist = mx.softmax(action_logits, axis=-1)

        # Entropy calculation (masked by valid generation tokens)
        entropy_per_token = (
            -mx.sum(probs_dist * log_probs_dist, axis=-1) * generation_mask
        )
        sum_entropy = mx.sum(entropy_per_token, axis=1)
        num_valid_tokens = mx.sum(generation_mask, axis=1)
        # Avoid division by zero if mask is all zeros
        mean_sequence_entropy = sum_entropy / mx.maximum(num_valid_tokens, 1)

        # Get log probability of the actual generated actions
        action_log_probs = mx.take_along_axis(
            log_probs_dist, generations_padded[..., None], axis=-1
        ).squeeze(-1)

        # Apply mask and sum to get sequence log probability
        masked_action_log_probs = action_log_probs * generation_mask
        sequence_log_probs = mx.sum(masked_action_log_probs, axis=1)

        # --- Critic Evaluation ---
        # Get state embedding using the *same actor model instance* used for logits
        state_embeddings = self._get_state_embedding(actor_model, prompts_padded)

        # Get value prediction from the critic model (without state=...)
        values = critic_model(state_embeddings).squeeze(-1)

        # Evaluate results
        mx.eval(sequence_log_probs, values, mean_sequence_entropy)

        return sequence_log_probs, values, mean_sequence_entropy

    # ================================================================
    # === Core PPO Update Logic Implementation (MLX - Functional) ===
    # ================================================================
    def _compute_ppo_losses_and_grads_functional(
        self, batch: Dict[str, mx.array], actor_params: Dict, critic_params: Dict
    ) -> Tuple[Dict, Dict, Dict[str, float]]:
        """
        Computes PPO losses and gradients for one batch using functional calls.
        Returns actor_grads, critic_grads, loss_metrics.
        """

        # --- Actor Loss Function ---
        def actor_loss_fn(current_actor_params):
            # Create temporary model instances with the current parameters for evaluation
            # This ensures the forward passes use the parameters being differentiated
            temp_actor = self.actor
            temp_actor.update(tree_unflatten(list(current_actor_params.items())))

            # Critic parameters are fixed for actor loss calculation
            temp_critic = self.critic
            temp_critic.update(tree_unflatten(list(critic_params.items())))

            new_log_probs, _, entropy = self.evaluate_actions(
                temp_actor,  # Pass the model instance
                temp_critic,  # Pass the model instance
                batch["prompts_padded"],
                batch["generations_padded"],
                batch["generation_mask"],
            )
            log_probs_old = batch["log_probs_old"]  # Keep original shape (batch_size,)
            advantages = batch["advantages"]  # Keep original shape (batch_size,)
            # Ensure shapes match for ratio calculation
            new_log_probs = new_log_probs.reshape(log_probs_old.shape)
            entropy = entropy.reshape(advantages.shape)  # Entropy per sequence

            # Normalize advantages per-batch (optional, can improve stability)
            adv_mean = mx.mean(advantages)
            adv_std = mx.std(advantages) + 1e-8  # Epsilon for stability
            norm_advantages = (advantages - adv_mean) / adv_std

            # PPO Clipped Surrogate Objective
            log_ratio = new_log_probs - log_probs_old
            ratio = mx.exp(log_ratio)
            surr1 = ratio * norm_advantages
            surr2 = (
                mx.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * norm_advantages
            )
            actor_clip_loss = -mx.mean(mx.minimum(surr1, surr2))

            # Entropy Bonus (negative sign because we maximize entropy)
            entropy_loss = -mx.mean(entropy)

            # Total Actor Loss
            total_actor_loss = actor_clip_loss + self.entropy_coef * entropy_loss

            # Approximate KL divergence (for logging/debugging)
            # Ensure log_ratio has the same shape as ratio for mean calculation
            kl_approx = mx.mean(log_ratio).item()  # Average log ratio across batch
            ratio_mean = mx.mean(ratio).item()  # Average ratio across batch

            return total_actor_loss, (
                actor_clip_loss,
                entropy_loss,
                kl_approx,
                ratio_mean,
            )

        # --- Critic Loss Function ---
        def critic_loss_fn(current_critic_params):
            # Create temporary model instances with the current parameters
            temp_critic = self.critic
            temp_critic.update(tree_unflatten(list(current_critic_params.items())))

            # Actor parameters are fixed for critic loss calculation
            temp_actor = self.actor
            temp_actor.update(tree_unflatten(list(actor_params.items())))

            _, new_values, _ = self.evaluate_actions(
                temp_actor,  # Pass model instance
                temp_critic,  # Pass model instance
                batch["prompts_padded"],
                batch["generations_padded"],
                batch["generation_mask"],
            )
            returns = batch["returns"]  # Keep original shape (batch_size,)
            new_values = new_values.reshape(returns.shape)  # Ensure shapes match

            # Value Loss (Mean Squared Error)
            value_loss = mx.mean(mx.square(new_values - returns))

            # Total Critic Loss (scaled by coefficient)
            total_critic_loss = self.value_loss_coef * value_loss
            return total_critic_loss, value_loss  # Return base value loss too

        # --- Compute Gradients ---
        # Use value_and_grad for both actor and critic
        grad_actor_fn = mx.value_and_grad(actor_loss_fn, argnums=0)
        grad_critic_fn = mx.value_and_grad(critic_loss_fn, argnums=0)

        # Calculate actor loss, metrics, and gradients
        (
            actor_total_loss_val,
            (actor_clip_loss_val, entropy_loss_val, kl_approx, ratio_mean),
        ), actor_grads = grad_actor_fn(
            actor_params
        )  # Pass current actor params

        # Calculate critic loss and gradients
        (critic_total_loss_val, value_loss_val), critic_grads = grad_critic_fn(
            critic_params
        )  # Pass current critic params

        # --- Gradient Clipping (Optional) ---
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            actor_grads = optim.clip_grad_norm(actor_grads, self.grad_clip_norm)
            critic_grads = optim.clip_grad_norm(critic_grads, self.grad_clip_norm)

        # --- Prepare Loss Metrics Dictionary ---
        losses = {
            "actor_clip_loss": actor_clip_loss_val.item(),
            "entropy_loss": entropy_loss_val.item(),
            "actor_total_loss": actor_total_loss_val.item(),
            "value_loss": value_loss_val.item(),  # Log base value loss
            "critic_loss": critic_total_loss_val.item(),  # Log scaled critic loss
            "total_ppo_loss": actor_total_loss_val.item()
            + critic_total_loss_val.item(),
            "kl_approx": kl_approx,
            "ratio_mean": ratio_mean,
        }

        return actor_grads, critic_grads, losses

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Update actor/critic using PPO and data from the buffer (functional style)."""
        if buffer.advantages is None or buffer.returns is None:
            self.logger.error("Advantages/Returns not computed. Skipping update.")
            return {"actor_loss": 0, "critic_loss": 0, "total_ppo_loss": 0}

        total_losses_accum = {
            "actor_clip_loss": 0.0,
            "entropy_loss": 0.0,
            "actor_total_loss": 0.0,
            "value_loss": 0.0,  # Added
            "critic_loss": 0.0,
            "total_ppo_loss": 0.0,
            "kl_approx": 0.0,
            "ratio_mean": 0.0,
        }
        total_batches_processed = 0

        self.actor.train()  # Ensure models are in training mode
        self.critic.train()
        self.logger.info(
            f"Starting PPO update phase ({self.ppo_epochs} epochs, batch size {self.ppo_batch_size})..."
        )

        # PPO Update Loop (multiple epochs over the same rollout data)
        for ppo_epoch in range(self.ppo_epochs):
            batch_generator = buffer.get_batch_generator(
                self.ppo_batch_size, self.tokenizer
            )
            epoch_batches = 0
            for batch in batch_generator:
                if not batch:
                    continue  # Skip if batch generation failed

                # Get current parameters and optimizer states for the functional update step
                # We pass the *trainable* actor parameters to the loss function
                current_actor_params = (
                    self.actor.trainable_parameters()
                )  # Get trainable params
                current_critic_params = self.critic_params  # Critic params are separate
                current_actor_opt_state = self.actor_opt_state
                current_critic_opt_state = self.critic_opt_state

                try:
                    # Compute Gradients and Losses using current params
                    (
                        actor_grads,  # Gradients are only for trainable actor params
                        critic_grads,
                        losses,
                    ) = self._compute_ppo_losses_and_grads_functional(
                        batch, current_actor_params, current_critic_params
                    )

                    # Apply Gradients (Functional Update) - returns NEW state and NEW params
                    # Apply actor gradients only to the trainable parameters
                    (
                        new_actor_opt_state,
                        updated_actor_trainable_params,  # Optimizer updates only trainable part
                    ) = self.actor_optimizer.apply_gradients(
                        actor_grads, current_actor_params, current_actor_opt_state
                    )
                    # Apply critic gradients to all critic parameters
                    (
                        new_critic_opt_state,
                        updated_critic_params,
                    ) = self.critic_optimizer.apply_gradients(
                        critic_grads, current_critic_params, current_critic_opt_state
                    )

                    # Update the Agent's State:
                    # - Update the trainable parameters in the main actor model
                    # - Update the full critic parameters
                    # - Update optimizer states

                    # Update the stateful actor model with the updated *trainable* parameters
                    self.actor.update(
                        tree_unflatten(list(updated_actor_trainable_params.items()))
                    )
                    # Update the stateful critic model
                    self.critic.update(
                        tree_unflatten(list(updated_critic_params.items()))
                    )

                    # Update the parameter dictionaries stored in the agent (optional but good practice)
                    self.actor_params = (
                        self.actor.parameters()
                    )  # Recapture full params if needed elsewhere
                    self.critic_params = (
                        updated_critic_params  # Store updated critic params
                    )

                    # Update optimizer states stored in the agent
                    self.actor_opt_state = new_actor_opt_state
                    self.critic_opt_state = new_critic_opt_state
                    self.actor_optimizer.state = (
                        new_actor_opt_state  # Also update optimizer instance state
                    )
                    self.critic_optimizer.state = (
                        new_critic_opt_state  # Also update optimizer instance state
                    )

                    # Evaluate updated state to ensure computation happens and sync parameters
                    mx.eval(
                        updated_actor_trainable_params,  # Evaluate the things that changed
                        self.critic_params,
                        self.actor_opt_state,
                        self.critic_opt_state,
                    )

                    # Accumulate losses for averaging
                    for k, v in losses.items():
                        total_losses_accum[k] += v
                    epoch_batches += 1
                    total_batches_processed += 1

                except Exception as e:
                    self.logger.error(
                        f"Error during PPO batch update {epoch_batches+1} in epoch {ppo_epoch+1}: {e}",
                        exc_info=True,
                    )
                    # Optionally break or continue depending on severity
                    # break

            self.logger.debug(
                f"  PPO Epoch {ppo_epoch+1}/{self.ppo_epochs} finished ({epoch_batches} batches processed)."
            )
            # Check for shutdown request between epochs
            if shutdown_requested:
                self.logger.warning(
                    f"Shutdown requested during PPO epoch {ppo_epoch+1}. Stopping update phase."
                )
                break

        # Calculate average losses over all processed batches
        avg_losses = (
            {k: v / total_batches_processed for k, v in total_losses_accum.items()}
            if total_batches_processed > 0
            else total_losses_accum  # Avoid division by zero
        )
        self.logger.info(
            f"PPO Update finished. Avg Losses: { {k: f'{v:.4f}' for k,v in avg_losses.items()} }"
        )
        return avg_losses


# === Checkpoint and Resume Logic ===
def save_checkpoint(
    agent: PPOAgent, output_dir: Path, global_step: int, num_updates: int
):
    """Saves model, critic, optimizers, and training state."""
    # (Implementation remains the same as previous version)
    save_path = output_dir / f"checkpoint_update_{num_updates}_step_{global_step}"
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoint to {save_path}...")
    save_time = time.time()
    try:
        # Save actor weights (potentially LoRA adapters + base model structure)
        # save_weights handles saving potentially sharded models correctly
        # It saves model-*.safetensors and model.safetensors.index.json if sharded
        # It saves non-trainable weights by default.
        save_weights(
            str(save_path),
            agent.actor.parameters(),  # Save all actor params (including non-trainable)
        )

        # Save critic weights
        mx.save_safetensors(str(save_path / "critic.safetensors"), agent.critic_params)

        # Save optimizer states
        # Ensure state is saved correctly (as dict for safetensors)
        actor_opt_state_save = dict(tree_flatten(agent.actor_opt_state))
        critic_opt_state_save = dict(tree_flatten(agent.critic_opt_state))
        if actor_opt_state_save:  # Only save if not empty
            mx.save_safetensors(
                str(save_path / "actor_optimizer.safetensors"),
                actor_opt_state_save,
            )
        if critic_opt_state_save:  # Only save if not empty
            mx.save_safetensors(
                str(save_path / "critic_optimizer.safetensors"),
                critic_opt_state_save,
            )

        # Save training state (step, update, random states)
        state = {
            "global_step": global_step,
            "num_updates": num_updates,
            "np_random_state": np.random.get_state(),
            "random_state": random.getstate(),
            "mx_random_state": mx.random.get_state().tolist(),  # Convert mx state to list
        }
        with open(save_path / "training_state.json", "w") as f:
            # Use a default handler for non-serializable types if any sneak in
            json.dump(state, f, indent=4, default=lambda o: "<not serializable>")

        # Save tokenizer configuration
        tok_save_path = save_path / "tokenizer"
        tok_save_path.mkdir(exist_ok=True)
        # Handle potential nesting of tokenizer/processor
        tokenizer_to_save = (
            agent.tokenizer.processor
            if hasattr(agent.tokenizer, "processor")
            and hasattr(agent.tokenizer.processor, "save_pretrained")
            else agent.tokenizer
        )
        if hasattr(tokenizer_to_save, "save_pretrained"):
            try:
                tokenizer_to_save.save_pretrained(str(tok_save_path))
            except Exception as e:
                logger.error(f"Failed to save tokenizer using save_pretrained: {e}")
        else:
            logger.warning(
                "Could not save tokenizer automatically (no save_pretrained method found)."
            )

        logger.info(f"Checkpoint saved successfully ({time.time()-save_time:.2f}s).")

        # Update the latest checkpoint pointer file
        with open(output_dir / "checkpoint_latest.txt", "w") as f:
            f.write(str(save_path.resolve()))  # Use resolved path

    except Exception as e:
        logger.error(f"Failed to save checkpoint at {save_path}: {e}", exc_info=True)


def load_checkpoint(agent: PPOAgent, checkpoint_dir: Path) -> Tuple[int, int]:
    """Loads state from a checkpoint directory into the agent object."""
    logger.info(f"Attempting to load checkpoint from: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    try:
        # --- Load Actor Weights ---
        # mlx_lm load function handles loading potentially sharded models and config
        # We reload the model structure and then apply weights.
        # This assumes the checkpoint dir contains the necessary config files if the model was saved using save_weights.
        logger.info("Reloading actor model structure and weights...")
        # Note: This re-instantiates the model. If LoRA was used, it needs reapplying.
        # A better approach might be to load weights directly into the existing agent.actor
        # if the structure hasn't changed significantly. Let's try loading directly first.

        try:
            # Attempt to load weights directly into the existing model structure
            # This assumes the checkpoint was saved from a compatible model.
            # load_weights expects a directory path containing the safetensors files (and index if sharded)
            agent.actor.load_weights(str(checkpoint_dir))
            agent.actor_params = tree_map(
                lambda p: p, agent.actor.parameters()
            )  # Recapture potentially updated params
            logger.info("Loaded actor weights directly into existing model.")
            # If LoRA was used during the saved run, it should be part of these loaded weights.
            # We might need to re-apply the LoRA configuration logic if it's not saved/loaded automatically.
            # For now, assume load_weights handles adapter loading if saved correctly.

        except Exception as load_direct_error:
            logger.warning(
                f"Direct weight loading failed ({load_direct_error}). Trying full model reload..."
            )
            # Fallback: Reload the entire model and tokenizer from the checkpoint path
            # This requires the checkpoint dir to be a valid model directory (config.json etc.)
            try:
                loaded_model, loaded_tokenizer = load_mlx_model_and_tokenizer(
                    str(checkpoint_dir)
                )
                # Replace the agent's actor and tokenizer
                agent.actor = loaded_model
                agent.tokenizer = loaded_tokenizer
                agent.actor_params = tree_map(lambda p: p, agent.actor.parameters())
                # Re-check hidden size compatibility
                new_hidden_size = getattr(agent.actor.args, "hidden_size", None)
                if new_hidden_size is None and hasattr(
                    agent.actor, "config"
                ):  # Check config dict
                    new_hidden_size = agent.actor.config.get("hidden_size")

                if new_hidden_size != agent.hidden_size:
                    logger.warning(
                        f"Hidden size mismatch after reload ({agent.hidden_size} vs {new_hidden_size}). Critic might be incompatible."
                    )
                    # Potentially re-initialize critic if size changed? Risky.
                logger.info(
                    "Reloaded full actor model and tokenizer from checkpoint directory."
                )
            except Exception as reload_error:
                logger.error(
                    f"Failed to reload full model from checkpoint: {reload_error}",
                    exc_info=True,
                )
                raise reload_error  # Re-raise error if both methods fail

        # --- Load Critic Weights ---
        critic_weights_path = checkpoint_dir / "critic.safetensors"
        if not critic_weights_path.exists():
            raise FileNotFoundError(f"Critic weights not found: {critic_weights_path}")
        loaded_critic_weights = mx.load(str(critic_weights_path))
        agent.critic_params = tree_unflatten(list(loaded_critic_weights.items()))
        agent.critic.update(
            agent.critic_params
        )  # Apply weights to the stateful critic model
        logger.info("Loaded critic parameters.")

        # --- Load Optimizer States ---
        actor_opt_path = checkpoint_dir / "actor_optimizer.safetensors"
        critic_opt_path = checkpoint_dir / "critic_optimizer.safetensors"

        if actor_opt_path.exists():
            actor_opt_state_flat = list(mx.load(str(actor_opt_path)).items())
            agent.actor_opt_state = tree_unflatten(actor_opt_state_flat)
            agent.actor_optimizer.state = (
                agent.actor_opt_state
            )  # Update optimizer instance
            logger.info("Loaded actor optimizer state.")
        else:
            logger.warning(
                "Actor optimizer state not found in checkpoint. Resetting state."
            )
            # Reset optimizer state if not found
            agent.actor_opt_state = tree_map(
                lambda _: None, agent.actor.trainable_parameters()
            )
            agent.actor_optimizer.state = agent.actor_opt_state

        if critic_opt_path.exists():
            critic_opt_state_flat = list(mx.load(str(critic_opt_path)).items())
            agent.critic_opt_state = tree_unflatten(critic_opt_state_flat)
            agent.critic_optimizer.state = (
                agent.critic_opt_state
            )  # Update optimizer instance
            logger.info("Loaded critic optimizer state.")
        else:
            logger.warning(
                "Critic optimizer state not found in checkpoint. Resetting state."
            )
            # Reset optimizer state if not found
            agent.critic_opt_state = tree_map(lambda _: None, agent.critic.parameters())
            agent.critic_optimizer.state = agent.critic_opt_state

        # --- Load Training State ---
        state_path = checkpoint_dir / "training_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Training state file not found: {state_path}")
        with open(state_path, "r") as f:
            state = json.load(f)
        global_step = state["global_step"]
        num_updates = state["num_updates"]

        # Restore random states carefully
        try:
            np_state = state["np_random_state"]
            # Ensure correct format for np.random.set_state
            np.random.set_state(
                (
                    np_state[0],  # String identifier
                    np.array(np_state[1], dtype=np.uint32),  # Keys array
                    int(np_state[2]),  # Position
                    int(np_state[3]),  # Has Gaussian flag
                    float(np_state[4]),  # Cached Gaussian value
                )
            )
            random.setstate(tuple(state["random_state"]))  # Python random state
            mx.random.set_state(
                mx.array(state["mx_random_state"], dtype=mx.uint64)
            )  # MLX random state
            logger.info("Restored random states.")
        except KeyError as e:
            logger.warning(f"Could not restore random state ({e}), using current seed.")
        except Exception as e:
            logger.error(f"Error restoring random states: {e}", exc_info=True)

        logger.info(
            f"Resuming from global_step={global_step}, num_updates={num_updates}."
        )

        # Evaluate loaded parameters and states
        mx.eval(
            agent.actor_params,
            agent.critic_params,
            agent.actor_opt_state,
            agent.critic_opt_state,
        )
        return global_step, num_updates

    except Exception as e:
        logger.error(
            f"Failed to load checkpoint from {checkpoint_dir}: {e}", exc_info=True
        )
        raise  # Re-raise the exception


# === Evaluation Function ===
def evaluate(
    agent: PPOAgent, eval_env: LLMEnv, eval_steps: int, generation_temp: float = 0.0
) -> Dict[str, float]:
    """Performs evaluation on the validation set."""
    # (Implementation remains the same as previous version)
    logger.info(f"Starting evaluation for {eval_steps} steps...")
    agent.actor.eval()
    agent.critic.eval()
    total_reward = 0.0
    total_valid_format = 0.0
    actual_eval_steps = 0
    all_metrics = []  # Collect all step metrics

    # Reset environment and get initial state
    current_prompt_ids = eval_env.reset()
    if current_prompt_ids is None:
        logger.error("Evaluation environment failed initial reset.")
        return {"eval_mean_reward": 0.0, "eval_valid_format_pc": 0.0}

    # Setup progress bar if rich is available
    progress_context = (
        Progress(
            TextColumn("Evaluating..."),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,  # Bar disappears after completion
        )
        if RICH_AVAILABLE
        else contextlib.nullcontext()  # No progress bar otherwise
    )

    eval_task = None
    iterable = range(eval_steps)

    with progress_context as progress:
        if progress:  # Check if progress is the real Progress object
            eval_task = progress.add_task("Eval", total=eval_steps)
            # Use progress.track for automatic updates if available
            iterable = progress.track(iterable, task_id=eval_task)

        # Evaluation loop
        for i in iterable:  # Iterate using progress.track or range
            if current_prompt_ids is None or shutdown_requested:
                logger.warning(f"Evaluation interrupted at step {i+1}.")
                break

            actual_eval_steps += 1

            # Generate action (text) using the agent
            # Use the specified generation temperature (often 0.0 for deterministic eval)
            try:
                action_text, _, _, _, _ = agent.generate_action(
                    current_prompt_ids, temp=generation_temp
                )
            except Exception as e:
                logger.error(f"Evaluation step {i+1}: generate_action failed: {e}")
                action_text = "[Eval Gen Error]"  # Handle generation error

            # Step the environment with the generated action
            step_result = eval_env.step(action_text)
            if step_result is None:
                logger.warning(
                    f"Evaluation step {i+1}: env.step returned None. Resetting."
                )
                # Attempt to reset and continue if step fails
                current_prompt_ids = eval_env.reset()
                continue  # Skip reward accumulation for this failed step

            reward, metrics = step_result
            total_reward += reward
            total_valid_format += metrics.get("generation_valid_format", 0.0)
            all_metrics.append(metrics)  # Store metrics for potential aggregation

            # Reset environment for the next step
            current_prompt_ids = eval_env.reset()
            # Progress update is handled by progress.track if used

    # Restore training mode for models
    agent.actor.train()
    agent.critic.train()

    # Calculate final metrics
    mean_reward = total_reward / actual_eval_steps if actual_eval_steps > 0 else 0.0
    valid_format_pc = (
        (total_valid_format / actual_eval_steps) * 100.0
        if actual_eval_steps > 0
        else 0.0
    )

    # Aggregate other metrics if needed (example: average format score)
    avg_format_score = 0.0
    if all_metrics and actual_eval_steps > 0:
        format_scores = [
            m.get("format_score", 0.0) for m in all_metrics if "format_score" in m
        ]
        if format_scores:
            avg_format_score = sum(format_scores) / len(format_scores)

    logger.info(
        f"Evaluation finished. Mean Reward: {mean_reward:.4f}, Valid Format: {valid_format_pc:.2f}% ({actual_eval_steps}/{eval_steps} steps)"
    )

    eval_results = {
        "eval_mean_reward": mean_reward,
        "eval_valid_format_pc": valid_format_pc,
        "eval_avg_format_score": avg_format_score,  # Example aggregated metric
        "eval_steps_completed": actual_eval_steps,
    }
    return eval_results


# === Training Orchestration (MLX - Enhanced) ===
def train(
    model: nn.Module,
    critic: CriticNetwork,
    tokenizer: Any,
    train_env: LLMEnv,
    val_env: Optional[LLMEnv],
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    reward_fn: SimpleRewardFunction,
    # PPO Hyperparameters
    ppo_epochs: int,
    num_rollout_steps: int,
    ppo_batch_size: int,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
    grad_clip_norm: Optional[float],
    # Generation HPs
    max_gen_len: int,
    generation_temp: float,
    # Training Loop Control
    total_timesteps: int,
    save_freq: int,
    eval_freq: int,
    generate_sample_text_every: int,
    output_dir: str,
    metrics_log_file: str,
    resume_from: Optional[str] = None,
):
    """Main MLX PPO training loop with enhancements."""
    global shutdown_requested
    agent = PPOAgent(
        model,
        critic,
        tokenizer,
        actor_optimizer,
        critic_optimizer,
        gamma,
        gae_lambda,
        clip_epsilon,
        value_loss_coef,
        entropy_coef,
        max_gen_len,
        ppo_epochs,  # Pass ppo_epochs
        ppo_batch_size,  # Pass ppo_batch_size
        grad_clip_norm,
    )
    metrics_logger = MetricsLogger(log_file=metrics_log_file)
    output_path = Path(output_dir)

    # --- Resume Logic ---
    global_step = 0
    num_updates = 0
    if resume_from:
        checkpoint_path = Path(resume_from)
        checkpoint_dir = None
        # Check if it's the 'latest' pointer file
        if (
            checkpoint_path.is_file()
            and checkpoint_path.name == "checkpoint_latest.txt"
        ):
            try:
                checkpoint_dir_str = checkpoint_path.read_text().strip()
                checkpoint_dir = Path(checkpoint_dir_str)
                if not checkpoint_dir.is_dir():
                    logger.error(
                        f"Latest checkpoint path points to non-existent directory: {checkpoint_dir}"
                    )
                    checkpoint_dir = None  # Invalidate if directory doesn't exist
            except Exception as e:
                logger.error(
                    f"Could not read or resolve latest checkpoint file {checkpoint_path}: {e}"
                )
        # Check if it's directly a directory path
        elif checkpoint_path.is_dir():
            checkpoint_dir = checkpoint_path
        else:
            logger.error(
                f"Invalid resume path provided: {resume_from}. Must be a directory or checkpoint_latest.txt."
            )

        # Attempt loading if a valid directory was found
        if checkpoint_dir and checkpoint_dir.is_dir():
            try:
                global_step, num_updates = load_checkpoint(agent, checkpoint_dir)
                logger.info(f"Successfully resumed from checkpoint: {checkpoint_dir}")
            except Exception as e:
                logger.error(
                    f"Failed to load checkpoint from {checkpoint_dir}: {e}. Starting fresh.",
                    exc_info=True,
                )
                global_step = 0
                num_updates = 0
        else:
            logger.warning(
                f"Could not find a valid checkpoint at {resume_from}. Starting fresh."
            )
    else:
        logger.info("Starting training from scratch.")

    # --- Training Setup ---
    rollout_buffer = RolloutBuffer()
    start_time = time.time()
    last_save_update = num_updates  # Initialize based on potential resume
    last_eval_update = num_updates  # Initialize based on potential resume

    # Initial environment reset
    reset_result = train_env.reset()
    if reset_result is None:
        logger.critical(
            "Failed initial training environment reset! Cannot start training."
        )
        return  # Exit if env cannot be reset
    current_prompt_ids = reset_result

    logger.info(
        f"Starting training loop. Start Step: {global_step}, Start Update: {num_updates + 1}, Target Steps: {total_timesteps}"
    )

    # --- Main Loop Progress Bar Setup ---
    progress_context = (
        Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("Steps: {task.completed}/{task.total}"),
            TextColumn("Upd: {task.fields[updates]}"),
            TextColumn("Rew: {task.fields[reward]:.2f}"),
            TextColumn("KL: {task.fields[kl]:.2f}"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
            # transient=False # Keep bar visible until end
        )
        if RICH_AVAILABLE
        else contextlib.nullcontext()  # No progress bar if rich not available
    )

    # --- Main Training Loop ---
    with progress_context as progress:
        main_task = None
        if progress:  # Check if progress is the real Progress object
            main_task = progress.add_task(
                "[green]Training",
                total=total_timesteps,
                completed=global_step,  # Start from resumed step
                updates=num_updates,  # Start from resumed update
                reward=0.0,  # Initial values
                kl=0.0,
            )

        while global_step < total_timesteps and not shutdown_requested:
            # --- Rollout Phase ---
            rollout_phase_desc = f"[yellow]Rollout {num_updates + 1}..."
            if progress:
                progress.update(main_task, description=rollout_phase_desc)
            else:
                logger.info(rollout_phase_desc)  # Log phase start if no progress bar

            rollout_start_time = time.time()
            collected_rewards = []
            collected_steps_in_rollout = 0  # Track steps *within* this rollout
            rollout_buffer.clear()  # Clear buffer for new rollout data
            agent.actor.eval()  # Set models to eval mode for generation
            agent.critic.eval()

            # Collect samples until buffer is full or total timesteps reached
            for step_in_rollout in range(num_rollout_steps):
                if global_step >= total_timesteps or shutdown_requested:
                    break  # Exit inner loop if target reached or interrupted

                # Ensure we have a valid prompt
                if current_prompt_ids is None:
                    logger.warning(
                        f"Rollout step {step_in_rollout+1}: current_prompt_ids is None. Resetting env."
                    )
                    reset_result = train_env.reset()
                    if reset_result is None:
                        logger.error(
                            "Rollout: Env reset failed permanently. Stopping training."
                        )
                        shutdown_requested = True
                        break
                    current_prompt_ids = reset_result
                    continue  # Skip this step if reset was needed

                # Generate action and get associated data
                try:
                    (
                        action_text,
                        seq_log_prob,  # Log prob of the generated sequence
                        value,  # Value estimate of the prompt state
                        gen_ids,  # IDs of the generated sequence
                        full_seq_ids,  # Prompt + Generated IDs
                    ) = agent.generate_action(current_prompt_ids, temp=generation_temp)
                except Exception as gen_e:
                    logger.error(
                        f"Rollout step {step_in_rollout+1} (Global {global_step+1}): generate_action failed: {gen_e}",
                        exc_info=True,
                    )
                    # Attempt to reset env and continue
                    reset_result = train_env.reset()
                    if reset_result is None:
                        logger.error(
                            "Rollout: Env reset failed after generation error. Stopping."
                        )
                        shutdown_requested = True
                        break
                    current_prompt_ids = reset_result
                    continue

                # Step the environment with the generated text
                step_result = train_env.step(action_text)
                if step_result is None:
                    logger.warning(
                        f"Rollout step {step_in_rollout+1} (Global {global_step+1}): env.step returned None. Resetting."
                    )
                    reset_result = train_env.reset()
                    if reset_result is None:
                        logger.error(
                            "Rollout: Env reset failed after env step error. Stopping."
                        )
                        shutdown_requested = True
                        break
                    current_prompt_ids = reset_result
                    continue  # Skip storing data for this failed step

                reward, step_info = step_result
                collected_rewards.append(reward)

                # Add experience to the buffer
                # We store the prompt, generation, log_prob, reward, done=True, value
                rollout_buffer.add(
                    current_prompt_ids,
                    gen_ids,
                    action_text,
                    seq_log_prob,
                    reward,
                    True,  # Assume 'done' after each generation for PPO update
                    value,
                )

                # Increment global step count and steps collected in this rollout
                global_step += 1
                collected_steps_in_rollout += 1

                # Get the next prompt
                reset_result = train_env.reset()
                if reset_result is None:
                    logger.error(
                        f"Rollout: Env reset failed after step {global_step}. Stopping rollout phase."
                    )
                    break  # Exit rollout loop if env cannot be reset
                current_prompt_ids = reset_result

                # Update progress bar for global steps
                if progress:
                    progress.update(main_task, advance=1)  # Advance global step count

                # Log sample generation periodically
                if (
                    generate_sample_text_every > 0
                    and global_step % generate_sample_text_every == 0
                ):
                    # Use logger for sample text to avoid cluttering console with progress bar
                    logger.info(
                        f"\n--- Sample Step {global_step} ---\n"
                        f"Prompt: {train_env.current_prompt_text[:200]}...\n"  # Use the prompt *before* the reset
                        f"Gen: {action_text[:300]}...\n"
                        f"Reward: {reward:.3f}\n"
                        f"---------------------------"
                    )

            # --- End of Rollout Phase ---
            mean_reward_rollout = (
                np.mean(collected_rewards) if collected_rewards else 0.0
            )
            rollout_duration = time.time() - rollout_start_time
            logger.info(
                f"Rollout {num_updates + 1} ({collected_steps_in_rollout} steps) finished ({rollout_duration:.2f}s). Mean reward: {mean_reward_rollout:.3f}"
            )
            if progress:
                progress.update(
                    main_task, reward=mean_reward_rollout
                )  # Update reward field in progress bar

            # Check if buffer has data before proceeding
            if len(rollout_buffer) == 0:
                logger.warning(
                    "Rollout buffer is empty after collection phase, skipping PPO update."
                )
                # Ensure we don't get stuck if env keeps failing reset
                if collected_steps_in_rollout == 0 and not shutdown_requested:
                    logger.error(
                        "Failed to collect any steps in rollout. Stopping due to potential env issue."
                    )
                    shutdown_requested = True
                continue  # Skip to next rollout iteration

            # --- Compute Advantages and Returns ---
            compute_gae_desc = f"[cyan]Calculating GAE..."
            if progress:
                progress.update(main_task, description=compute_gae_desc)
            else:
                logger.info(compute_gae_desc)

            try:
                # Get value of the *next* state (which is the state after the last rollout step)
                # Ensure current_prompt_ids is valid before getting value
                if current_prompt_ids is None:
                    logger.warning(
                        "Cannot compute GAE: next state (current_prompt_ids) is None. Resetting."
                    )
                    reset_result = train_env.reset()
                    if reset_result is None:
                        logger.error(
                            "Failed to reset env for GAE computation. Skipping update."
                        )
                        continue
                    current_prompt_ids = reset_result

                # Calculate last value needed for GAE
                last_value = agent.get_value(current_prompt_ids)
                rollout_buffer.compute_advantages_and_returns(
                    last_value, gamma, gae_lambda
                )
            except Exception as e:
                logger.error(
                    f"Failed to compute GAE: {e}. Skipping PPO update.", exc_info=True
                )
                continue  # Skip update if GAE fails

            # --- PPO Update Phase ---
            ppo_update_desc = f"[blue]PPO Update {num_updates + 1}..."
            if progress:
                progress.update(main_task, description=ppo_update_desc)
            else:
                logger.info(ppo_update_desc)

            update_start_time = time.time()
            # Perform PPO updates using the collected buffer data
            avg_losses = agent.update(
                rollout_buffer
            )  # This runs the inner PPO epochs/batches
            update_duration = time.time() - update_start_time
            num_updates += 1  # Increment update counter *after* successful update

            # Update progress bar fields
            if progress:
                progress.update(
                    main_task,
                    updates=num_updates,
                    kl=avg_losses.get("kl_approx", 0.0),  # Update KL divergence field
                )
            logger.info(
                f"Update {num_updates} finished ({update_duration:.2f}s). Losses: { {k: f'{v:.4f}' for k,v in avg_losses.items()} }"
            )

            # --- Log Metrics ---
            log_data = {
                "global_step": global_step,
                "update_step": num_updates,
                "timestamp": time.time(),
                "mean_rollout_reward": mean_reward_rollout,
                "rollout_steps": collected_steps_in_rollout,
                "rollout_duration_sec": rollout_duration,
                "update_duration_sec": update_duration,
                "memory_peak_gb": mx.get_peak_memory() / 1e9,
                "phase": "train",
                **avg_losses,  # Include all loss components
            }
            metrics_logger.log(log_data)

            # --- Evaluation Step ---
            if val_env and num_updates % eval_freq == 0:
                eval_desc = f"[magenta]Evaluating..."
                if progress:
                    progress.update(main_task, description=eval_desc)
                else:
                    logger.info(eval_desc)

                eval_start_time = time.time()
                # Determine number of eval steps (e.g., full validation set or a fixed number)
                num_eval_steps = len(val_env)  # Evaluate on the full validation set
                eval_metrics = evaluate(
                    agent,
                    val_env,
                    eval_steps=num_eval_steps,
                    generation_temp=0.0,  # Use temp=0 for deterministic eval
                )
                eval_duration = time.time() - eval_start_time
                logger.info(f"Evaluation took {eval_duration:.2f}s.")

                # Log evaluation metrics
                eval_log = {
                    "global_step": global_step,
                    "update_step": num_updates,
                    "timestamp": time.time(),
                    "phase": "eval",
                    "eval_duration_sec": eval_duration,
                    **eval_metrics,
                }
                metrics_logger.log(eval_log)
                last_eval_update = num_updates

            # --- Save Checkpoint Step ---
            if num_updates % save_freq == 0:
                save_desc = f"[green]Saving Ckpt {num_updates}..."
                if progress:
                    progress.update(main_task, description=save_desc)
                else:
                    logger.info(save_desc)

                save_checkpoint(agent, output_path, global_step, num_updates)
                last_save_update = num_updates

            # --- Cleanup after update ---
            rollout_buffer.clear()  # Clear buffer for next iteration
            gc.collect()  # Explicit garbage collection

        # --- End of Main Training Loop ---
        if progress:
            progress.update(
                main_task,
                description="[bold green]Training Finished"
                if not shutdown_requested
                else "[bold yellow]Training Interrupted",
            )

    train_duration = time.time() - start_time
    logger.info(
        f"Training loop finished or interrupted. Total Updates: {num_updates}, Total Steps: {global_step}"
    )
    logger.info(f"Total training time: {train_duration / 3600:.2f} hours.")
    if shutdown_requested:
        logger.warning("Training was interrupted by user (SIGINT).")

    # --- Final Save ---
    # Save if interrupted or if the last update wasn't saved
    if num_updates > last_save_update or shutdown_requested:
        logger.info("Saving final model state...")
        save_checkpoint(agent, output_path, global_step, num_updates)

    logger.info(f"Script finished. Check logs and outputs in {output_path}")


# === CLI Interface (Enhanced) ===
@click.command()
# Required Paths
@click.option(
    "--model-path",
    required=True,
    type=str,
    help="Path or HF repo ID of the base Llama model (MLX format).",
)
@click.option(
    "--train-dataset-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help='Path to training JSONL (with "prompt", "completion").',
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to save checkpoints, logs, and final model.",
)
# Optional Paths
@click.option(
    "--val-dataset-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Optional path to validation JSONL.",
)
@click.option(
    "--resume-from",
    default=None,
    type=click.Path(exists=True, path_type=Path),  # Path object validation
    help='Path to checkpoint directory or "checkpoint_latest.txt" to resume training.',
)
@click.option(
    "--metrics-log-file",
    default="metrics.jsonl",
    type=str,
    show_default=True,
    help="Filename for saving detailed metrics within the output directory.",
)
# Training Duration & Control
@click.option(
    "--total-timesteps",
    default=10,
    type=int,
    show_default=True,
    help="Total environment interaction steps (global steps) for training.",
)
@click.option("--seed", default=42, type=int, show_default=True, help="Random seed.")
@click.option(
    "--lora-layers",
    default=-1,
    type=int,
    show_default=True,
    help="Number of layers to apply LoRA to (from the top). -1 means no LoRA/full fine-tune.",
)
# Rollout & PPO Hyperparameters
@click.option(
    "--num-rollout-steps",
    default=256,
    type=int,
    show_default=True,
    help="Number of environment steps collected per rollout before PPO update.",
)
@click.option(
    "--ppo-batch-size",
    default=32,
    type=int,
    show_default=True,
    help="Mini-batch size for PPO update phase.",
)
@click.option(
    "--ppo-epochs",
    default=4,
    type=int,
    show_default=True,
    help="Number of optimization epochs over rollout data per PPO update.",
)
@click.option(
    "--actor-lr",
    default=2e-6,  # Adjusted default LR, often needs tuning
    type=float,
    show_default=True,
    help="Learning rate for the actor (LLM).",
)
@click.option(
    "--critic-lr",
    default=1e-5,  # Adjusted default LR, often needs tuning
    type=float,
    show_default=True,
    help="Learning rate for the critic.",
)
@click.option(
    "--gamma",
    default=0.99,
    type=float,
    show_default=True,
    help="Discount factor (GAE).",
)
@click.option(
    "--gae-lambda",
    default=0.95,
    type=float,
    show_default=True,
    help="Lambda factor (GAE).",
)
@click.option(
    "--clip-epsilon",
    default=0.2,
    type=float,
    show_default=True,
    help="PPO clipping epsilon.",
)
@click.option(
    "--value-loss-coef",
    default=0.5,  # Standard coefficient, might need tuning
    type=float,
    show_default=True,
    help="Coefficient for the critic's value loss.",
)
@click.option(
    "--entropy-coef",
    default=0.01,  # Standard coefficient, might need tuning
    type=float,
    show_default=True,
    help="Coefficient for the entropy bonus in actor loss.",
)
@click.option(
    "--grad-clip-norm",
    default=1.0,  # Common value for gradient clipping
    type=float,
    show_default=True,
    help="Maximum norm for gradient clipping (0 or negative to disable).",
)
# Generation & Environment Hyperparameters
@click.option(
    "--max-prompt-len",
    default=512,
    type=int,
    show_default=True,
    help="Maximum length for tokenized prompts (truncation applied).",
)
@click.option(
    "--max-gen-len",
    default=512,
    type=int,
    show_default=True,
    help="Maximum number of tokens generated per environment step.",
)
@click.option(
    "--generation-temp",
    default=0.7,  # Temperature for sampling during rollouts
    type=float,
    show_default=True,
    help="Temperature for sampling during generation in rollouts.",
)
# Reporting & Saving Frequencies
@click.option(
    "--save-freq",
    default=50,  # Save every 50 PPO updates
    type=int,
    show_default=True,
    help="Save checkpoint every N PPO UPDATES.",
)
@click.option(
    "--eval-freq",
    default=25,  # Evaluate every 25 PPO updates
    type=int,
    show_default=True,
    help="Evaluate every N PPO UPDATES (requires validation set).",
)
@click.option(
    "--generate-sample-text-every",
    default=1000,  # Log sample text every 1000 global steps
    type=int,
    show_default=True,
    help="Log sample generation every N global steps (0 to disable).",
)
# Verbosity
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def cli_main(**kwargs):
    """Fine-tunes a Llama model using PPO and MLX with enhanced features."""
    # --- Configure Logging ---
    log_level_name = kwargs.pop("log_level").upper()
    log_level = getattr(logging, log_level_name)
    root_logger = logging.getLogger()  # Get root logger
    root_logger.setLevel(log_level)  # Set level for all handlers unless overridden

    # Remove existing handlers to avoid duplicates if script is re-run in same process
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure handler (Rich if available, basic otherwise)
    if RICH_AVAILABLE:
        rich_handler = RichHandler(
            rich_tracebacks=True,  # Enable rich tracebacks
            show_path=log_level <= logging.DEBUG,  # Show path only in debug
            console=console,  # Use the globally defined console
            markup=True,  # Enable markup formatting
            log_time_format="[%Y-%m-%d %H:%M:%S]",  # Consistent time format
        )
        # Use a simple format focusing on the message for Rich
        rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        root_logger.addHandler(rich_handler)
    else:
        # Basic stream handler if Rich is not available
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="[%Y-%m-%d %H:%M:%S]",
            )
        )
        root_logger.addHandler(stream_handler)

    global logger
    logger = logging.getLogger(__name__)  # Get logger for this module
    logger.info(f"Logging level set to {log_level_name}")
    logger.info(f"Rich logging {'enabled' if RICH_AVAILABLE else 'enabled'}.")

    # Check essential dependencies
    if not MLX_LM_AVAILABLE or not REWARD_FUNC_AVAILABLE:
        logger.critical(
            "Missing essential dependencies (mlx-lm or reward function). Please install/check imports. Exiting."
        )
        sys.exit(1)

    # --- Seed ---
    seed = kwargs["seed"]
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    logger.info(f"Set random seed to {seed}")

    # --- Output Dir & Metrics Log ---
    output_dir: Path = kwargs["output_dir"]
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.resolve()}")
    except OSError as e:
        logger.critical(
            f"Failed to create output directory {output_dir}: {e}. Exiting."
        )
        sys.exit(1)

    metrics_log_file = output_dir / kwargs.pop("metrics_log_file")
    logger.info(f"Metrics log file: {metrics_log_file}")

    # --- Load Model & Tokenizer ---
    model_path_str = kwargs["model_path"]
    logger.info(f"Loading base model and tokenizer from: {model_path_str}")
    try:
        # Use mlx_lm's load utility
        model, tokenizer = load_mlx_model_and_tokenizer(model_path_str)

        # Determine hidden size robustly
        hidden_size = None
        if hasattr(model, "args") and hasattr(model.args, "hidden_size"):
            hidden_size = model.args.hidden_size
        elif (
            hasattr(model, "config")
            and isinstance(model.config, dict)
            and "hidden_size" in model.config
        ):
            hidden_size = model.config["hidden_size"]
        elif (
            hasattr(model, "model")
            and hasattr(model.model, "config")
            and isinstance(model.model.config, dict)
            and "hidden_size" in model.model.config
        ):
            hidden_size = model.model.config[
                "hidden_size"
            ]  # Handle nested model attribute

        if hidden_size is None:
            # Attempt to guess from parameters as a last resort
            params = model.parameters()
            for p in tree_flatten(params)[0]:
                if isinstance(p, mx.array) and p.ndim > 0:
                    hidden_size = p.shape[-1]
                    logger.warning(
                        f"Guessed hidden_size={hidden_size} from parameter shapes."
                    )
                    break
            if hidden_size is None:
                raise ValueError(
                    "Could not determine hidden_size from model args, config, or parameters."
                )

        logger.info(
            f"Model loaded. Type: {getattr(model, 'model_type', 'N/A')}, Hidden Size: {hidden_size}"
        )

        # Ensure tokenizer has a pad token ID, defaulting to EOS if necessary
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                logger.info(
                    f"Set tokenizer pad_token_id to eos_token_id ({tokenizer.pad_token_id})"
                )
            else:
                # Set a default pad token ID if neither pad nor EOS exists (e.g., 0)
                # This might require adjustment based on the specific tokenizer
                tokenizer.pad_token_id = 0
                logger.warning(
                    f"Tokenizer lacks pad_token_id and eos_token_id. Defaulting pad_token_id to 0. This might be incorrect."
                )

        # Evaluate model parameters after loading
        mx.eval(model.parameters())

    except Exception as e:
        logger.critical(
            f"Failed to load model/tokenizer from {model_path_str}: {e}", exc_info=True
        )
        sys.exit(1)

    # --- Apply LoRA if requested ---
    lora_layers = kwargs.pop("lora_layers")
    if lora_layers > 0:
        logger.info(f"Applying LoRA to the top {lora_layers} layers...")
        try:
            # Example LoRA config (consider making these CLI args)
            lora_config = {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "scale": 10.0,
            }
            model.freeze()  # Freeze base model parameters
            # Apply LoRA adapters
            linear_to_lora_layers(
                model, lora_layers, lora_config
            )  # use_dora defaults to False
            mx.eval(model.parameters())  # Evaluate parameters after adding adapters

            # Log trainable parameter count
            trainable_params_count = sum(
                p.size
                for p in tree_flatten(model.trainable_parameters())[
                    0
                ]  # Use index 0 for leaves
            )
            total_params_count = sum(
                p.size
                for p in tree_flatten(model.parameters())[0]  # Use index 0 for leaves
            )
            logger.info(
                f"LoRA applied. Trainable params: {trainable_params_count / 1e6:.3f}M "
                f"({(trainable_params_count / total_params_count) * 100:.2f}% of total)"
            )
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("LoRA not applied (lora_layers <= 0). Performing full fine-tuning.")

    # --- Reward Function ---
    # Use default RewardConfig, can be customized if needed
    reward_config = RewardConfig()
    reward_function = SimpleRewardFunction(
        config=reward_config, verbose=(log_level <= logging.DEBUG)
    )
    logger.info("Initialized SimpleRewardFunction.")

    # --- Environments ---
    try:
        logger.info(f"Loading training dataset from: {kwargs['train_dataset_path']}")
        train_prompt_dataset = JsonlPromptDataset(str(kwargs["train_dataset_path"]))
        train_env = LLMEnv(
            train_prompt_dataset,
            tokenizer,
            reward_function,
            kwargs["max_prompt_len"],
            "train",  # Environment ID
        )
        logger.info(f"Training environment created with {len(train_env)} samples.")

        val_env = None
        if kwargs["val_dataset_path"]:
            try:
                logger.info(
                    f"Loading validation dataset from: {kwargs['val_dataset_path']}"
                )
                val_prompt_dataset = JsonlPromptDataset(str(kwargs["val_dataset_path"]))
                val_env = LLMEnv(
                    val_prompt_dataset,
                    tokenizer,
                    reward_function,
                    kwargs["max_prompt_len"],
                    "val",  # Environment ID
                )
                logger.info(
                    f"Validation environment created with {len(val_env)} samples."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize validation environment from {kwargs['val_dataset_path']}: {e}. Skipping validation.",
                    exc_info=log_level <= logging.DEBUG,
                )
        else:
            logger.info("No validation dataset path provided, skipping validation.")

    except FileNotFoundError as e:
        logger.critical(f"Dataset file not found: {e}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.critical(
            f"Failed to initialize training environment: {e}", exc_info=True
        )
        sys.exit(1)

    # --- Critic & Optimizers ---
    logger.info(f"Initializing CriticNetwork with input dimension {hidden_size}.")
    critic = CriticNetwork(input_dim=hidden_size)
    mx.eval(critic.parameters())  # Evaluate initial critic parameters

    # Ensure optimizers only target trainable parameters
    actor_params_to_optimize = (
        model.trainable_parameters()
    )  # Gets LoRA params if applied, else all params
    critic_params_to_optimize = critic.parameters()  # Critic is fully trainable

    actor_optimizer = optim.AdamW(learning_rate=kwargs["actor_lr"])
    critic_optimizer = optim.AdamW(learning_rate=kwargs["critic_lr"])

    # Initialize optimizer states correctly based on the parameters they optimize
    # Note: State is handled internally by the PPOAgent during load/save/update
    # actor_optimizer.state = tree_map(lambda _: None, actor_params_to_optimize)
    # critic_optimizer.state = tree_map(lambda _: None, critic_params_to_optimize)

    logger.info(
        f"Critic initialized. Optimizers created (AdamW - Actor LR: {kwargs['actor_lr']}, Critic LR: {kwargs['critic_lr']})"
    )

    # --- Prepare Args & Run Training ---
    # Extract resume path before passing kwargs to train function
    resume_from_path = kwargs.pop("resume_from")
    resume_from_str = str(resume_from_path) if resume_from_path else None

    # Filter kwargs to only include those expected by the train function
    # This avoids passing CLI-specific args like 'log_level' or paths already processed
    train_func_expected_args = [
        "ppo_epochs",
        "num_rollout_steps",
        "ppo_batch_size",
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "value_loss_coef",
        "entropy_coef",
        "grad_clip_norm",
        "max_gen_len",
        "generation_temp",
        "total_timesteps",
        "save_freq",
        "eval_freq",
        "generate_sample_text_every",
    ]
    train_hyperparams = {k: kwargs[k] for k in train_func_expected_args if k in kwargs}

    # Log the hyperparameters being used for the training run
    logger.info("Starting PPO training with the following hyperparameters:")
    for key, value in train_hyperparams.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  actor_lr: {kwargs['actor_lr']}")
    logger.info(f"  critic_lr: {kwargs['critic_lr']}")
    logger.info(f"  max_prompt_len: {kwargs['max_prompt_len']}")
    logger.info(f"  seed: {kwargs['seed']}")
    logger.info(f"  lora_layers: {lora_layers}")  # Log the actual value used

    try:
        # Call the main training function
        train(
            model=model,
            critic=critic,
            tokenizer=tokenizer,
            train_env=train_env,
            val_env=val_env,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            reward_fn=reward_function,
            output_dir=str(output_dir),  # Pass resolved paths as strings
            metrics_log_file=str(metrics_log_file),
            resume_from=resume_from_str,  # Pass resume path as string or None
            **train_hyperparams,  # Pass the filtered hyperparameters
        )
    except Exception as e:
        logger.critical(
            f"Unhandled exception during training process: {e}", exc_info=True
        )
        # Attempt a final save on critical failure? Maybe too risky if state is corrupt.
        # Consider adding a try/finally block around train() for final save attempt.
        sys.exit(1)
    finally:
        logger.info("Training script finished or exited.")


if __name__ == "__main__":
    cli_main()
