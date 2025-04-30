# mlx_ppo_trainer_v6.py

import json
import random
import click
import sys
import math
import gc
import logging
import os
import time
import re
import signal
import csv
from typing import Tuple, Dict, Any, List, Optional, Union, Generator, Set
from dataclasses import dataclass, field, asdict, fields
from functools import partial
from pathlib import Path

# Import MLX components
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map

# Import rich
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
)
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table

# Numpy
import numpy as np

# Tiktoken (optional)
try:
    import tiktoken

    _temp_logger = logging.getLogger(__name__)
    try:
        LENGTH_CHECK_TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        LENGTH_CHECK_TOKENIZER = None
        _temp_logger.warning("Tiktoken cl100k_base failed.")
except ImportError:
    _temp_logger = logging.getLogger(__name__)
    _temp_logger.warning("tiktoken not found.")
    LENGTH_CHECK_TOKENIZER = None

# MLX LM specific imports
try:
    from mlx_lm.utils import (
        fetch_from_hub,
        # save_weights, # Replaced with manual saving logic
        load_config,
        get_model_path,
        load,           # Used for initial loading if not resuming
        _get_classes,   # Helper to get model classes from config
    )
    # Import specific model classes if needed for instantiation during resume
    # We might need to make this more dynamic if supporting more than llama
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.llama import ModelArgs as LlamaModelArgs

    # Use TokenizerWrapper directly
    from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
except ImportError as e:
    print(f"Import error (mlx_lm): {e}. Install mlx-lm.", file=sys.stderr)
    sys.exit(1)

# Local reward function import
try:
    from reward import SimpleRewardFunction, RewardConfig
except ImportError as e:
    print(f"Import error (reward): {e}. Ensure reward.py exists.", file=sys.stderr)
    sys.exit(1)

# llm_templates for chat formatting
try:
    from llm_templates import Formatter, Conversation, Content
except ImportError as e:
    print(f"Import error (llm_templates): {e}. Install llm_templates.", file=sys.stderr)
    sys.exit(1)


# --- Global Variables ---
logger = logging.getLogger(__name__)
console = Console()
shutdown_requested = False


# --- Signal Handler ---
def handle_signal(signum, frame):
    """Handles SIGINT and SIGTERM for graceful shutdown."""
    global shutdown_requested
    if not shutdown_requested:
        logger.warning(f"Signal {signum} received. Requesting shutdown...")
        shutdown_requested = True
    else:
        logger.warning("Shutdown already requested. Force exiting...")
        sys.exit(1)


# --- Utility Functions ---

# NOTE: This is the first definition from the original code.
# It's used in LLMEnv to format the prompt *before* deduplication and training rollouts.
# It adds a <thinking> tag, possibly to guide the model's generation start.
def create_chat_text_format(
    prompt: str, final_completion: str, model_name: str = "llama31" # Default model name used here
) -> str:
    """Creates the chat formatted string using llm_templates, adding a thinking prompt."""
    try:
        # Construct messages list in standard format expected by llm_templates
        messages = [
            Content(role="user", content=prompt),
            Content(role="assistant", content=final_completion), # Completion is empty when called in LLMEnv
        ]
        # Note: model_name parameter is currently hardcoded/defaulted here.
        conversation = Conversation(model=model_name, messages=messages)
        # Assuming default formatter works; adjust if specific template needed
        formatter = Formatter()
        # Render *with* adding the assistant prompt turn marker at the end
        # This prepares the input for the model to start generating the assistant's response.
        formatted_str = formatter.render(conversation, add_assistant_prompt=True)

        # Add the thinking tag after the assistant prompt marker
        return formatted_str + "\n<thinking>\n"
    except Exception as e:
        logging.error(f"Error creating chat text format: {e}")
        # Return a basic fallback representation
        return f"User: {prompt.strip()}\nAssistant: {final_completion.strip()}"


def _extract_block_content(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Extracts content between specified start and end tags using regex."""
    if not text or not isinstance(text, str):
        return None
    try:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    except Exception as e:
        logger.error(f"Regex error in _extract_block_content: {e}", exc_info=False)
        return None


def validate_text_format(
    generated_text: str, config: RewardConfig
) -> Tuple[bool, List[str]]:
    """Validates if the generated text contains required tags in the correct order."""
    warnings = []
    tags = config.special_tokens
    t_s, t_e, a_s, a_e = (
        tags["think_start"],
        tags["think_end"],
        tags["answer_start"],
        tags["answer_end"],
    )
    req = [t_s, t_e, a_s, a_e] # Required tags

    if not isinstance(generated_text, str) or not generated_text.strip():
        return False, ["Input empty or not a string."]

    pos = {}
    missing, multiple = [], []
    for tag in req:
        indices = [m.start() for m in re.finditer(re.escape(tag), generated_text)]
        count = len(indices)
        if count == 0:
            missing.append(tag)
            pos[tag] = -1
        elif count > 1:
            multiple.append(tag)
            pos[tag] = indices[0] # Use first occurrence if multiple
        else:
            pos[tag] = indices[0]

    if missing:
        warnings.append(f"Missing tags: {','.join(missing)}")
    if multiple:
        warnings.append(f"Multiple occurrences of tags: {','.join(multiple)}")

    if missing or multiple:
        return False, warnings

    # Check order: think_start < think_end < answer_start < answer_end
    p0, p1, p2, p3 = pos[t_s], pos[t_e], pos[a_s], pos[a_e]
    if not (p0 < p1 < p2 < p3):
        warnings.append(f"Tag order incorrect: ThinkS({p0}) < ThinkE({p1}) < AnsS({p2}) < AnsE({p3}) is False.")
        return False, warnings

    # Check for non-whitespace content between think_end and answer_start
    sep = generated_text[p1 + len(t_e) : p2]
    if sep.strip():
        warnings.append("Non-whitespace content found between </thinking> and <answer> tags.")
        return False, warnings # Fail if there's content between blocks

    return True, warnings # Format is valid


def truncate_completion_smart(
    completion_text: str,
    prompt: str,
    max_total_tokens: int,
    tokenizer: TokenizerWrapper,
    config: RewardConfig,
) -> Optional[str]:
    """
    Truncates the <thinking> block of a completion if the total token count
    (prompt + completion) exceeds the maximum, while preserving the <answer> block.
    Returns None if truncation is impossible or results in invalid format.
    """
    tags = config.special_tokens
    t_s, t_e, a_s, a_e = (
        tags["think_start"],
        tags["think_end"],
        tags["answer_start"],
        tags["answer_end"],
    )
    sep = "\n" # Separator used between tags and content
    _encode = tokenizer.encode # Local alias for encoding function

    # 1. Initial length check
    try:
        prompt_tokens = _encode(prompt)
        completion_tokens = _encode(completion_text)
        if len(prompt_tokens) + len(completion_tokens) <= max_total_tokens:
            return completion_text # No truncation needed
    except Exception as e:
        logger.warning(f"Tokenization error during initial length check: {e}")
        return None # Cannot proceed if tokenization fails

    # 2. Extract content blocks
    think_c = _extract_block_content(completion_text, t_s, t_e)
    answer_c = _extract_block_content(completion_text, a_s, a_e)

    if think_c is None or answer_c is None:
        logger.warning("Smart truncation failed: Could not extract thinking or answer blocks.")
        return None

    # 3. Calculate fixed token budget
    try:
        answer_toks = _encode(answer_c)
        # Calculate tokens needed for structure (tags + separators)
        struct_text = f"{t_s}{sep}{t_e}{sep}{a_s}{sep}{a_e}"
        struct_toks_n = len(_encode(struct_text))
    except Exception as e:
        logger.warning(f"Tokenization error calculating fixed token budget: {e}")
        return None

    fixed_toks = len(prompt_tokens) + len(answer_toks) + struct_toks_n
    available_think_tokens = max_total_tokens - fixed_toks
    min_think_tokens = 1 # Require at least one token for the thinking block

    if available_think_tokens < min_think_tokens:
        logger.debug(f"Smart truncation failed: Not enough tokens available for thinking block ({available_think_tokens} < {min_think_tokens}).")
        return None

    # 4. Truncate thinking block if necessary
    try:
        think_toks = _encode(think_c)
        if len(think_toks) > available_think_tokens:
            trunc_think_ids = think_toks[:available_think_tokens]
            try:
                # Decode truncated tokens, handle potential decoding errors
                decoded_think = tokenizer.decode(trunc_think_ids).strip()
                # Ensure decoded text isn't empty after stripping
                trunc_think = decoded_think if decoded_think else "..."
            except Exception as dec_e:
                logger.warning(f"Decoding error during thinking block truncation: {dec_e}")
                trunc_think = "..." # Fallback if decoding fails
            logger.debug(f"Truncated thinking block from {len(think_toks)} to {len(trunc_think_ids)} tokens.")
        else:
            trunc_think = think_c # No truncation needed for thinking block
    except Exception as e:
        logger.warning(f"Tokenization error during thinking block processing: {e}")
        return None

    # 5. Reconstruct and validate
    reconstructed = (
        f"{t_s}{sep}{trunc_think}{sep}{t_e}{sep}{a_s}{sep}{answer_c}{sep}{a_e}"
    )

    # Final length check (optional but recommended)
    try:
        final_total_tokens = len(_encode(prompt)) + len(_encode(reconstructed))
        # Allow a small buffer (e.g., 5 tokens) for minor tokenization variations
        if final_total_tokens > max_total_tokens + 5:
            logger.warning(
                f"Smart truncation post-check failed: Final token count {final_total_tokens} exceeds limit {max_total_tokens}."
            )
            return None
    except Exception as e:
        logger.warning(f"Tokenization error during final length check: {e}")
        return None

    # Final format validation
    is_valid, warnings = validate_text_format(reconstructed, config)
    if not is_valid:
        logger.warning(f"Smart truncation resulted in invalid format: {warnings}")
        return None

    return reconstructed


# --- Configuration Dataclass ---
@dataclass
class TrainingArgs:
    # Paths
    train_dataset_path: str
    model_path: str # Base model path (used if not resuming)
    output_dir: str
    val_dataset_path: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None # Path to checkpoint directory to resume from
    # Model & Tokenizer
    max_prompt_len: int = 750 # Default increased
    max_gen_len: int = 50 # Default increased
    # Optimizers
    actor_lr: float = 5e-6
    critic_lr: float = 1e-5
    # PPO Hyperparameters
    ppo_epochs: int = 2 # Default reduced
    num_rollout_steps: int = 128 # Default increased
    ppo_batch_size: int = 32 # Default reduced
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    target_kl: Optional[float] = 0.05 # Set > 1.0 to disable KL check
    # Training Control
    total_timesteps: int = 100000
    save_every: int = 10
    eval_every: int = 20
    log_every: int = 1 # Less relevant with progress bar
    generate_samples_every: int = 5
    seed: int = 42
    shuffle_data: bool = True


# --- Custom Dataset (Iterable) ---
class JsonlIterableDataset:
    """Reads, validates, and yields records from JSONL, with progress."""

    def __init__(
        self, file_path: str, tokenizer: TokenizerWrapper, reward_config: RewardConfig
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.reward_config = reward_config
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized dataset reader for: {self.file_path}")

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        total_lines = self.__len__() if self.__len__() > 0 else None
        skipped = 0
        yielded = 0
        try:
            with open(self.file_path, "r", encoding="utf-8") as f, Progress(
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                TextColumn("[magenta]{task.percentage:>3.0f}%"),
                TextColumn("• Items: {task.fields[items]}"),
                TimeElapsedColumn(),
                console=console,
                transient=False,  # Keep progress bar visible after completion
            ) as progress:
                task_id = progress.add_task(
                    f"Loading {os.path.basename(self.file_path)}",
                    total=total_lines,
                    items="0/0",
                )
                for i, line in enumerate(f):
                    line_num = i + 1
                    line = line.strip()
                    progress.update(task_id, advance=1)
                    if not line:
                        skipped += 1
                        continue
                    try:
                        data = json.loads(line)
                        # Only 'prompt' is strictly required for PPO rollouts
                        if "prompt" not in data or not data["prompt"]:
                            self.logger.debug(
                                f"L{line_num}: Skipping - missing or empty required key 'prompt'."
                            )
                            skipped += 1
                            continue

                        # Check 'completion' format only if it exists (for validation/reference)
                        # This helps filter out bad reference data but doesn't stop training if missing.
                        completion_text = data.get("completion")
                        if completion_text is not None:
                            is_valid, warnings = validate_text_format(
                                completion_text, self.reward_config
                            )
                            if not is_valid:
                                self.logger.debug(
                                    f"L{line_num}: Skipping - invalid reference completion format: {warnings}. Completion: {completion_text[:100]}..."
                                )
                                skipped += 1
                                continue
                        # If prompt is valid (and completion, if present, is valid), yield it.
                        yield data
                        yielded += 1
                    except json.JSONDecodeError:
                        self.logger.warning(f"L{line_num}: Skipping - invalid JSON.")
                        skipped += 1
                    except Exception as e:
                        self.logger.warning(f"L{line_num}: Skipping - processing error: {e}")
                        skipped += 1
                    finally:
                        # Update progress with yielded/skipped counts
                        progress.update(task_id, items=f"{yielded}/{skipped}")

                # Final update to progress bar status
                progress.update(
                    task_id,
                    items=f"{yielded}/{skipped}",
                    completed=progress.tasks[0].total if total_lines else i + 1,
                    description=f"Loaded {os.path.basename(self.file_path)} ({yielded} items)",
                )
        except Exception as e:
            self.logger.error(f"Fatal dataset read error for {self.file_path}: {e}", exc_info=True)
            raise

    def __len__(self):
        """Provides an efficient estimate of the number of lines in the file."""
        try:
            # More efficient line count using buffered reading
            with open(self.file_path, "rb") as f:
                lines = 0
                buf_size = 1024 * 1024
                read_f = f.raw.read # type: ignore # Access raw for performance
                buf = read_f(buf_size)
                while buf:
                    lines += buf.count(b"\n")
                    buf = read_f(buf_size)
                # Add one if the last line doesn't end with a newline
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    lines += 1
            return lines
        except Exception as e:
            self.logger.warning(
                f"Could not accurately count lines in {self.file_path} using fast method: {e}. Falling back to slower iteration count."
            )
            # Fallback to slower method if fast counting fails
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                     return sum(1 for _ in f)
            except Exception as e_slow:
                 self.logger.error(f"Could not count lines with fallback method either: {e_slow}")
                 return 0


# --- MLX Environment ---
class LLMEnv:
    """Environment for LLM PPO training using MLX."""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: TokenizerWrapper,
        reward_function: SimpleRewardFunction,
        max_prompt_len: int,
        env_id: str, # Identifier (e.g., "train", "val")
        shuffle_data: bool,
    ):
        self.tokenizer = tokenizer
        self.reward_fn = reward_function
        self.max_prompt_len = max_prompt_len
        self.env_id = env_id
        self.reward_config = reward_function.config
        self.logger = logging.getLogger(__name__)
        self.shuffle_data = shuffle_data

        self.logger.info(
            f"Initializing env '{self.env_id}': Loading/validating/deduplicating data from {dataset_path}..."
        )

        # Load initial data using the iterable dataset (handles validation)
        initial_data = list(
            JsonlIterableDataset(dataset_path, tokenizer, self.reward_config)
        )
        if not initial_data:
            raise ValueError(f"No valid data loaded from {dataset_path} for env '{self.env_id}'.")

        # Deduplicate based on prompt (after formatting) + reference completion (if exists)
        unique_data_list: List[Dict] = []
        seen_keys: Set[Tuple[str, str]] = set()
        duplicates_found = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[blue]{task.percentage:>3.0f}%"),
            console=console,
            transient=True, # Hide after completion
        ) as progress:
            task_id = progress.add_task(
                f"Deduplicating {env_id}", total=len(initial_data)
            )
            for record in initial_data:
                # --- Prompt Formatting Step ---
                # Format the raw prompt using the chat template *before* deduplication.
                # This ensures the model always sees prompts in the desired chat format
                # during rollouts, including the initial assistant/thinking prompt structure.
                # The completion passed here is empty as we only format the user prompt part.
                formatted_prompt = create_chat_text_format(record['prompt'], "")

                # Use formatted prompt + reference completion (if exists) as key for uniqueness
                key = (formatted_prompt, record.get("completion", ""))
                progress.update(task_id, advance=1)

                if key not in seen_keys:
                    seen_keys.add(key)
                    # Store the *formatted* prompt in the record for use during training
                    record['prompt'] = formatted_prompt
                    unique_data_list.append(record)
                else:
                    duplicates_found += 1

        self.logger.info(
            f"Removed {duplicates_found} duplicates from {env_id}. Kept {len(unique_data_list)} unique records."
        )
        self.data = unique_data_list
        del initial_data, seen_keys # Free memory
        gc.collect()

        if not self.data:
            raise ValueError(f"No unique valid data remaining for env '{self.env_id}'.")

        # Initial shuffle if required
        if self.shuffle_data:
            random.shuffle(self.data)

        self.data_iterator = iter(self.data)
        self.current_sample: Optional[Dict] = None # Holds the current sample being processed
        self.logger.info(f"LLMEnv '{self.env_id}' ready with {len(self.data)} samples.")

    def _get_next_sample(self) -> Optional[Dict]:
        """Retrieves the next sample, reshuffling if the dataset is exhausted."""
        try:
            return next(self.data_iterator)
        except StopIteration:
            self.logger.debug(
                f"'{self.env_id}' dataset exhausted. Resetting iterator{ ' and shuffling' if self.shuffle_data else ''}."
            )
            if self.shuffle_data:
                random.shuffle(self.data)
            self.data_iterator = iter(self.data)
            try:
                # Try fetching again after reset
                return next(self.data_iterator)
            except StopIteration:
                # This should only happen if the dataset was empty initially
                self.logger.error(f"'{self.env_id}' dataset is empty even after resetting iterator!")
                return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching next sample in '{self.env_id}': {e}")
            return None

    def reset(self) -> Optional[Tuple[mx.array, str, str]]:
        """
        Resets the environment by fetching a new sample.
        Returns:
            - prompt_mx (mx.array): Tokenized and potentially truncated prompt IDs.
            - prompt_text (str): The original (formatted) prompt text.
            - ref_completion (str): The reference completion text, if available, otherwise empty string.
            Returns None if fetching a sample fails.
        """
        self.current_sample = self._get_next_sample()
        if self.current_sample is None:
            self.logger.error(f"Failed to get next sample during reset for env '{self.env_id}'.")
            return None # Indicate failure

        # Retrieve the (already formatted) prompt and reference completion
        prompt_text = self.current_sample["prompt"]
        ref_completion = self.current_sample.get("completion", "") # Use reference if available

        try:
            # Tokenize the formatted prompt
            prompt_ids = self.tokenizer.encode(prompt_text)

            # Truncate prompt if it exceeds max length
            if len(prompt_ids) > self.max_prompt_len:
                prompt_ids = prompt_ids[: self.max_prompt_len]
                self.logger.debug(f"Prompt truncated to {self.max_prompt_len} tokens for env '{self.env_id}'.")

            prompt_mx = mx.array(prompt_ids)
            mx.eval(prompt_mx) # Ensure array is materialized

        except Exception as e:
            self.logger.error(
                f"Tokenization error during reset for prompt '{prompt_text[:100]}...' in env '{self.env_id}': {e}"
            )
            # Try to recover by fetching the next sample recursively
            return self.reset()

        # Reset the reward function state for the new sample
        self.reward_fn.reset()

        return prompt_mx, prompt_text, ref_completion

    def step(self, generated_text: str) -> Optional[Tuple[float, Dict[str, Any]]]:
        """
        Calculates the reward for the generated text based on the current sample.
        Args:
            generated_text (str): The text generated by the model.
        Returns:
            - reward_val (float): The calculated reward.
            - metrics (Dict): Additional metrics from the reward function.
            Returns None if called before a successful reset or if reward calculation fails.
        """
        if self.current_sample is None:
            self.logger.error("step() called before successful reset(). Cannot calculate reward.")
            return None

        # Get reference completion from the current sample (might be empty)
        ref_completion = self.current_sample.get("completion", "")

        try:
            # Calculate reward using the provided reward function
            reward_val, metrics = self.reward_fn.calculate_reward(
                str(generated_text), str(ref_completion) # Ensure inputs are strings
            )
            # Ensure reward is a float
            return float(reward_val), metrics
        except Exception as e:
            self.logger.error(f"Reward calculation error in env '{self.env_id}': {e}", exc_info=True)
            # Return minimum possible reward and error info on failure
            return self.reward_fn.config.min_reward_clip, {"error": str(e)}

    def __len__(self):
        """Returns the number of unique samples in the environment."""
        return len(self.data)


class SelfAttention(nn.Module):
    """
    Custom self-attention module.
    NOTE: Consider using the built-in `mlx.nn.MultiHeadAttention` if possible,
          as it might be more optimized and feature-rich. This custom version
          is kept for compatibility with the original code structure.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0): # Dropout not used here
        super().__init__()

        # Validate dimensions
        if dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({dim}) must be divisible by number of heads ({num_heads})"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim) # For scaled dot-product attention

        # Linear projections for Q, K, V and output
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dropout probability (currently not applied in forward pass)
        self.dropout = dropout

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass for self-attention."""
        # Handle different input dimensions (e.g., single vector vs. batch of sequences)
        input_ndim = x.ndim
        if input_ndim == 1:
            # Input: [dim] -> Reshape to [1, 1, dim] (batch=1, seq_len=1)
            x = x.reshape(1, 1, -1)
        elif input_ndim == 2:
            # Assume input is [batch, dim] -> Reshape to [batch, 1, dim] (seq_len=1)
            x = x.reshape(x.shape[0], 1, -1)
        elif input_ndim != 3:
            raise ValueError(f"Unsupported input ndim for SelfAttention: {input_ndim}")

        batch_size, seq_len, _ = x.shape

        # 1. Project to queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape for multi-head attention
        # [batch, seq_len, dim] -> [batch, seq_len, num_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim] for batch matrix multiplication
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # 3. Compute attention scores (scaled dot-product)
        # (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
        attn_weights = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        # 4. Apply mask if provided (e.g., for causal attention)
        # Mask should be additive (e.g., -inf for masked positions)
        if mask is not None:
            attn_weights = attn_weights + mask

        # 5. Apply softmax to get attention probabilities
        attn_weights = mx.softmax(attn_weights, axis=-1)

        # NOTE: Dropout would typically be applied here during training
        # if self.dropout > 0.0 and self.training:
        #     attn_weights = nn.dropout(attn_weights, p=self.dropout)

        # 6. Apply attention weights to values
        # (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
        attn_output = mx.matmul(attn_weights, v)

        # 7. Reshape back to original dimensions
        # Transpose back: [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        # Concatenate heads: [batch, seq_len, dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.dim)

        # 8. Final linear projection
        output = self.out_proj(attn_output)

        # Reshape output to match original input shape if necessary
        if input_ndim == 1:
            output = output.reshape(-1) # Return flat vector [dim]
        elif input_ndim == 2:
             output = output.reshape(batch_size, self.dim) # Return [batch, dim]

        return output

    def update_shared(self, params):
        """Helper method to update parameters, potentially for shared optimization schemes."""
        self.update(params)


class CriticNetwork(nn.Module):
    """
    Critic network for PPO. Estimates the value function.
    Uses a small transformer block (SelfAttention + FFN) to process hidden states.
    Handles both sequence inputs ([batch, seq_len, hidden]) and single state inputs ([batch, hidden]).
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,      # Number of heads for self-attention
        ffn_multiplier: int = 4, # Multiplier for FFN intermediate dim
        dropout: float = 0.0,    # Dropout rate (currently not applied)
    ):
        super().__init__()

        # --- Small Transformer Block ---
        self.norm1 = nn.LayerNorm(hidden_size)
        # Uses the custom SelfAttention defined above
        self.attn = SelfAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout, # Pass dropout, though SelfAttention doesn't apply it yet
        )
        self.norm2 = nn.LayerNorm(hidden_size)

        # Feed-Forward Network (FFN) part
        ffn_dim = hidden_size * ffn_multiplier
        self.ffn_1 = nn.Linear(hidden_size, ffn_dim, bias=False) # Expansion
        self.ffn_2 = nn.Linear(ffn_dim, hidden_size, bias=False) # Contraction

        # --- Value Head ---
        # Linear layer projecting the final hidden state to a single value estimate
        self.value_head = nn.Linear(hidden_size, 1)

    def __call__(self, hidden_states: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass for the critic.
        Args:
            hidden_states (mx.array): Input hidden states. Can be:
                                      - [batch, seq_len, hidden] (processes sequence)
                                      - [batch, hidden] (processes single state per batch item)
            mask (Optional[mx.array]): Additive mask for the self-attention layer.
        Returns:
            mx.array: Value estimates with shape [batch, 1].
        """
        if hidden_states.ndim == 2:
            # Input is [batch, hidden] - Treat as single state, no sequence processing needed
            processed_state = hidden_states

        elif hidden_states.ndim == 3:
            # Input is [batch, seq_len, hidden] - Process through the transformer block
            # Pre-Normalization architecture (like Llama)

            # 1. Self-Attention with residual connection
            residual = hidden_states
            x = self.norm1(hidden_states)
            x = self.attn(x, mask)
            x = x + residual # Add back original input

            # 2. FFN with residual connection
            residual = x
            y = self.norm2(x)
            y = self.ffn_1(y)
            y = nn.gelu(y) # GELU activation
            y = self.ffn_2(y)
            processed_state_seq = x + y # Add back input to FFN

            # Pool the sequence information. Using mean pooling here.
            # Alternatives: Use last token state (x[:, -1, :]) or first token state ([CLS]).
            processed_state = processed_state_seq.mean(axis=1) # [batch, hidden]

        else:
            raise ValueError(f"Unsupported input ndim for CriticNetwork: {hidden_states.ndim}")

        # --- Value Head ---
        # Project the final processed state to a scalar value
        value_estimate = self.value_head(processed_state) # [batch, 1]
        return value_estimate

    def update_shared(self, params):
        """Helper method to update parameters, potentially for shared optimization schemes."""
        self.update(params)


# --- MLX Rollout Buffer ---
@dataclass
class RolloutBuffer:
    """Stores experiences collected during PPO rollouts."""
    prompt_ids: List[mx.array] = field(default_factory=list)      # List of prompt token ID arrays
    generated_ids: List[mx.array] = field(default_factory=list)   # List of generated token ID arrays
    actions_text: List[str] = field(default_factory=list)         # List of generated text sequences
    log_probs: List[mx.array] = field(default_factory=list)       # List of sequence log probabilities (scalar mx.array)
    rewards: List[float] = field(default_factory=list)            # List of rewards (float)
    values: List[mx.array] = field(default_factory=list)          # List of value estimates from critic (scalar mx.array)
    dones: List[bool] = field(default_factory=list)               # List of done flags (currently always True per step)

    # Calculated during compute_advantages_and_returns
    advantages: Optional[mx.array] = None # Shape [num_steps, 1]
    returns: Optional[mx.array] = None    # Shape [num_steps, 1]

    logger = logging.getLogger(__name__)

    def add(
        self,
        prompt: mx.array,
        gen_ids: mx.array,
        action_text: str,
        log_prob: mx.array,  # Should be scalar sequence log prob
        reward: float,
        done: bool,
        value: mx.array,     # Should be scalar value prediction
    ):
        """Adds a single step of experience to the buffer."""
        # Basic validation
        if not isinstance(log_prob, mx.array) or log_prob.size != 1:
             self.logger.warning(f"Adding non-scalar log_prob to buffer: shape={log_prob.shape}")
        if not isinstance(value, mx.array) or value.size != 1:
             self.logger.warning(f"Adding non-scalar value to buffer: shape={value.shape}")

        # Ensure arrays are standard MLX arrays (usually not needed unless mixing frameworks)
        self.prompt_ids.append(prompt)
        self.generated_ids.append(gen_ids)
        self.actions_text.append(action_text)
        self.log_probs.append(log_prob.reshape(())) # Ensure scalar shape
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.reshape(())) # Ensure scalar shape

    def _pad_and_stack(self, arrays: List[mx.array], pad_value: int = 0) -> mx.array:
        """Pads a list of 1D mx.arrays to the same length and stacks them."""
        if not arrays:
            return mx.array([], dtype=mx.int32) # Return empty int array if list is empty

        # Ensure all arrays are 1D before padding
        arrays_1d = [(arr.reshape(-1) if arr.ndim > 1 else arr) for arr in arrays]
        if not arrays_1d:
            return mx.array([], dtype=mx.int32) # Should not happen if initial check passed

        # Determine dtype from the first non-empty array
        first_valid_arr = next((arr for arr in arrays_1d if arr.size > 0), None)
        dtype = first_valid_arr.dtype if first_valid_arr is not None else mx.int32

        # Handle case where all arrays are empty
        if all(a.size == 0 for a in arrays_1d):
            # Return empty array with the determined (or default) dtype
             return mx.array([], dtype=dtype)

        # Find max length among non-empty arrays
        max_len = max((arr.size for arr in arrays_1d if arr.size > 0), default=0)
        if max_len == 0:
            # This case should be covered by the 'all empty' check above, but belt-and-suspenders
            return mx.array([], dtype=dtype)

        padded_list = []
        for arr in arrays_1d:
            if arr.size == max_len:
                padded_list.append(arr)
            elif arr.size > 0: # Pad shorter arrays
                pad_width = max_len - arr.size
                # Pad only on the right side (axis 1 effectively)
                padded_list.append(mx.pad(arr, (0, pad_width), constant_values=pad_value))
            else: # arr.size == 0 (handle empty arrays explicitly)
                # Create a full array of pad values
                padded_list.append(mx.full((max_len,), pad_value, dtype=dtype))

        # Stack the padded arrays along the batch dimension (axis 0)
        return mx.stack(padded_list, axis=0)

    def compute_advantages_and_returns(
        self, last_value: mx.array, gamma: float, gae_lambda: float
    ):
        """
        Computes Generalized Advantage Estimation (GAE) advantages and returns.
        Assumes each step added corresponds to a full episode (done=True).
        """
        num_steps = len(self.rewards)
        if num_steps == 0:
            self.advantages = mx.array([])
            self.returns = mx.array([])
            self.logger.warning("Cannot compute GAE: Rollout buffer is empty.")
            return

        # Ensure last_value (value estimate for the state *after* the last rollout step) is scalar
        if last_value.size != 1:
            self.logger.warning(
                f"last_value size is {last_value.size}, expected 1. Taking mean for GAE calculation."
            )
            last_value = mx.mean(last_value) # Attempt recovery by taking mean
        last_value = last_value.reshape(()) # Ensure scalar shape [()]

        # Prepare values from the buffer, ensuring they are scalar and float32
        values_list = []
        for i, v in enumerate(self.values):
            if v.size == 1:
                values_list.append(v.reshape(()).astype(mx.float32))
            else:
                # This shouldn't happen if add() ensures scalar, but handle defensively
                self.logger.warning(
                    f"Buffer value at index {i} has size {v.size}, expected 1. Taking mean."
                )
                values_list.append(mx.mean(v).reshape(()).astype(mx.float32))

        # Stack buffer values and append the last_value for easier GAE calculation loop
        all_values = mx.stack(values_list + [last_value.astype(mx.float32)], axis=0)
        rewards_tensor = mx.array(self.rewards, dtype=mx.float32)
        # Dones tensor (currently all True, meaning next_non_terminal is always 0)
        # dones_tensor = mx.array(self.dones, dtype=mx.bool_)

        # Calculate GAE advantages iterating backwards
        advantages_list = []
        last_gae_lam = mx.array(0.0, dtype=mx.float32) # Initialize GAE trace

        for step in reversed(range(num_steps)):
            # Value of current state V(s_t) and next state V(s_{t+1})
            current_values = all_values[step]
            next_values = all_values[step + 1]

            # Determine if the next state is non-terminal (1.0 if not done, 0.0 if done)
            # Since we assume each step is a full episode, next_non_terminal is always 0.0
            next_non_terminal = 0.0 # Based on self.dones[step] being True

            # Calculate the TD error (delta)
            delta = (
                rewards_tensor[step]
                + gamma * next_values * next_non_terminal # This term becomes 0
                - current_values
            )

            # Update the GAE trace: A_t = delta_t + gamma * lambda * next_non_terminal * A_{t+1}
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam # This term also becomes 0

            # Prepend advantage to maintain correct order
            advantages_list.insert(0, last_gae_lam)

        if not advantages_list: # Should not happen if num_steps > 0
            self.advantages = mx.array([])
            self.returns = mx.array([])
            return

        # Stack advantages and calculate returns
        self.advantages = mx.stack(advantages_list, axis=0).reshape(-1, 1) # Shape [num_steps, 1]
        # Returns are Advantages + Values
        values_tensor = mx.stack(values_list, axis=0).reshape(-1, 1) # Shape [num_steps, 1]
        self.returns = self.advantages + values_tensor

        self.logger.debug(
            f"Computed GAE: Advantages shape={self.advantages.shape}, Returns shape={self.returns.shape}, "
            f"Adv mean={mx.mean(self.advantages).item():.3f}, Ret mean={mx.mean(self.returns).item():.3f}"
        )
        mx.eval(self.advantages, self.returns) # Ensure computation is finished

    def get_batch_indices(self, batch_size: int) -> Generator[np.ndarray, None, None]:
        """Generates random batches of indices for iterating over the buffer."""
        n_samples = len(self.prompt_ids)
        if n_samples == 0:
            self.logger.warning("Attempted to get batch indices from empty buffer.")
            return # Yield nothing if buffer is empty

        indices = np.random.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            yield indices[i : i + batch_size]

    def clear(self):
        """Clears all stored experiences from the buffer."""
        self.prompt_ids.clear()
        self.generated_ids.clear()
        self.actions_text.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None
        # Optional: Trigger garbage collection if memory is a concern after clearing
        # gc.collect() # Use judiciously


# --- MLX PPO Agent ---
class PPOAgent:
    """PPO Agent managing MLX Actor (LLM) and Critic networks."""

    def __init__(
        self,
        actor_model: nn.Module,      # The language model (e.g., LlamaModel from mlx_lm)
        critic_model: CriticNetwork, # The value network
        tokenizer: TokenizerWrapper,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        value_loss_coef: float,
        entropy_coef: float,
        max_gen_len: int,
    ):
        self.actor = actor_model
        self.critic = critic_model
        self.tokenizer = tokenizer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_gen_len = max_gen_len
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized PPO Agent (MLX) with max_gen_len={max_gen_len}.")

        # Verify actor has expected structure for critic input if possible
        # This helps catch potential issues early if a non-standard model is used.
        if not hasattr(self.actor, "model") or not callable(getattr(self.actor, "model", None)):
            self.logger.warning(
                "Actor model instance does not have a standard `.model` attribute "
                "containing the base transformer (expected for optimal critic input). "
                "Will attempt fallback using full actor pass for critic input."
            )

    def get_value(self, prompt_ids: mx.array) -> mx.array:
        """
        Gets value prediction from the critic for given prompt IDs.
        Args:
            prompt_ids (mx.array): Token IDs of the prompt, shape [batch, seq_len] or [seq_len].
        Returns:
            mx.array: Value estimate(s), shape [batch] or scalar [()].
        """
        try:
            # Ensure batch dimension exists
            if prompt_ids.ndim == 1:
                prompt_ids = prompt_ids[None, :] # Add batch dimension: [1, seq_len]

            # --- Get Hidden States from Actor ---
            # Prefer using the base model output if available (more efficient)
            # NOTE: This assumes the actor (LLM) follows the mlx_lm convention
            #       of having a `.model` attribute for the base transformer layers.
            base_output = None
            if hasattr(self.actor, "model") and callable(self.actor.model):
                # Pass prompt through base model only, no KV cache needed for value estimate
                actor_model_output = self.actor.model(prompt_ids, cache=None)

                # Handle different return types from base model
                if isinstance(actor_model_output, tuple) and len(actor_model_output) >= 1:
                    base_output = actor_model_output[0] # Assume first element is hidden states
                elif isinstance(actor_model_output, mx.array):
                    base_output = actor_model_output # Assume it directly returns hidden states
                else:
                    self.logger.warning(
                        f"Unexpected output type from actor.model: {type(actor_model_output)}. Falling back."
                    )
            else:
                 self.logger.debug("actor.model attribute not found or not callable. Using full actor pass.")

            # Fallback: Use the full actor forward pass if base model access failed
            if base_output is None:
                self.logger.warning(
                    "Using full actor forward pass for critic input (suboptimal)."
                )
                actor_output = self.actor(prompt_ids, cache=None)
                if isinstance(actor_output, tuple) and len(actor_output) >= 1:
                    # Use the first output (likely logits or hidden states) as input for critic
                    base_output = actor_output[0]
                elif isinstance(actor_output, mx.array):
                    base_output = actor_output
                else:
                    raise ValueError(
                        f"Unexpected output from full actor pass: {type(actor_output)}"
                    )

            # --- Pass Hidden States to Critic ---
            # The critic network handles sequence pooling (if input is 3D)
            value = self.critic(base_output) # Expect shape [batch, 1]
            mx.eval(value)

            # Return value, squeezed to remove trailing dimension: [batch] or [()]
            return value.squeeze(-1)

        except Exception as e:
            self.logger.error(f"Error getting value from critic: {e}", exc_info=True)
            # Return zeros as a fallback, matching batch size
            return mx.zeros((prompt_ids.shape[0],))

    def generate_step(
        self, prompt: mx.array, temp: float = 0.7
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        """
        Step-by-step token generation using the actor model with KV caching.
        Yields the next token ID and its log probability at each step.
        Args:
            prompt (mx.array): Initial prompt token IDs, shape [seq_len].
            temp (float): Temperature for sampling. 0 means greedy.
        Yields:
            Tuple[mx.array, mx.array]: (next_token_id [shape (1,)], log_prob_of_token [shape (1,)])
        """
        cache = None # Initialize KV cache
        # Add batch dimension for model input: [1, seq_len]
        current_input = prompt[None, :]

        try:
            # 1. Process the initial prompt to populate the KV cache
            actor_output = self.actor(current_input, cache=cache)

            # Check output format - standard mlx_lm models return (logits, cache)
            if isinstance(actor_output, tuple) and len(actor_output) == 2:
                logits, cache = actor_output # Unpack logits and updated cache
            elif isinstance(actor_output, mx.array):
                # Handle case where only logits are returned (cache might not be supported/used)
                self.logger.warning(
                    "Actor model returned only logits during initial step, expected (logits, cache). "
                    "KV caching may be disabled or malfunctioning."
                )
                logits = actor_output
                # cache remains None, generation loop might fail if model requires cache
            else:
                raise ValueError(
                    f"Actor model returned unexpected output format: type={type(actor_output)}. "
                    "Expected tuple (logits, cache) or mx.array (logits)."
                )

            # Get logits for the *next* token prediction (after the prompt)
            # Logits shape: [batch, seq_len, vocab_size] -> [1, seq_len, vocab_size]
            next_token_logits = logits[:, -1, :] # Shape: [1, vocab_size]

            # Sample the first token and get its log probability
            next_token, log_prob = self._sample_token(next_token_logits, temp)
            # next_token shape [1], log_prob shape [1]
            mx.eval(next_token, log_prob) # Ensure computation before yield
            yield next_token, log_prob

            # 2. Generate subsequent tokens one by one using the cache
            count = 1
            while True:
                # Input for the next step is the *last generated token*
                current_input = next_token[:, None] # Shape [1, 1]

                # Pass the single token and the *updated* cache
                actor_output = self.actor(current_input, cache=cache) # Cache might be None

                # Check output format again for the generation loop
                if isinstance(actor_output, tuple) and len(actor_output) == 2:
                    logits, cache = actor_output # Update cache if returned
                elif isinstance(actor_output, mx.array):
                    self.logger.debug(
                        "Actor model returned only logits during generation loop. Cache not updated."
                    )
                    logits = actor_output
                    # Cache remains unchanged, potentially stale or None
                else:
                    raise ValueError(
                        f"Actor model returned unexpected output format during generation: type={type(actor_output)}."
                    )

                # Get logits for the very next token
                next_token_logits = logits[:, -1, :] # Shape: [1, vocab_size]

                # Sample the next token and get its log probability
                next_token, log_prob = self._sample_token(next_token_logits, temp)
                mx.eval(next_token, log_prob) # Evaluate before yielding and checking termination

                yield next_token, log_prob
                count += 1

                # Check termination conditions AFTER yielding
                token_item = next_token.item() # Get Python int value for comparison
                # Stop if EOS token is generated or max length is reached
                if (
                    token_item == self.tokenizer.eos_token_id
                    or count >= self.max_gen_len
                ):
                    break # Stop generation

        except Exception as e:
            # Log error with step count if available
            step_info = f"step {count}" if "count" in locals() else "initial prompt processing"
            self.logger.error(
                f"Generation error during {step_info}: {e}", exc_info=True
            )
            # Stop the generator by returning (yields no more tokens)
            return

    def _sample_token(self, logits: mx.array, temp: float) -> Tuple[mx.array, mx.array]:
        """Samples the next token ID from logits and calculates its log probability."""
        # Logits shape: [batch, vocab_size] (batch is 1 in generate_step)

        if temp == 0:
            # Greedy sampling: choose the token with the highest logit
            token = mx.argmax(logits, axis=-1) # Shape [batch]
        else:
            # Temperature sampling: sample from distribution scaled by temperature
            token = mx.random.categorical(logits * (1 / temp)) # Shape [batch]

        # Calculate log probability of the *sampled* token
        # Use log_softmax for numerical stability
        log_probs_all = nn.log_softmax(logits, axis=-1) # Shape [batch, vocab_size]

        # Gather the log probability corresponding to the sampled token ID
        # Need to add dimension for take_along_axis: token shape [batch] -> [batch, 1]
        token_for_gather = token[:, None]
        # log_probs_all shape [batch, vocab_size], token_for_gather shape [batch, 1]
        log_prob = mx.take_along_axis(log_probs_all, token_for_gather, axis=-1).squeeze(-1)
        # Resulting log_prob shape should be [batch]

        return token, log_prob # Return token ID [batch] and its log prob [batch]

    def generate_action(
        self, prompt_ids: mx.array, temp: float = 0.7
    ) -> Tuple[str, mx.array, mx.array, mx.array]:
        """
        Generates a sequence (action) based on the prompt, calculates its total log probability,
        and gets the value estimate for the initial prompt state.
        Args:
            prompt_ids (mx.array): Token IDs for the prompt, shape [seq_len].
            temp (float): Sampling temperature.
        Returns:
            Tuple:
                - gen_text (str): The decoded generated text sequence.
                - sequence_log_prob (mx.array): Total log probability of the generated sequence (scalar).
                - value (mx.array): Critic's value estimate for the initial prompt state (scalar).
                - gen_ids_mx (mx.array): Token IDs of the generated sequence, shape [gen_len].
        """
        gen_tokens_list = [] # To store generated token IDs (mx.array of shape [1])
        log_probs_list = []  # To store log probs of generated tokens (mx.array of shape [1])

        try:
            # Use the step-by-step generator
            for token, log_prob in self.generate_step(prompt_ids, temp=temp):
                # token shape [1], log_prob shape [1]
                gen_tokens_list.append(token)
                log_probs_list.append(log_prob)

            if not gen_tokens_list:
                # Handle case where generate_step failed immediately or yielded nothing
                self.logger.warning("generate_step yielded no tokens.")
                gen_text = "[generation error - no tokens]"
                sequence_log_prob = mx.array(-1e9) # Penalize heavily
                value = mx.array(0.0) # Default value
                gen_ids_mx = mx.array([], dtype=mx.int32) # Empty generated IDs
                # Still try to get value for the prompt, as it might be valid
                try:
                    value = self.get_value(prompt_ids)
                except Exception as val_err:
                    self.logger.error(f"Failed to get value even after generation error: {val_err}")
                    value = mx.array(0.0) # Keep default if value fails too
                return gen_text, sequence_log_prob.reshape(()), value.reshape(()), gen_ids_mx

            # Concatenate generated tokens and log probabilities into single tensors
            gen_ids_mx = mx.concatenate(gen_tokens_list, axis=0) # Resulting shape [gen_len]
            log_probs_mx = mx.concatenate(log_probs_list, axis=0) # Resulting shape [gen_len]

            # Sum log probabilities for the entire sequence
            sequence_log_prob = mx.sum(log_probs_mx)
            mx.eval(sequence_log_prob) # Evaluate the sum

            # Decode the generated sequence IDs to text
            try:
                gen_text = self.tokenizer.decode(gen_ids_mx.tolist())
            except Exception as dec_e:
                self.logger.error(
                    f"Decoding failed for token IDs {gen_ids_mx.tolist()}: {dec_e}"
                )
                gen_text = "[decode error]"

            # Get value estimate for the initial prompt state
            value = self.get_value(prompt_ids) # Should return scalar value for this prompt

        except Exception as e:
            self.logger.error(
                f"generate_action failed unexpectedly: {e}", exc_info=True
            )
            gen_text = "[action generation error]"
            sequence_log_prob = mx.array(-1e9) # Penalize
            value = mx.array(0.0) # Default value
            gen_ids_mx = mx.array([], dtype=mx.int32) # Empty generated IDs

        # Ensure value and log_prob are scalar shapes before returning
        if value.size != 1:
            self.logger.warning(f"Value size is {value.size}, expected 1. Taking mean.")
            value = mx.mean(value)
        if sequence_log_prob.size != 1:
             self.logger.warning(f"Sequence log prob size is {sequence_log_prob.size}, expected 1.")
             # Taking mean might not make sense here, maybe keep as is or use first element?
             # Let's ensure shape is scalar for consistency with buffer add.
             sequence_log_prob = sequence_log_prob.reshape(())


        return gen_text, sequence_log_prob.reshape(()), value.reshape(()), gen_ids_mx

    def loss(
        self, model_params: dict, critic_params: dict, batch_data: Dict[str, mx.array]
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Computes the PPO loss (policy, value, entropy) for a batch of experiences.
        Args:
            model_params (dict): Flat dictionary of actor model parameters.
            critic_params (dict): Flat dictionary of critic model parameters.
            batch_data (Dict[str, mx.array]): Dictionary containing batch tensors:
                - "prompts": Padded prompt IDs [batch_size, prompt_len]
                - "gen_ids": Padded generated IDs [batch_size, gen_len]
                - "old_log_probs": Sequence log probs from rollout [batch_size]
                - "advantages": Calculated GAE advantages [batch_size, 1]
                - "returns": Calculated returns (targets for value function) [batch_size, 1]
        Returns:
            Tuple[mx.array, mx.array, mx.array, mx.array]:
                - total_loss: Combined PPO loss.
                - policy_loss: Clipped surrogate objective loss.
                - value_loss: Mean squared error loss for the value function.
                - approx_kl: Approximate KL divergence between old and new policies.
        """
        prompts = batch_data["prompts"]          # Shape: [batch_size, prompt_len]
        gen_ids = batch_data["gen_ids"]          # Shape: [batch_size, gen_len]
        old_logp_seq = batch_data["old_log_probs"] # Shape: [batch_size]
        advantages = batch_data["advantages"]    # Shape: [batch_size, 1]
        returns = batch_data["returns"]          # Shape: [batch_size, 1]

        # Ensure advantages and returns are squeezed to [batch_size] for loss calculations
        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)

        # --- Create temporary models with updated parameters for gradient calculation ---
        # This is standard practice in MLX for functional gradient computation.
        # We update temporary instances of the models with the parameters passed in.
        temp_actor = self.actor   # Assumes self.actor is the full nn.Module instance
        temp_critic = self.critic
        # Update the temporary models with the potentially modified parameters from the optimizer step
        temp_actor.update(tree_unflatten(list(model_params.items())))
        temp_critic.update(tree_unflatten(list(critic_params.items())))

        # --- Policy Loss Calculation ---
        # Concatenate prompt and generated IDs for a single forward pass
        combined_ids = mx.concatenate([prompts, gen_ids], axis=1) # Shape: [batch, prompt_len + gen_len]

        # Forward pass through the *updated* actor model to get new logits
        # No KV cache needed here as we process the full sequence at once.
        actor_output = temp_actor(combined_ids, cache=None)
        if isinstance(actor_output, tuple) and len(actor_output) >= 1:
            logits = actor_output[0] # Assume first output is logits
        elif isinstance(actor_output, mx.array):
            logits = actor_output # Assume only logits returned
        else:
            raise ValueError(
                f"Unexpected actor output type in loss function: {type(actor_output)}"
            )
        # Logits shape: [batch_size, combined_len, vocab_size]

        # Extract logits corresponding *only* to the generated tokens (actions)
        prompt_len = prompts.shape[1]
        # Logits from position prompt_len-1 up to the second-to-last position predict
        # the tokens from prompt_len to the end (i.e., the gen_ids).
        action_logits = logits[:, prompt_len - 1 : -1, :]
        gen_len = gen_ids.shape[1]

        # Validate shapes - crucial if padding/truncation caused inconsistencies
        if action_logits.shape[1] != gen_len:
            # This can happen if max_gen_len was reached differently across batch items,
            # leading to different actual generation lengths before padding.
            # A robust solution involves using attention masks, but here we'll try to slice
            # to the minimum length if a mismatch occurs.
            self.logger.warning(
                 f"Logits shape mismatch in loss: Action logits seq length ({action_logits.shape[1]}) "
                 f"!= Generated IDs seq length ({gen_len}). Truncating to minimum."
            )
            min_len = min(action_logits.shape[1], gen_len)
            action_logits = action_logits[:, :min_len, :]
            gen_ids_for_loss = gen_ids[:, :min_len]
        else:
             gen_ids_for_loss = gen_ids


        # Calculate log probabilities of the generated actions under the *new* policy
        log_probs_all_actions = nn.log_softmax(action_logits, axis=-1) # Shape: [batch, gen_len_used, vocab_size]

        # Gather the log probabilities corresponding to the actual generated tokens
        new_logp_tokens = mx.take_along_axis(
            log_probs_all_actions, gen_ids_for_loss[..., None], axis=-1 # Add trailing dim for gather
        ).squeeze(-1) # Shape: [batch_size, gen_len_used]

        # Sum log probabilities across the sequence length dimension to get sequence log prob
        # Apply masking here if padding was used and needs to be ignored in sum?
        # Assuming for now that padding tokens have negligible probability or loss handles it.
        # TODO: Implement masking if padding significantly affects loss.
        new_logp_seq = mx.sum(new_logp_tokens, axis=1) # Shape: [batch_size]

        # Calculate the PPO clipped surrogate objective
        log_ratio = new_logp_seq - old_logp_seq # Log prob ratio log(pi_new / pi_old)
        ratio = mx.exp(log_ratio)               # Prob ratio pi_new / pi_old

        # Unclipped objective term
        pg_loss1 = advantages * ratio
        # Clipped objective term
        pg_loss2 = advantages * mx.clip(
            ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        )
        # Policy loss is the negative mean of the minimum of the two terms
        policy_loss = -mx.mean(mx.minimum(pg_loss1, pg_loss2))

        # --- Value Loss Calculation ---
        # Get value prediction from the *updated* critic for the prompt state
        # We need hidden states from the prompt part of the sequence.
        # Re-use the logic from get_value, but with the temporary actor.
        try:
            if hasattr(temp_actor, "model") and callable(temp_actor.model):
                prompt_output = temp_actor.model(prompts, cache=None)
                if isinstance(prompt_output, tuple):
                    prompt_hiddens = prompt_output[0]
                else:
                    prompt_hiddens = prompt_output
            else:
                # Fallback if actor doesn't have separate .model
                self.logger.debug("Using full actor pass for critic input in loss value calc.")
                prompt_output = temp_actor(prompts, cache=None)
                if isinstance(prompt_output, tuple):
                    prompt_hiddens = prompt_output[0]
                else:
                    prompt_hiddens = prompt_output

            # Get current value estimates from the temporary critic
            current_values = temp_critic(prompt_hiddens).squeeze(-1) # Shape: [batch_size]

        except Exception as e:
             self.logger.error(f"Failed to get hidden states or value in loss function: {e}")
             # Fallback: use zeros, loss will be high but avoids crash
             current_values = mx.zeros_like(returns)


        # Value loss: Mean Squared Error between predicted values and calculated returns
        value_loss = mx.mean((current_values - returns) ** 2) * 0.5

        # --- Entropy Loss Calculation ---
        # Encourages exploration by penalizing low-entropy (peaked) distributions.
        # Use the action log probabilities calculated earlier.
        # Entropy = - sum(p * log(p))
        token_probs = mx.exp(log_probs_all_actions) # Convert log probs to probs
        # Calculate entropy per token: sum over vocab dim
        entropy_per_token = -mx.sum(token_probs * log_probs_all_actions, axis=-1) # Shape: [batch, gen_len_used]
        # Average entropy over batch and sequence length
        # TODO: Apply masking here if padding tokens should be excluded from entropy calculation.
        mean_entropy = mx.mean(entropy_per_token)
        # Entropy loss term (negative entropy, scaled by coefficient)
        entropy_loss = -self.entropy_coef * mean_entropy

        # --- Total Loss ---
        # Combine the losses
        total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss

        # --- Approximate KL Divergence (for monitoring) ---
        # Provides an estimate of how much the policy changed during the update.
        # Calculated using the log ratio from the policy loss calculation.
        # Note: old_logp_seq is treated as detached (constant) here.
        approx_kl = mx.mean(log_ratio) # Simpler KL approximation: E[log(pi_new / pi_old)]
        # Alternative KL approx: approx_kl = mx.mean(ratio - 1 - log_ratio) * 0.5

        return total_loss, policy_loss, value_loss, approx_kl


def _rows(a: mx.array, idx: Union[np.ndarray, List[int]]) -> mx.array:
    """
    Helper function to gather rows from an mx.array `a` using NumPy-style integer indices `idx`.
    This is a workaround for potential limitations in MLX's fancy indexing capabilities,
    ensuring compatibility across versions.
    Args:
        a (mx.array): The array to gather rows from (e.g., shape [N, ...]).
        idx (Union[np.ndarray, List[int]]): A 1D NumPy array or list of row indices.
    Returns:
        mx.array: An array containing the selected rows (e.g., shape [len(idx), ...]).
    """
    # Ensure indices are an mx.array of integers for mx.take
    if isinstance(idx, list):
        idx_mx = mx.array(idx, dtype=mx.int32)
    elif isinstance(idx, np.ndarray):
         idx_mx = mx.array(idx.astype(np.int32)) # Convert numpy array to mx array
    else:
         # Should not happen with get_batch_indices, but handle defensively
         raise TypeError(f"Unsupported index type for _rows: {type(idx)}")

    return mx.take(a, idx_mx, axis=0)


# --- Metrics Logger ---
class MetricsLogger:
    """Logs training metrics dictionary to a CSV file."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._file = None       # File handle
        self._writer = None     # csv.DictWriter instance
        self._headers: Optional[List[str]] = None # Store current headers to detect changes
        self.logger = logging.getLogger(__name__)

        # Ensure output directory exists
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Metrics will be logged to: {self.file_path}")
        except OSError as e:
            self.logger.error(f"Could not create directory for metrics file {self.file_path.parent}: {e}")
            # Allow initialization, but logging will fail later

    def log(self, metrics: Dict[str, Any]):
        """Logs a dictionary of metrics to the CSV file."""
        if self._file is None and self._writer is None and not self.file_path.parent.exists():
             self.logger.error("Cannot log metrics, output directory does not exist.")
             return

        # Convert mx.array metrics to native Python types (float/int/str) for CSV logging
        loggable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, mx.array):
                if v.size == 1:
                    try:
                        loggable_metrics[k] = v.item() # Convert scalar array to Python number
                    except Exception as e:
                        self.logger.debug(f"Could not convert metric '{k}' (size 1) to scalar: {e}")
                        loggable_metrics[k] = str(v) # Log as string if conversion fails
                else:
                    # Log multi-element arrays as string representation of list
                    loggable_metrics[k] = str(v.tolist())
            elif isinstance(v, (int, float, str, bool)):
                loggable_metrics[k] = v # Keep standard types as is
            elif v is None:
                loggable_metrics[k] = "" # Represent None as empty string in CSV
            else:
                loggable_metrics[k] = str(v) # Convert other types to string

        if not loggable_metrics:
            self.logger.debug("No loggable metrics provided to MetricsLogger.")
            return

        # Determine headers based on the current metrics dictionary keys
        current_headers = sorted(loggable_metrics.keys())

        try:
            # Open/reopen file if:
            # 1. It's not open yet.
            # 2. The headers have changed since the last write.
            if self._writer is None or self._headers != current_headers:
                if self._file: # Close existing file if headers changed
                    self._file.close()

                # Check if the file is new or empty to decide whether to write header
                is_new_file = not self.file_path.exists() or self.file_path.stat().st_size == 0

                # Open in append mode ('a')
                self._file = open(self.file_path, "a", newline="", encoding="utf-8")
                self._headers = current_headers # Update stored headers
                self._writer = csv.DictWriter(self._file, fieldnames=self._headers)

                if is_new_file:
                    self.logger.info(f"Writing CSV header to new file: {self._headers}")
                    self._writer.writeheader()
                elif self._headers != current_headers:
                    # Log if headers changed mid-run (might make CSV harder to parse)
                    self.logger.warning(
                        f"CSV headers changed mid-training. New headers: {self._headers}. "
                        f"File: {self.file_path}"
                    )
                    # Consider adding a comment line or separator in the CSV? For now, just append.

            # Write the data row, ensuring only known headers are included
            # Use default value "" for any missing keys (shouldn't happen if headers match)
            row_to_write = {k: loggable_metrics.get(k, "") for k in self._headers}
            self._writer.writerow(row_to_write)
            self._file.flush() # Ensure data is written to disk immediately

        except IOError as e:
            self.logger.error(
                f"Failed to open or write to metrics file {self.file_path}: {e}"
            )
            # Reset state on error to potentially allow recovery on next call
            self._writer = None
            self._headers = None
            if self._file:
                try:
                    self._file.close()
                except Exception: pass
                self._file = None
        except Exception as e:
            self.logger.error(f"Failed to write metrics row: {e}", exc_info=True)


    def close(self):
        """Closes the metrics log file if it's open."""
        if self._file is not None:
            try:
                self._file.close()
                self.logger.info(f"Closed metrics file: {self.file_path}")
            except Exception as e:
                self.logger.error(f"Error closing metrics file {self.file_path}: {e}")
            finally:
                # Reset state regardless of close success/failure
                self._file = None
                self._writer = None
                self._headers = None


# --- Evaluation Function ---
def evaluate(
    agent: PPOAgent,
    eval_env: LLMEnv,
    eval_iters: int, # Number of samples to evaluate on
    progress: Progress,
    task_id: TaskID,
) -> Dict[str, float]:
    """Evaluates the PPO agent on the evaluation environment."""
    logger = logging.getLogger(__name__)
    logger.info(
        f"Starting evaluation on '{eval_env.env_id}' for up to {eval_iters} samples..."
    )
    total_reward = 0.0
    total_samples_processed = 0
    generated_samples_log = [] # Store first few samples for qualitative check

    # Ensure eval_iters doesn't exceed actual dataset size
    max_iters = min(eval_iters, len(eval_env))
    if max_iters <= 0:
        logger.warning(f"Evaluation dataset '{eval_env.env_id}' is empty or eval_iters is zero. Skipping evaluation.")
        return {}

    # Update progress bar for evaluation phase
    progress.update(
        task_id,
        total=max_iters,
        completed=0,
        description=f"Evaluating ({eval_env.env_id})",
    )

    # Loop through the evaluation dataset
    for i in range(max_iters):
        if shutdown_requested:
            logger.warning("Evaluation interrupted by shutdown signal.")
            break

        # Get a sample from the environment
        reset_result = eval_env.reset()
        if reset_result is None:
            logger.error(
                f"Evaluation environment '{eval_env.env_id}' failed to reset at iteration {i}. Stopping evaluation."
            )
            break # Stop evaluation if reset fails critically

        current_prompt_ids, prompt_text, ref_completion = reset_result

        # Generate action greedily (temp=0.0) for deterministic evaluation
        try:
            # We only need the generated text for evaluation reward
            action_text, _, _, _ = agent.generate_action(current_prompt_ids, temp=0.0)
        except Exception as e:
            logger.error(
                f"Evaluation generate_action failed for prompt '{prompt_text[:50]}...': {e}",
                exc_info=False, # Less verbose traceback for eval errors
            )
            action_text = "[generation error]" # Mark as error for logging

        # Step the environment with the generated action to get reward
        step_result = eval_env.step(action_text)

        if step_result is None:
            # This might happen if reward calculation itself fails
            logger.warning(f"Evaluation step failed for sample {i+1} in '{eval_env.env_id}'. Skipping sample.")
        else:
            reward, reward_metrics = step_result
            total_reward += reward
            total_samples_processed += 1

            # Log first few samples (e.g., first 5) for qualitative inspection
            if i < 5:
                generated_samples_log.append(
                    {
                        "prompt": prompt_text[:100] + "...", # Log snippet
                        "generated": action_text[:200] + "...", # Log snippet
                        "reference": ref_completion[:200] + "..." if ref_completion else "N/A",
                        "reward": f"{reward:.3f}",
                    }
                )
        # Update evaluation progress bar
        progress.update(task_id, advance=1)

    # --- Log Evaluation Results ---
    mean_reward = total_reward / total_samples_processed if total_samples_processed > 0 else 0.0
    logger.info(
        f"Evaluation complete on '{eval_env.env_id}'. Mean reward: {mean_reward:.4f} "
        f"({total_samples_processed}/{max_iters} samples evaluated)."
    )

    # Print sample outputs table if any were logged
    if generated_samples_log:
        logger.info("--- Sample Generated Outputs (Evaluation) ---")
        log_table = Table(
            title=f"Eval Samples ({eval_env.env_id})",
            show_header=True,
            header_style="bold magenta",
            box=None, padding=(0,1)
        )
        log_table.add_column("Prompt Snippet", style="cyan")
        log_table.add_column("Generated Snippet", style="white")
        log_table.add_column("Reference Snippet", style="dim")
        log_table.add_column("Reward", style="green")
        for s in generated_samples_log:
            log_table.add_row(s["prompt"], s["generated"], s["reference"], s["reward"])
        console.print(log_table)
        logger.info("-------------------------------------------")

    # Return summary metrics
    return {"eval_mean_reward": mean_reward, "eval_samples_processed": float(total_samples_processed)}


# --- Training Orchestration ---
def train(
    args: TrainingArgs,
    model: nn.Module,           # Actor model (LLM)
    model_config: dict,         # Loaded model config dictionary
    critic: CriticNetwork,      # Critic model
    tokenizer: TokenizerWrapper,
    train_env: LLMEnv,
    val_env: Optional[LLMEnv],  # Optional validation environment
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    reward_fn: SimpleRewardFunction, # Instance of the reward function
):
    """Main PPO training loop."""
    global shutdown_requested
    logger = logging.getLogger(__name__)

    # Initialize PPO Agent
    agent = PPOAgent(
        model,
        critic,
        tokenizer,
        actor_optimizer,
        critic_optimizer,
        args.gamma,
        args.gae_lambda,
        args.clip_epsilon,
        args.value_loss_coef,
        args.entropy_coef,
        args.max_gen_len,
    )

    # Define the function to compute loss and gradients for both actor and critic
    # argnums=(0, 1) specifies that we want gradients w.r.t. model_params (0) and critic_params (1)
    loss_and_grad_fn = mx.value_and_grad(agent.loss, argnums=(0, 1))

    # Get initial model parameters as flat dictionaries (required by optimizers)
    actor_params = model.parameters()
    critic_params = critic.parameters()
    mx.eval(actor_params, critic_params) # Evaluate initial params

    # Optimizer states are implicitly managed by the optimizer instances

    # Rollout buffer to store experiences
    rollout_buffer = RolloutBuffer()

    # Training loop variables
    start_time = time.monotonic()
    global_step = 0 # Total environment steps taken
    num_updates = 0 # Number of PPO update phases completed
    last_save_step = 0
    last_eval_step = 0
    all_rollout_rewards = [] # Track rewards over time

    # Prepare output directory and metrics logger
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_logger = MetricsLogger(output_dir / "training_metrics.csv")
    logger.info(f"Output directory: {output_dir}")
    # Save config only if not resuming (assuming config is saved in checkpoint)
    if not args.resume_from_checkpoint:
        logger.info(f"Saving training configuration to {output_dir / 'training_args.json'}")
        try:
            with open(output_dir / "training_args.json", "w") as f:
                json.dump(asdict(args), f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save training args: {e}")


    # Initial environment reset to get the first prompt
    reset_result = train_env.reset()
    if reset_result is None:
        logger.critical(
            "Failed initial training environment reset. Cannot start training."
        )
        return # Exit if environment fails at the start
    current_prompt_ids, current_prompt_text, current_ref_completion = reset_result

    logger.info(f"Starting PPO training for {args.total_timesteps} timesteps...")

    # Setup Rich Progress Bar for the main training loop
    progress_cols = (
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        MofNCompleteColumn(), # Shows steps/total_timesteps
        TextColumn("• Upd: {task.fields[update]}"),
        TextColumn("• RollRew: {task.fields[roll_rew]:.2f}"), # Mean reward in last rollout
        TextColumn("• KL: {task.fields[kl]:.4f}"),          # Approx KL from last update
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    try:
        with Progress(*progress_cols, console=console, transient=False) as progress:
            main_task = progress.add_task(
                "Training",
                total=args.total_timesteps,
                update=0,
                roll_rew=math.nan, # Initialize metrics
                kl=math.nan,
            )

            # --- Main Training Loop ---
            while global_step < args.total_timesteps and not shutdown_requested:
                rollout_start_time = time.monotonic()
                rollout_buffer.clear() # Clear buffer for new rollout
                collected_rewards_in_rollout = [] # Track rewards for this specific rollout

                # --- Rollout Phase: Collect Experience ---
                # Loop for `num_rollout_steps` or until total timesteps are reached
                for rollout_step_idx in range(args.num_rollout_steps):
                    if shutdown_requested or global_step >= args.total_timesteps:
                        break # Exit inner loop if shutdown or target steps reached

                    # Basic check for valid prompt from previous reset/step
                    if current_prompt_ids is None or current_prompt_ids.size == 0:
                        logger.warning(
                            f"Invalid prompt_ids encountered at start of rollout step {global_step}. Resetting environment."
                        )
                        reset_result = train_env.reset()
                        if reset_result is None:
                            logger.critical("Training environment failed permanently during rollout. Stopping.")
                            shutdown_requested = True # Trigger shutdown
                            break
                        current_prompt_ids, current_prompt_text, current_ref_completion = reset_result
                        continue # Skip this step and try again with the new prompt

                    # Generate action (sequence) and get value estimate from agent
                    try:
                        action_text, seq_log_prob, value, gen_ids = agent.generate_action(
                            current_prompt_ids, temp=0.7 # Use temperature sampling during rollout
                        )
                        # Ensure results are computed before proceeding
                        mx.eval(seq_log_prob, value, gen_ids)
                    except Exception as gen_err:
                        logger.error(
                            f"Rollout generate_action error at step {global_step}: {gen_err}",
                            exc_info=False, # Less verbose for generation errors
                        )
                        # Attempt to recover by resetting the environment
                        reset_result = train_env.reset()
                        if reset_result is None:
                            logger.critical("Training environment failed permanently after generation error. Stopping.")
                            shutdown_requested = True
                            break
                        current_prompt_ids, current_prompt_text, current_ref_completion = reset_result
                        continue # Skip adding this faulty step to buffer

                    # Step the environment with the generated action to get reward
                    terminated = True # Assume each generation completes an "episode" for PPO purposes
                    step_result = train_env.step(action_text)

                    if step_result is not None:
                        reward, reward_metrics = step_result
                        collected_rewards_in_rollout.append(reward)

                        # Add experience to the rollout buffer
                        rollout_buffer.add(
                            current_prompt_ids,
                            gen_ids,
                            action_text,
                            seq_log_prob, # Scalar array
                            reward,
                            terminated,
                            value,        # Scalar array
                        )

                        # Log generated samples periodically for qualitative checks
                        if global_step % args.generate_samples_every == 0 and global_step > 0:
                            logger.info(
                                f"\n--- Sample @ Step {global_step} ---\n"
                                f"Prompt: {current_prompt_text[:200]}...\n"
                                f"Gen: {action_text[:500]}...\n"
                                f"Reward: {reward:.3f} | Value: {value.item():.3f} | LogProb: {seq_log_prob.item():.3f}\n"
                                f"--------------------------"
                            )
                    else:
                        # Environment step failed (e.g., reward calculation error)
                        logger.warning(
                            f"Environment step failed at global_step {global_step}. Not adding step to buffer."
                        )

                    # Reset environment for the next step in the rollout
                    reset_result = train_env.reset()
                    if reset_result is None:
                        logger.critical("Training environment failed permanently during reset. Stopping.")
                        shutdown_requested = True
                        break
                    current_prompt_ids, current_prompt_text, current_ref_completion = reset_result

                    # Increment global step count and update progress bar
                    global_step += 1
                    progress.update(main_task, advance=1)

                # --- End of Rollout Phase ---
                if shutdown_requested: break # Exit main loop if shutdown requested

                rollout_duration = time.monotonic() - rollout_start_time
                steps_collected = len(rollout_buffer.prompt_ids)
                mean_rollout_reward = (
                    np.mean(collected_rewards_in_rollout)
                    if collected_rewards_in_rollout
                    else math.nan
                )
                all_rollout_rewards.append(mean_rollout_reward) # Store mean reward for overall tracking

                # Update progress bar with rollout reward
                progress.update(main_task, roll_rew=mean_rollout_reward)

                logger.info(
                    f"[Rollout {num_updates+1}] Collected {steps_collected}/{args.num_rollout_steps} steps in {rollout_duration:.2f}s. "
                    f"Mean Reward: {mean_rollout_reward:.3f}"
                )

                # Skip update phase if no valid steps were collected
                if steps_collected == 0:
                    logger.warning("Rollout buffer is empty after collection phase. Skipping PPO update.")
                    continue

                # --- Compute Advantages and Returns ---
                try:
                    # Get value estimate for the state *after* the last rollout step
                    if current_prompt_ids is not None and current_prompt_ids.size > 0:
                        # Use stop_gradient? Not strictly necessary as it's not part of grad calc path here
                        # with mx.stop_gradient():
                        last_value = agent.get_value(current_prompt_ids)
                        mx.eval(last_value)
                    else:
                        # Handle case where the last reset failed somehow
                        logger.warning("Last prompt_ids after rollout were invalid. Using zero value for GAE.")
                        last_value = mx.array(0.0)

                    # Compute GAE advantages and returns, storing them in the buffer
                    rollout_buffer.compute_advantages_and_returns(
                        last_value, args.gamma, args.gae_lambda
                    )
                    # Check if calculation succeeded
                    if rollout_buffer.advantages is None or rollout_buffer.returns is None:
                        raise ValueError("Advantages or returns calculation failed (returned None).")
                    # Ensure calculations are complete
                    mx.eval(rollout_buffer.advantages, rollout_buffer.returns)

                except Exception as gae_e:
                    logger.error(f"GAE calculation failed: {gae_e}", exc_info=True)
                    continue # Skip update if GAE fails

                # --- PPO Update Phase ---
                update_start_time = time.monotonic()
                total_policy_loss, total_value_loss, total_approx_kl = 0.0, 0.0, 0.0
                num_batches_processed = 0
                actual_ppo_epochs_run = 0 # Track how many epochs actually ran

                # Prepare data from buffer for batching
                try:
                    # Pad sequences to the same length within the buffer for batching
                    prompt_ids_padded = rollout_buffer._pad_and_stack(
                        rollout_buffer.prompt_ids, tokenizer.pad_token_id or 0
                    )
                    gen_ids_padded = rollout_buffer._pad_and_stack(
                        rollout_buffer.generated_ids, tokenizer.pad_token_id or 0
                    )
                    # Stack scalar arrays into batches
                    old_log_probs_stacked = mx.stack(rollout_buffer.log_probs, axis=0) # Shape [N]
                    advantages_stacked = rollout_buffer.advantages # Shape [N, 1]
                    returns_stacked = rollout_buffer.returns       # Shape [N, 1]

                    # Materialize all stacked data before slicing in the loop
                    mx.eval(
                        prompt_ids_padded,
                        gen_ids_padded,
                        old_log_probs_stacked,
                        advantages_stacked,
                        returns_stacked,
                    )
                    num_samples_in_buffer = prompt_ids_padded.shape[0]

                except Exception as buffer_prep_e:
                    logger.error(f"Error preparing buffer data for update: {buffer_prep_e}", exc_info=True)
                    continue # Skip update if data prep fails

                if num_samples_in_buffer == 0:
                    logger.warning("Buffer empty after padding/stacking. Skipping update.")
                    continue

                # --- PPO Epoch Loop ---
                for ppo_epoch in range(args.ppo_epochs):
                    actual_ppo_epochs_run += 1
                    if shutdown_requested: break # Allow interruption between epochs
                    epoch_kls = [] # Track KL divergence within this epoch

                    # --- Mini-batch Loop ---
                    # Iterate over random mini-batches from the buffer
                    for batch_indices in rollout_buffer.get_batch_indices(args.ppo_batch_size): # Use args.ppo_batch_size
                        if shutdown_requested: break # Allow interruption between batches
                        if len(batch_indices) == 0: continue # Skip empty batch

                        # Use helper function to gather rows based on indices
                        idx = batch_indices # NumPy array of indices
                        batch_data = {
                            "prompts":       _rows(prompt_ids_padded, idx),
                            "gen_ids":       _rows(gen_ids_padded, idx),
                            "old_log_probs": _rows(old_log_probs_stacked, idx),
                            "advantages":    _rows(advantages_stacked, idx),
                            "returns":       _rows(returns_stacked, idx),
                            # "values_old" removed as it wasn't used in loss
                        }

                        try:
                            # Calculate loss and gradients for the current batch
                            # Pass flat parameter dictionaries to the loss function
                            (loss, p_loss, v_loss, approx_kl), (actor_grads, critic_grads,) = loss_and_grad_fn(
                                dict(tree_flatten(actor_params)), # Pass current actor params
                                dict(tree_flatten(critic_params)),# Pass current critic params
                                batch_data,
                            )

                            # Apply gradients using the optimizers
                            # This updates the parameter dictionaries (actor_params, critic_params) in-place
                            # and also updates the optimizer's internal state.
                            actor_optimizer.apply_gradients(actor_grads, actor_params)
                            critic_optimizer.apply_gradients(critic_grads, critic_params)

                            # Evaluate updated parameters and loss metrics
                            mx.eval(
                                actor_params, critic_params, # Ensure param updates are done
                                loss, p_loss, v_loss, approx_kl # Ensure metrics are computed
                            )

                            # Accumulate metrics for logging
                            total_policy_loss += p_loss.item()
                            total_value_loss += v_loss.item()
                            kl_val = approx_kl.item()
                            total_approx_kl += kl_val
                            epoch_kls.append(kl_val)
                            num_batches_processed += 1

                        except Exception as batch_err:
                            logger.error(
                                f"PPO Batch Error (Update {num_updates+1}, Epoch {ppo_epoch+1}, "
                                f"Batch {num_batches_processed % (num_samples_in_buffer // args.ppo_batch_size + 1)}): {batch_err}",
                                exc_info=True,
                            )
                            # Optionally: break inner loop or continue to next batch?
                            # Continue for now to maximize data usage.

                    # --- End of Mini-batch Loop ---
                    if shutdown_requested: break

                    # --- Early Stopping Check (based on KL divergence) ---
                    mean_epoch_kl = np.mean(epoch_kls) if epoch_kls else 0
                    # Check if KL target is set (not None) and exceeded
                    if args.target_kl is not None and mean_epoch_kl > args.target_kl:
                        logger.warning(
                            f"KL divergence {mean_epoch_kl:.4f} exceeded target {args.target_kl}. "
                            f"Stopping PPO epochs early at epoch {ppo_epoch+1}."
                        )
                        break # Stop PPO epochs for this update phase

                # --- End of PPO Epoch Loop ---
                if shutdown_requested: break # Exit main loop

                # --- Post-Update Logging & Cleanup ---
                update_duration = time.monotonic() - update_start_time
                num_updates += 1 # Increment update counter

                # Calculate average metrics for this update phase
                avg_p_loss = total_policy_loss / num_batches_processed if num_batches_processed else math.nan
                avg_v_loss = total_value_loss / num_batches_processed if num_batches_processed else math.nan
                avg_kl = total_approx_kl / num_batches_processed if num_batches_processed else math.nan

                # Update progress bar with KL divergence
                progress.update(main_task, update=num_updates, kl=avg_kl)
                logger.info(
                    f"[Update {num_updates}] Duration: {update_duration:.2f}s | "
                    f"Avg Loss P: {avg_p_loss:.4f} | Avg Loss V: {avg_v_loss:.4f} | Avg KL: {avg_kl:.4f} "
                    f"({actual_ppo_epochs_run}/{args.ppo_epochs} epochs run)"
                )

                # Log metrics to CSV file
                if num_batches_processed > 0:
                    metrics_logger.log(
                        {
                            "global_step": global_step,
                            "update": num_updates,
                            "mean_rollout_reward": mean_rollout_reward,
                            "policy_loss": avg_p_loss,
                            "value_loss": avg_v_loss,
                            "approx_kl": avg_kl,
                            "rollout_steps_collected": steps_collected,
                            "rollout_duration_sec": rollout_duration,
                            "update_duration_sec": update_duration,
                            "ppo_epochs_run": actual_ppo_epochs_run,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

                # --- Evaluation Phase ---
                # Check if it's time to evaluate based on `eval_every` or if training is ending
                if val_env and (
                    global_step - last_eval_step >= args.eval_every
                    or global_step >= args.total_timesteps
                ):
                    eval_start_time = time.monotonic()
                    # Use a separate progress bar for evaluation
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                        console=console,
                        transient=True, # Hide after completion
                    ) as eval_prog:
                        eval_task = eval_prog.add_task(
                            f"Evaluating ({val_env.env_id})", total=len(val_env) # Evaluate on full val set
                        )
                        # Run evaluation function
                        eval_metrics = evaluate(
                            agent,
                            val_env,
                            eval_iters=len(val_env), # Number of samples to use
                            progress=eval_prog,
                            task_id=eval_task,
                        )
                    eval_duration = time.monotonic() - eval_start_time
                    logger.info(
                        f"--- Evaluation @ Step {global_step} ({eval_duration:.2f}s) ---"
                    )
                    if eval_metrics:
                        logger.info(f"Eval Metrics: {eval_metrics}")
                        # Log evaluation metrics along with current step info
                        metrics_logger.log({"global_step": global_step, **eval_metrics})
                    last_eval_step = global_step # Record step of last evaluation

                # --- Saving Phase ---
                # Check if it's time to save based on `save_every` or if training is ending
                if (
                    global_step - last_save_step >= args.save_every
                    or global_step >= args.total_timesteps
                ):
                    save_path = Path(output_dir / f"checkpoint_step_{global_step}")
                    # save_path.mkdir(parents=True, exist_ok=True) # Moved inside try block
                    try:
                        logger.info(f"Saving checkpoint to {save_path}...")
                        # Ensure latest parameters are computed before saving
                        mx.eval(actor_params, critic_params)

                        # --- Manual Saving Logic (Replaces save_weights) ---
                        save_path.mkdir(parents=True, exist_ok=True) # Create directory

                        # Save actor weights (parameters)
                        actor_weights_dict = dict(tree_flatten(actor_params))
                        mx.save_safetensors(str(save_path / "actor_weights.safetensors"), actor_weights_dict)

                        # Save actor model configuration
                        with open(save_path / "actor_config.json", "w") as f:
                             json.dump(model_config, f, indent=4)
                        # --- End Manual Actor Saving ---

                        # Save critic weights separately using safetensors
                        mx.save_safetensors(
                            str(save_path / "critic.safetensors"),
                            dict(tree_flatten(critic_params)), # Save as flat dict
                        )

                        # Save tokenizer config/files
                        # Use the underlying tokenizer object if wrapped
                        tok_to_save = getattr(tokenizer, "tokenizer", tokenizer)
                        if hasattr(tok_to_save, "save_pretrained"):
                            tok_to_save.save_pretrained(str(save_path))
                        else:
                             logger.warning(f"Tokenizer type {type(tok_to_save)} does not have save_pretrained method.")

                        # TODO: Consider saving optimizer states here if needed for full resume
                        # actor_opt_state = actor_optimizer.state
                        # critic_opt_state = critic_optimizer.state
                        # mx.savez(save_path / "optimizer_states.npz", actor=actor_opt_state, critic=critic_opt_state)
                        # TODO: Consider saving training state (global_step, num_updates)
                        # with open(save_path / "training_state.json", "w") as f:
                        #     json.dump({"global_step": global_step, "num_updates": num_updates}, f)

                        logger.info(
                            f"Checkpoint saved successfully at step {global_step}."
                        )
                        last_save_step = global_step # Record step of last save
                    except Exception as e:
                        logger.error(
                            f"Failed to save checkpoint at step {global_step}: {e}", exc_info=True
                        )

                # --- Memory Cleanup ---
                # Optional: Clear buffer and collect garbage periodically
                rollout_buffer.clear()
                # Trigger GC less frequently, e.g., every 10 updates
                if num_updates % 10 == 0:
                    gc.collect()
                    logger.debug("Triggered garbage collection.")

            # --- End of Main Training Loop ---

    except Exception as train_err:
        logger.critical("Training loop encountered a critical error.", exc_info=True)
        # Ensure cleanup happens even if loop crashes
    finally:
        # --- Final Cleanup and Save ---
        train_duration = time.monotonic() - start_time
        status = "Interrupted" if shutdown_requested else "Completed"
        logger.info(
            f"Training {status}. Ran {num_updates} updates ({global_step} total steps) "
            f"in {train_duration / 3600:.2f} hours."
        )

        # Determine final save directory name
        final_dir_name = f"interrupted_step_{global_step}" if shutdown_requested else "final_model"
        final_dir = output_dir / final_dir_name
        # final_dir.mkdir(parents=True, exist_ok=True) # Moved inside try block
        logger.info(f"Saving final model state to {final_dir}...")

        try:
            # Ensure latest parameters are computed before saving
            mx.eval(actor_params, critic_params)

            # --- Manual Saving Logic (Replaces save_weights) ---
            final_dir.mkdir(parents=True, exist_ok=True) # Create directory

            # Save actor weights (parameters)
            actor_weights_dict = dict(tree_flatten(actor_params))
            mx.save_safetensors(str(final_dir / "actor_weights.safetensors"), actor_weights_dict)

            # Save actor model configuration
            with open(final_dir / "actor_config.json", "w") as f:
                 json.dump(model_config, f, indent=4)
            # --- End Manual Actor Saving ---

            # Save critic weights
            mx.save_safetensors(
                str(final_dir / "critic.safetensors"),
                dict(tree_flatten(critic_params)),
            )

            # Save tokenizer
            tok_to_save = getattr(tokenizer, "tokenizer", tokenizer)
            if hasattr(tok_to_save, "save_pretrained"):
                tok_to_save.save_pretrained(str(final_dir))

            # Save training arguments used
            with open(final_dir / "training_args.json", "w") as fp:
                json.dump(asdict(args), fp, indent=4)

            logger.info("Final model state saved successfully.")

        except Exception as final_save_err:
            logger.error(f"Failed during final model saving: {final_save_err}", exc_info=True)

        finally:
            # Close the metrics logger file handle
            metrics_logger.close()


# --- CLI Definition ---
@click.command()
# Define options using TrainingArgs fields + verbose flag
@click.option(
    "--train-dataset-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
    help="Path to the training JSONL file.",
)
@click.option(
    "--model-path",
    required=True,
    type=str, # Can be local path or HF repo ID
    help="Path or HuggingFace repo ID of the base MLX model (used if not resuming).",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(resolve_path=True),
    help="Directory to save checkpoints, logs, and final model.",
)
@click.option(
    "--val-dataset-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
    help="Path to the validation JSONL file (optional).",
)
# --- Resume Option ---
@click.option(
    "--resume-from-checkpoint",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True),
    help="Path to a checkpoint directory to resume training from.",
)
# --- End Resume Option ---
@click.option("--max-prompt-len", default=750, type=int, show_default=True, help="Maximum token length for prompts.")
@click.option("--max-gen-len", default=1024, type=int, show_default=True, help="Maximum token length for generated responses.")
@click.option("--actor-lr", default=5e-6, type=float, show_default=True, help="Learning rate for the actor (policy) network.")
@click.option("--critic-lr", default=1e-5, type=float, show_default=True, help="Learning rate for the critic (value) network.")
@click.option("--ppo-epochs", default=2, type=int, show_default=True, help="Number of optimization epochs per PPO update.")
@click.option("--num-rollout-steps", default=128, type=int, show_default=True, help="Number of environment steps (generations) to collect per rollout.")
@click.option("--ppo-batch-size", default=32, type=int, show_default=True, help="Mini-batch size for PPO updates.")
@click.option("--gamma", default=0.99, type=float, show_default=True, help="Discount factor for rewards.")
@click.option("--gae-lambda", default=0.95, type=float, show_default=True, help="Lambda factor for Generalized Advantage Estimation (GAE).")
@click.option("--clip-epsilon", default=0.2, type=float, show_default=True, help="Clipping parameter for PPO surrogate objective.")
@click.option("--value-loss-coef", default=0.5, type=float, show_default=True, help="Coefficient for the value loss term.")
@click.option("--entropy-coef", default=0.01, type=float, show_default=True, help="Coefficient for the entropy bonus term.")
@click.option("--target-kl", default=0.05, type=float, show_default=True, help="Target KL divergence for early stopping PPO epochs (set > 1.0 to disable).")
@click.option("--total-timesteps", default=100, type=int, show_default=True, help="Total number of environment steps (generations) to train for.")
@click.option("--save-every", default=2000, type=int, show_default=True, help="Save a checkpoint every N environment steps.")
@click.option("--eval-every", default=1000, type=int, show_default=True, help="Evaluate on validation set every N environment steps.")
@click.option("--log-every", default=50, type=int, show_default=True, help="Log basic info every N steps (mostly handled by progress bar now).") # Less critical now
@click.option("--generate-samples-every", default=500, type=int, show_default=True, help="Log generated samples during training every N environment steps.")
@click.option("--seed", default=42, type=int, show_default=True, help="Random seed for reproducibility.")
@click.option("--shuffle-data/--no-shuffle-data", default=True, is_flag=True, show_default=True, help="Shuffle training data each time the dataset is iterated.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable DEBUG logging level.")
def cli_main(**kwargs):
    """Fine-tunes a language model using PPO with MLX."""

    # --- Logging Setup ---
    log_level = logging.DEBUG if kwargs["verbose"] else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            RichHandler( # Use Rich for nicely formatted console output
                markup=True,
                rich_tracebacks=True,
                show_path=log_level == logging.DEBUG, # Show file paths only in debug
                log_time_format="[%X]", # Time format H:M:S
                console=console, # Use the global Rich console
            )
        ],
        force=True, # Override any root logger configuration
    )
    global logger # Ensure global logger is updated
    logger = logging.getLogger(__name__)
    # Reduce verbosity of underlying libraries
    logging.getLogger("mlx_lm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logger.info(f"Logging level set to: {logging.getLevelName(log_level)}")

    # --- Signal Handling ---
    signal.signal(signal.SIGINT, handle_signal)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal) # Handle termination signals
    logger.info("Registered signal handlers for SIGINT and SIGTERM.")

    # --- Parse Arguments into Dataclass ---
    # Filter kwargs to only include fields defined in TrainingArgs
    training_args_fields = {f.name for f in fields(TrainingArgs)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in training_args_fields}

    # Handle special case for target_kl (convert > 1.0 to None)
    if filtered_kwargs.get("target_kl", 0.05) > 1.0:
        filtered_kwargs["target_kl"] = None
        logger.info("Target KL check for early stopping is disabled (target_kl > 1.0).")

    try:
        # Instantiate TrainingArgs dataclass
        args = TrainingArgs(**filtered_kwargs)
    except TypeError as e:
        logger.critical(f"Error parsing training arguments: {e}", exc_info=True)
        sys.exit(1)

    # --- Display Configuration ---
    logger.info("[bold green]--- Training Configuration ---[/]")
    args_table = Table(show_header=False, box=None, padding=(0, 1))
    args_table.add_column("Parameter", style="cyan", justify="right")
    args_table.add_column("Value", style="white")
    for k, v in sorted(asdict(args).items()):
        args_table.add_row(k, str(v))
    console.print(args_table)
    logger.info("[bold green]----------------------------[/]")


    # --- Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    logger.info(f"Set random seed for Python, NumPy, and MLX: {args.seed}")

    # --- Load Model, Tokenizer, Critic ---
    # Initialize variables
    model = None
    tokenizer = None
    critic = None
    model_config = None
    hidden_size = None

    if args.resume_from_checkpoint:
        # --- Resume from Checkpoint ---
        logger.info(f"Attempting to resume training from checkpoint: [cyan]{args.resume_from_checkpoint}[/cyan]")
        checkpoint_path = Path(args.resume_from_checkpoint)
        if not checkpoint_path.is_dir():
            logger.critical(f"Checkpoint path is not a valid directory: {checkpoint_path}")
            sys.exit(1)

        # Define expected file paths
        actor_config_file = checkpoint_path / "actor_config.json"
        actor_weights_file = checkpoint_path / "actor_weights.safetensors"
        critic_weights_file = checkpoint_path / "critic.safetensors"

        # Check if required files exist
        required_files = [actor_config_file, actor_weights_file, critic_weights_file]
        if not all(f.exists() for f in required_files):
            logger.critical(f"Checkpoint directory missing required files: {required_files}")
            sys.exit(1)

        try:
            # 1. Load Actor Config
            logger.info(f"Loading actor config from {actor_config_file}...")
            with open(actor_config_file, "r") as f:
                model_config = json.load(f)
            logger.info("Actor config loaded.")

            # 2. Instantiate Actor Model from Config
            logger.info("Instantiating actor model from config...")
            model_type = model_config.get("model_type")
            if not model_type:
                raise ValueError("Model type not found in actor_config.json")

            # Use mlx_lm helper to get model classes based on type
            model_class, args_class = _get_classes(model_config)
            model_args = args_class.from_dict(model_config)
            model = model_class(model_args)
            logger.info(f"Actor model ({model_type}) instantiated.")

            # 3. Load Actor Weights
            logger.info(f"Loading actor weights from {actor_weights_file}...")
            actor_weights = mx.load(str(actor_weights_file))
            model.update(tree_unflatten(list(actor_weights.items())))
            mx.eval(model.parameters()) # Ensure weights are loaded
            logger.info("Actor weights loaded.")

            # 4. Load Tokenizer
            logger.info(f"Loading tokenizer from {checkpoint_path}...")
            tokenizer = load_tokenizer(checkpoint_path)
            logger.info("Tokenizer loaded.")

            # 5. Determine Hidden Size (needed for Critic)
            hidden_size = model_config.get("hidden_size")
            if not hidden_size:
                 # Attempt inference again if missing (shouldn't happen if saved correctly)
                 if hasattr(model, "args") and hasattr(model.args, "hidden_size"):
                     hidden_size = model.args.hidden_size
                 else:
                     raise ValueError("Could not determine hidden_size from loaded config/model.")

            # 6. Instantiate Critic
            logger.info(f"Instantiating critic network (hidden_size={hidden_size})...")
            critic = CriticNetwork(hidden_size=hidden_size)

            # 7. Load Critic Weights
            logger.info(f"Loading critic weights from {critic_weights_file}...")
            critic_weights = mx.load(str(critic_weights_file))
            critic.update(tree_unflatten(list(critic_weights.items())))
            mx.eval(critic.parameters()) # Ensure weights are loaded
            logger.info("Critic weights loaded.")

            # TODO: Load optimizer states and training state (global_step etc.) if saved

            logger.info(f"[bold green]Successfully resumed model, tokenizer, and critic from {checkpoint_path}[/]")

        except Exception as e:
            logger.critical(f"Failed to resume from checkpoint '{checkpoint_path}': {e}", exc_info=True)
            sys.exit(1)

    else:
        # --- Start Fresh ---
        logger.info(f"Loading base model and tokenizer from: [cyan]{args.model_path}[/cyan]...")
        try:
            # Use mlx_lm load utility for initial load
            model, tokenizer = load(args.model_path)
            logger.info(f"Loaded model type via mlx_lm.load(): {type(model)}")

            # Basic check for model structure
            if not isinstance(model, nn.Module) or not callable(model):
                logger.critical(f"Loaded object type {type(model)} is not a callable nn.Module.")
                sys.exit(1)

            # Load model config
            model_path_obj = get_model_path(args.model_path)
            model_config = load_config(model_path_obj)

            # Determine hidden size
            hidden_size = None
            if hasattr(model, "args") and hasattr(model.args, "hidden_size"):
                 hidden_size = model.args.hidden_size
            if hidden_size is None:
                 hidden_size = model_config.get("hidden_size")
            if hidden_size is None:
                logger.warning("Could not find hidden_size in model.args or config. Attempting inference...")
                try:
                    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                        hidden_size = model.model.embed_tokens.weight.shape[-1]
                    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                        hidden_size = model.transformer.wte.weight.shape[-1]
                    else:
                        raise AttributeError("Cannot infer hidden_size from known model structures.")
                    logger.info(f"Inferred hidden size from model weights: {hidden_size}")
                except Exception as infer_e:
                     logger.critical(f"Failed to infer hidden_size: {infer_e}")
                     raise ValueError("Cannot determine model's hidden_size.") from infer_e

            model_type = model_config.get("model_type", "unknown")
            logger.info(f"Model loaded successfully. Type: {model_type}, Hidden Size: {hidden_size}")
            logger.info(f"Tokenizer loaded: {tokenizer}")

            # Initialize Critic network
            logger.info(f"Initializing new critic network (hidden_size={hidden_size})...")
            critic = CriticNetwork(hidden_size=hidden_size)
            mx.eval(critic.parameters()) # Evaluate critic params
            logger.info("Critic network initialized.")

        except Exception as e:
            logger.critical(
                f"Failed to load base model or tokenizer from '{args.model_path}': {e}",
                exc_info=True,
            )
            sys.exit(1)

    # --- Final Checks and Setup ---
    # Ensure PAD token exists (common check for both resume and fresh start)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            logger.warning(
                f"Tokenizer lacks PAD token. Using EOS token '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id}) for padding."
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.error("Tokenizer lacks both PAD and EOS tokens. Cannot proceed with padding.")
            raise ValueError("Tokenizer must have a pad_token_id or eos_token_id.")

    # --- Initialize Reward Function, Environments ---
    logger.info("Initializing Reward function...")
    reward_config = RewardConfig()
    reward_function = SimpleRewardFunction(
        config=reward_config, verbose=kwargs["verbose"]
    )
    logger.info("Reward function initialized.")

    try:
        logger.info("Initializing training environment...")
        train_env = LLMEnv(
            args.train_dataset_path, tokenizer, reward_function,
            args.max_prompt_len, "train", args.shuffle_data,
        )
        logger.info(f"Training env '{train_env.env_id}' ready: {len(train_env)} samples.")

        val_env = None
        if args.val_dataset_path:
            logger.info("Initializing validation environment...")
            try:
                val_env = LLMEnv(
                    args.val_dataset_path, tokenizer, reward_function,
                    args.max_prompt_len, "val", shuffle_data=False,
                )
                logger.info(f"Validation env '{val_env.env_id}' ready: {len(val_env)} samples.")
            except Exception as e_val:
                logger.warning(f"Failed to initialize validation environment: {e_val}. Proceeding without validation.")
        else:
            logger.info("No validation dataset path provided. Skipping validation.")

    except Exception as e_env:
        logger.critical(f"Fatal error initializing environments: {e_env}", exc_info=True)
        sys.exit(1)

    # --- Initialize Optimizers ---
    # Note: Optimizer state is NOT loaded when resuming in this implementation.
    # The optimizer will start fresh.
    actor_optimizer = optim.AdamW(learning_rate=args.actor_lr)
    critic_optimizer = optim.AdamW(learning_rate=args.critic_lr)
    logger.info(
        f"Initialized Optimizers: AdamW (Actor LR={args.actor_lr}, Critic LR={args.critic_lr})"
        f"{' (Optimizer state not resumed)' if args.resume_from_checkpoint else ''}"
    )

    # --- Start Training ---
    logger.info("[bold blue]>>> Starting Training Process <<<[/]")
    start_train_time = time.monotonic()
    try:
        # Call the main training function
        train(
            args=args,
            model=model,
            model_config=model_config, # Pass config for saving checkpoints
            critic=critic,
            tokenizer=tokenizer,
            train_env=train_env,
            val_env=val_env,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            reward_fn=reward_function,
        )
    except Exception as train_err:
        # Catch unexpected errors during the train function execution
        logger.critical("Training process failed with an unexpected error.", exc_info=True)
    finally:
        # Log total training time regardless of success or failure
        duration_mins = (time.monotonic() - start_train_time) / 60
        logger.info(f"Total script execution time: {duration_mins:.2f} minutes.")
        logger.info("[bold blue]>>> Training Process Ended <<<[/]")


if __name__ == "__main__":
    # Entry point when script is executed
    cli_main()
