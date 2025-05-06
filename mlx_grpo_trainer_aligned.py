# mlx_grpo_trainer_aligned.py - Production-Ready GRPO Trainer
import traceback
import json
import random
import argparse
import sys
import math
import gc
import logging
import os
import time
import re
import signal
import csv

import argparse
import contextlib
import functools
import json
import math
import os
import random
import re
import shutil
import signal
import sys
import threading
import time
import traceback
from collections import Counter  # Keep if needed for vocab building, but not used in trainer
from dataclasses import (
    MISSING,
    asdict,
    dataclass,
    field,
    fields,
    is_dataclass,
)
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

# Numpy
import numpy as np

# MLX components
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from llama_rl.utils import (
    _save_and_commit,
    _save_directory_and_commit,  # (Needs definition if not in utils)
    _check_disk_space,
    MIN_REQUIRED_BYTES,
    limit_memory,
    # save_config # (Or a similar function for saving model config JSON)
)


# Import rich
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# --- Optional Dependencies ---
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print(
        "WARNING: psutil not found (`pip install psutil`). Memory monitoring disabled.",
        file=sys.stderr,
    )

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from datasets import Dataset, load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print(
        "WARNING: datasets library not found (`pip install datasets`). Loading from Hugging Face Hub or local JSONL will fail.",
        file=sys.stderr,
    )

# --- MLX-LM Imports ---
try:
    import mlx_lm
    from mlx_lm.generate import (
        GenerationResponse,
        generate_step,
        speculative_generate_step,
        wired_limit,
        maybe_quantize_kv_cache, # <-- CORRECT IMPORT LOCATION
    )
    from mlx_lm.models import cache
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import build_schedule, print_trainable_parameters
    from mlx_lm.utils import (
        _get_classes,
        get_model_path,
        load,
        load_config,
        make_shards,
        save_config,
    )

    # Import quantized layers for type checking/resizing
    from mlx.nn.layers.quantized import QuantizedEmbedding, QuantizedLinear

except ImportError as e:
    print(
        f"[bold red]Import error (mlx_lm components):[/] {e}. Please install/update mlx-lm: [code]pip install -U mlx-lm[/]"
    )
    traceback.print_exc()
    sys.exit(1)

# --- Assumed External Utilities (Replace if not available) ---
# These functions are assumed to be available from llama_rl.utils or similar.
# If not, they need to be implemented here. They provide safe, atomic file operations.
try:
    from llama_rl.utils import _save_and_commit, limit_memory

    # _save_directory_and_commit is not used in the revised save_checkpoint
    # _check_disk_space, MIN_REQUIRED_BYTES are not used
    # save_config is imported from mlx_lm.utils
except ImportError:
    print(
        "[yellow]Warning: llama_rl.utils not found. Using fallback dummy save functions. Checkpointing may not be atomic.[/]"
    )

    # Dummy fallback functions if llama_rl.utils is not available
    def _save_and_commit(temp_prefix, target_path, save_fn):
        """Dummy save function (not atomic)."""
        try:
            save_fn(target_path)
        except Exception as e:
            print(f"Dummy save failed for {target_path}: {e}", file=sys.stderr)
            raise

    def limit_memory(mb):
        """Dummy memory limit function."""
        print(f"Dummy memory limit called with {mb} MB.", file=sys.stderr)
        pass


# --- Assumed External Reward Function (Replace if not available) ---
# This function is assumed to be available from llama_rl.reward or similar.
try:
    from llama_rl.reward import SimpleRewardFunction, RewardConfig as SimpleRewardConfig

    # Instantiate with a default config
    EXRERNAL_REWARD_FN = SimpleRewardFunction(SimpleRewardConfig())
except ImportError:
    print(
        "[yellow]Warning: llama_rl.reward not found. External reward function will not be available.[/]"
    )
    EXRERNAL_REWARD_FN = None


# --- Global Variables ---
logger = logging.getLogger(__name__)  # Configured in main()
console = Console(stderr=True, force_terminal=True)
shutdown_requested = False
SAVE_ON_EXIT_FLAG_PATH = Path(".save_on_exit_request")
mlx_rng_key = mx.random.key(0)  # Default key, potentially updated by seeding/checkpoint
# MLX RNG state is managed internally by mx.random.seed/set_key/get_state
# We only need a variable to hold the key *if* we need to explicitly manage it,
# but mx.random functions use the internal state. Let's rely on the internal state.
# mlx_rng_key = None # Removed global mlx_rng_key variable

# Define target float dtype for calculations and resizing
TARGET_FLOAT_DTYPE = mx.bfloat16  # Or mx.float32


# --- Reward Logic ---
@dataclass
class RewardConfig:
    """Configuration for reward calculation tags."""

    think_start_tag: str = "<thinking>"
    think_end_tag: str = "</thinking>"
    answer_start_tag: str = "<answer>"
    answer_end_tag: str = "</answer>"


def format_reward(text: str, config: RewardConfig) -> float:
    """
    Assigns reward based on the presence and order of think/answer tags
    using regular expressions for robustness to whitespace.
    Penalizes missing structure or empty content between tags. Gives 0.5 for perfect format.
    """
    # Strip tags from config just in case they have extra whitespace, then escape for regex
    think_start_esc = re.escape(config.think_start_tag.strip())
    think_end_esc = re.escape(config.think_end_tag.strip())
    answer_start_esc = re.escape(config.answer_start_tag.strip())
    answer_end_esc = re.escape(config.answer_end_tag.strip())

    # Regex pattern:
    # - Looks for think_start, captures content (non-greedy), looks for think_end
    # - Allows flexible whitespace (\s*) between elements
    # - Looks for answer_start, captures content (non-greedy), looks for answer_end
    # - re.DOTALL makes '.' match newline characters.
    # - re.IGNORECASE makes tag matching case-insensitive.
    pattern = rf"{think_start_esc}\s*(.*?)\s*{think_end_esc}\s*{answer_start_esc}\s*(.*?)\s*{answer_end_esc}"

    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        # Structure <thinking>...</thinking>...<answer>...</answer> found in correct order.
        # Check if there is *any* non-whitespace content within the capture groups
        think_content = match.group(1)  # Raw captured content
        answer_content = match.group(2)  # Raw captured content

        if think_content.strip() and answer_content.strip():
            logger.debug("Format Reward: Correct structure and non-empty content found.")
            return 0.5  # Reward for correct structure and non-empty content
        else:
            logger.debug("Format Reward: Correct structure but empty content found.")
            return 0.0  # Neutral reward if tags okay but content missing
    else:
        logger.debug("Format Reward: Required tag structure not found.")
        return -0.5  # Penalty for missing structure or wrong order


def math_eval_reward(
    text: str, reference_answer_str: Optional[str], config: RewardConfig
) -> float:
    """
    Assigns reward based on evaluating the expression in answer tags
    and comparing to a numeric reference answer. Gives 1.0 for correct math eval.
    *** WARNING: Uses eval(), posing a SEVERE SECURITY RISK if inputs are not trusted. ***
    """
    if reference_answer_str is None:
        logger.debug("Math Eval Reward: No reference answer provided.")
        return 0.0

    start_tag_esc = re.escape(config.answer_start_tag.strip())
    end_tag_esc = re.escape(config.answer_end_tag.strip())
    pattern = rf"{start_tag_esc}\s*(.*?)\s*{end_tag_esc}"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        logger.debug("Math Eval Reward: <answer> tags not found or pattern mismatch.")
        return 0.0

    extracted_expr = match.group(1).strip()
    if not extracted_expr:
        logger.debug("Math Eval Reward: Extracted expression content is empty.")
        return 0.0

    try:
        # Clean the reference answer to get a numeric value
        cleaned_reference = reference_answer_str
        if "####" in cleaned_reference:
            try:
                cleaned_reference = cleaned_reference.split("####", 1)[1].strip()
            except IndexError:
                logger.debug(
                    f"Math Eval Reward: '####' found but no content after it in reference: '{reference_answer_str[:50]}...'"
                )
                return 0.0
        cleaned_reference = re.sub(r"[^\d\.\-eE]", "", cleaned_reference)

        if not cleaned_reference:
            logger.debug(
                f"Math Eval Reward: Could not extract numeric value from reference: '{reference_answer_str[:50]}...' (Cleaned: '{cleaned_reference}')"
            )
            return 0.0

        try:
            reference_answer_numeric = float(cleaned_reference)
        except ValueError:
            logger.debug(
                f"Math Eval Reward: Cleaned reference '{cleaned_reference}' is not a valid number."
            )
            return 0.0

        # --- SECURITY WARNING ---
        # Restrict evaluation environment severely
        safe_expr = extracted_expr.replace("^", "**")
        # Basic check for potentially malicious patterns
        if re.search(r'[;\'"`!@#$%^&*<>/\\|~]', safe_expr):
            logger.warning(
                f"Math Eval Reward: Detected potentially unsafe characters in expression '{safe_expr}'. Skipping evaluation."
            )
            return 0.0

        # Limited eval scope - explicitly list allowed functions/objects
        allowed_math_fns = [
            "sqrt",
            "pow",
            "exp",
            "log",
            "log10",
            "sin",
            "cos",
            "tan",
            "pi",
            "e",
            "radians",
            "degrees",
            "ceil",
            "floor",
            "round",
        ]
        allowed_math = {
            k: getattr(math, k) for k in allowed_math_fns if hasattr(math, k)
        }

        allowed_globals = {
            "__builtins__": {"abs": abs, "round": round, "min": min, "max": max},
            "math": allowed_math,
        }
        allowed_locals = {}

        evaluated_answer = eval(safe_expr, allowed_globals, allowed_locals)

        if not isinstance(evaluated_answer, (int, float)):
            logger.debug(
                f"Math Eval Reward: Evaluated expression '{extracted_expr}' did not result in a number ({type(evaluated_answer).__name__})."
            )
            return 0.0

        # Use math.isclose for float comparison
        return (
            1.0
            if math.isclose(
                float(evaluated_answer),
                reference_answer_numeric,
                rel_tol=1e-4,
                abs_tol=1e-6,
            )
            else 0.0
        )

    except (
        SyntaxError,
        NameError,
        TypeError,
        ValueError,
        OverflowError,
        ZeroDivisionError,
        Exception,
    ) as e:
        logger.debug(
            f"Math Eval Reward: Evaluation failed for expression '{extracted_expr}' (Ref: {reference_answer_str[:50]}...): {type(e).__name__} - {e}"
        )
        return 0.0


def jaccard_reward(
    text: str, reference_completion: Optional[str], config: RewardConfig
) -> float:
    """
    Assigns reward based on the Jaccard Similarity between the token sets
    of the text within <answer> tags and the reference completion string. Gives score 0-1.
    """
    import string # Ensure string is imported here if not globally

    if reference_completion is None:
        logger.debug("Jaccard Reward: No reference completion provided.")
        return 0.0

    start_tag_esc = re.escape(config.answer_start_tag.strip())
    end_tag_esc = re.escape(config.answer_end_tag.strip())
    pattern = rf"{start_tag_esc}\s*(.*?)\s*{end_tag_esc}"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        logger.debug("Jaccard Reward: <answer> tags not found or pattern mismatch.")
        return 0.0

    extracted_answer = match.group(1).strip()

    def preprocess_text(input_text: str) -> set:
        if not isinstance(input_text, str):
            logger.debug(
                f"Jaccard preprocess_text: Input is not a string ({type(input_text)}). Returning empty set."
            )
            return set()
        text_lower = input_text.lower()
        text_no_punct = re.sub(
            r"[%s]+" % re.escape(string.punctuation), " ", text_lower
        )
        tokens = set(text_no_punct.split())
        return tokens

    generated_tokens = preprocess_text(extracted_answer)
    reference_tokens = preprocess_text(reference_completion)

    if not generated_tokens and not reference_tokens:
        return 1.0
    if not generated_tokens or not reference_tokens:
        return 0.0

    intersection = generated_tokens.intersection(reference_tokens)
    union = generated_tokens.union(reference_tokens)

    if not union:
        return 0.0
    else:
        jaccard_score = len(intersection) / len(union)
        return float(jaccard_score)


# Map content reward type strings to functions
CONTENT_REWARD_FUNCTIONS: Dict[
    str, Callable[[str, Optional[str], RewardConfig], float]
] = {
    "math_eval": math_eval_reward,
    "jaccard": jaccard_reward,
}


def get_content_reward_fn(
    reward_content_type: str,
) -> Callable[[str, Optional[str], RewardConfig], float]:
    """Returns the content reward function based on the type string."""
    fn = CONTENT_REWARD_FUNCTIONS.get(reward_content_type.lower())
    if fn is None:
        valid_types = list(CONTENT_REWARD_FUNCTIONS.keys())
        logger.error(
            f"Unknown reward_content_type: '{reward_content_type}'. Valid types are: {valid_types}."
        )
        raise ValueError(f"Unknown reward_content_type: '{reward_content_type}'")
    return fn


def calculate_total_reward(
    generated_text: str,
    reference_answer_str: Optional[str],
    reward_config: RewardConfig,
    reward_format_weight: float,
    reward_content_weight: float,
    reward_content_type: str,
) -> Tuple[float, float, float]:
    """
    Calculates the total weighted reward based on format (0.5/-0.5) and content (0-1) components.
    Total reward = w_fmt * fmt_rew + w_cont * cont_rew.
    Returns (total_weighted_reward, raw_format_reward, raw_content_reward_combined).
    """
    if not math.isclose(reward_format_weight + reward_content_weight, 1.0):
        logger.warning(
            f"Reward weights ({reward_format_weight} + {reward_content_weight}) do not sum to 1.0. Total reward is a weighted sum, not a simple average."
        )

    format_rew = format_reward(generated_text, reward_config)
    logger.debug(f"Format Reward: {format_rew:.2f}")

    content_rew_internal = 0.0
    external_rew_value = 0.0
    external_rew_weight = 0.0 # Default weight for external reward

    try:
        content_reward_fn = get_content_reward_fn(reward_content_type)
        content_rew_internal = content_reward_fn(
            generated_text, reference_answer_str, reward_config
        )
        logger.debug(f"Content Reward (Internal {reward_content_type}): {content_rew_internal:.8f}")

        # Apply external reward if available and reference answer exists
        if EXRERNAL_REWARD_FN is not None and reference_answer_str is not None:
             try:
                 # Assuming calculate_reward returns a tuple (bonus_value, weight)
                 # The weight from the external function is ignored here, using a fixed weight from args or default
                 external_rew_value, _ = EXRERNAL_REWARD_FN.calculate_reward(generated_text, reference_answer_str)
                 # Use a fixed weight for external reward if desired, or make it configurable
                 external_rew_weight = 0.15 # Example fixed weight for external bonus
                 logger.debug(f"External Reward: {external_rew_value:.4f} (Fixed Weight: {external_rew_weight:.2f})")

                 # Combine internal and external content rewards additively
                 # Ensure external_rew_value is a number before adding
                 if isinstance(external_rew_value, (int, float)):
                      content_rew_combined = content_rew_internal + (external_rew_value * external_rew_weight)
                 else:
                      logger.warning(f"External reward value is not numeric ({type(external_rew_value)}). Skipping combination.")
                      content_rew_combined = content_rew_internal

                 logger.debug(f"Combined Content Reward (Internal + External): {content_rew_combined:.4f}")

             except Exception as e_external:
                 logger.error(f"Error calculating external reward: {e_external}", exc_info=False)
                 content_rew_combined = content_rew_internal # Fallback to internal only
        else:
             content_rew_combined = content_rew_internal # No external reward available or no reference answer

    except ValueError:
        logger.error(
            f"Invalid reward_content_type '{reward_content_type}' during reward calculation. Setting content reward to 0.",
            exc_info=True,
        )
        content_rew_combined = 0.0
    except Exception as e:
        logger.error(
            f"Error calculating content reward ({reward_content_type}): {e}",
            exc_info=True,
        )
        content_rew_combined = 0.0

    # Calculate total weighted reward
    total_rew = (reward_format_weight * format_rew) + (
        reward_content_weight * content_rew_combined
    )

    return (
        float(total_rew),
        float(format_rew),
        float(content_rew_combined),
    )


# --- Signal Handler ---
def handle_signal(signum, frame):
    """Handles SIGINT/SIGTERM for graceful shutdown."""
    global shutdown_requested
    signal_name = signal.Signals(signum).name
    if not shutdown_requested:
        console.print(
            f"\n[yellow]Signal {signal_name} received. Requesting shutdown & save...[/]"
        )
        shutdown_requested = True
        try:
            SAVE_ON_EXIT_FLAG_PATH.touch()
            logger.warning(f"Save flag created: {SAVE_ON_EXIT_FLAG_PATH}")
        except OSError as e:
            logger.debug(f"Could not create save flag: {e}")
    else:
        console.print(f"[red]Signal {signal_name} received again. Forcing exit...[/]")
        logging.shutdown()
        sys.exit(1)


# --- Configuration Dataclass ---
@dataclass
class TrainingArgs:
    """Configuration arguments for the GRPO training script."""

    # --- Paths & Setup ---
    output_dir: str = field(
        metadata={"help": "MANDATORY: Directory for checkpoints, logs, final model."}
    )

    # --- Paths & Setup ---
    max_kv_size: int = field(
        metadata={"help": "MANDATORY: Directory for checkpoints, logs, final model."}
    )

    model_path: str = field(
        metadata={"help": "MANDATORY: Path or ID of the base MLX model to train."}
    )
    ref_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path or ID of the reference model for KL penalty (defaults to model_path if None)."
        },
    )
    train_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local training JSONL path (overrides HF dataset)."},
    )
    val_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local validation JSONL path (overrides HF dataset)."},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Hugging Face dataset name (used if local paths are not provided)."
        },
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Hugging Face dataset configuration (e.g., 'main', 'viewer')."
        },
    )
    dataset_train_split: str = field(
        default="train", metadata={"help": "Name of the training split in the dataset."}
    )
    dataset_val_split: str = field(
        default="test",
        metadata={"help": "Name of the validation split in the dataset."},
    )
    dataset_prompt_key: str = field(
        default="prompt",
        metadata={"help": "Dataset dictionary key for the input prompt/question."},
    )
    dataset_answer_key: str = field(
        default="completion",
        metadata={"help": "Dataset dictionary key for the reference answer."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a checkpoint directory to resume training from (contains weights, optimizer, etc.). If not set, tries to find 'latest' symlink in output_dir."
        },
    )

    # --- Model & Tokenizer ---
    max_prompt_len: int = field(
        default=100,
        metadata={
            "help": "Maximum number of tokens for the input prompt (truncates longer prompts)."
        },
    )
    max_gen_len: int = field(
        default=50,
        metadata={"help": "Maximum number of tokens to generate during rollout."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": "System prompt to use in chat format (uses default if None)."
        },
    )
    # Special Tokens
    think_start_tag: str = field(
        default="<thinking>",
        metadata={"help": "Special token marking the start of the thinking process."},
    )
    think_end_tag: str = field(
        default="</thinking>",
        metadata={"help": "Special token marking the end of the thinking process."},
    )
    answer_start_tag: str = field(
        default="<answer>",
        metadata={"help": "Special token marking the start of the answer."},
    )
    answer_end_tag: str = field(
        default="</answer>",
        metadata={"help": "Special token marking the end of the answer."},
    )
    init_new_embeddings_with_mean: bool = field(
        default=False,
        metadata={
            "help": "Initialize embeddings for new special tokens with the mean of existing embeddings."
        },
    )

    # --- Optimizer & Scheduling ---
    learning_rate: float = field(
        default=3e-5, metadata={"help": "Peak learning rate."}
    )
    # Use a dictionary for schedule config, matching mlx_lm.tuner.utils.build_schedule
    lr_schedule_config: Dict = field(
        default_factory=lambda: {
            "name": "linear_schedule",
            "arguments": [3e-5, 1e-7, 500], # Example: start_lr, end_lr, total_steps
            "warmup": 10, # Example: warmup steps
            "warmup_init": 1e-8, # Example: initial LR for warmup
        },
        metadata={
            "help": "Configuration for the learning rate schedule (dict matching mlx_lm.tuner.utils.build_schedule format)."
        },
    )
    grad_clip_norm: Optional[float] = field(
        default=0.9,
        metadata={"help": "Maximum gradient norm for clipping (0 or None to disable)."},
    )
    optimizer_beta1: float = field(
        default=0.9, metadata={"help": "AdamW optimizer beta1 parameter."}
    )
    optimizer_beta2: float = field(
        default=0.95, metadata={"help": "AdamW optimizer beta2 parameter."}
    )
    optimizer_weight_decay: float = field(
        default=0.01, metadata={"help": "Weight decay for the AdamW optimizer."}
    )

    # --- GRPO/RL Parameters ---
    num_rollout_samples: int = field(
        default=2,
        metadata={
            "help": "Number of responses to generate per prompt during rollout (GRPO group size)."
        },
    )
    ppo_batch_size: int = field(
        default=2,
        metadata={
            "help": "Number of prompts processed in one rollout/gradient calculation step (micro-batch size)."
        },
    )
    sampling_temperature: float = field(
        default=0.8,
        metadata={
            "help": "Temperature for sampling during rollouts (higher is more random)."
        },
    )
    sampling_top_p: float = field(
        default=1.0,
        metadata={
            "help": "Top-p (nucleus) sampling probability during rollouts (0 to disable)."
        },
    )
    sampling_min_p: float = field(
        default=0.0,
        metadata={"help": "Min-p sampling probability during rollouts (0 to disable)."},
    )
    grpo_beta: float = field(
        default=0.15,
        metadata={"help": "Beta hyperparameter for GRPO KL penalty strength."},
    )
    advantage_epsilon: float = field(
        default=1e-8,
        metadata={
            "help": "Epsilon added to advantage normalization denominator for stability."
        },
    )

    # --- Speculative Decoding ---
    use_speculative_decoding: bool = field(
        default=False,
        metadata={"help": "Enable speculative decoding during rollouts."},
    )
    draft_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path or ID of the draft model for speculative decoding (defaults to ref_model_path if None)."
        },
    )
    num_draft_tokens: int = field(
        default=3,
        metadata={"help": "Number of tokens to draft when using speculative decoding."},
    )

    # --- KV Cache Quantization ---
    kv_bits: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of bits for KV cache quantization (None to disable)."
        },
    )
    kv_group_size: int = field(
        default=64, metadata={"help": "Group size for KV cache quantization."}
    )
    quantized_kv_start: int = field(
        default=5000,
        metadata={
            "help": "When --kv-bits is set, start quantizing the KV cache from this step onwards."
        },
    )

    # --- Control & Techniques ---
    num_training_steps: int = field(
        default=500,
        metadata={"help": "Total number of training update steps (optimizer steps)."},
    )
    save_every: int = field(
        default=2, metadata={"help": "Save checkpoint every N update steps."}
    )
    eval_every: int = field(
        default=10, metadata={"help": "Evaluate every N update steps."}
    )
    seed: int = field(default=42, metadata={"help": "Random seed."})
    shuffle_data: bool = field(
        default=True,
        metadata={"help": "Shuffle training data indices each epoch/pass."},
    )
    grad_accum_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation steps."}
    )
    use_grad_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing."},
    )
    grad_checkpoint_layers: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of layers for grad checkpointing (default: all applicable if enabled)."
        },
    )

    # --- Reward Configuration ---
    reward_format_weight: float = field(
        default=0.4,
        metadata={"help": "Weight for the format/structure reward component."},
    )
    reward_content_weight: float = field(
        default=0.6, metadata={"help": "Weight for the content reward component."}
    )
    reward_content_type: str = field(
        default="jaccard",
        metadata={
            "help": f"Type of content reward function to use: {list(CONTENT_REWARD_FUNCTIONS.keys())}. 'math_eval' uses eval() (SECURITY RISK!), 'jaccard' uses token overlap."
        },
    )

    # --- Logging ---
    verbose: bool = field(
        default=True, metadata={"help": "Enable DEBUG level logging."}
    )
    use_wandb: bool = field(
        default=True, metadata={"help": "Log metrics to Weights & Biases."}
    )
    wandb_project: Optional[str] = field(
        default="mlx-grpo-finetune", metadata={"help": "WandB project name."}
    )
    wandb_entity: Optional[str] = field(
        default=None, metadata={"help": "WandB entity (user or team name)."}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "Custom name for the WandB run."}
    )

    # --- Calculated Field ---
    effective_batch_size: int = field(init=False)

    def __post_init__(self):
        """Perform validation checks and calculate derived fields."""
        if not self.output_dir:
            raise ValueError("--output-dir is mandatory.")
        if not self.model_path:
            raise ValueError("--model-path is mandatory.")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1.")
        if self.ppo_batch_size < 1:
            raise ValueError("ppo_batch_size must be >= 1.")
        if self.num_rollout_samples < 1:
            raise ValueError("num_rollout_samples must be >= 1.")
        if self.num_rollout_samples > 1 and not math.isclose(
            self.reward_format_weight + self.reward_content_weight, 1.0
        ):
            logger.warning(
                f"Reward weights ({self.reward_format_weight} + {self.reward_content_weight}) do not sum to 1.0. Total reward is a weighted sum, not a simple average."
            )

        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            print("INFO: grad_clip_norm <= 0, disabling gradient clipping.")
            self.grad_clip_norm = None

        if self.ref_model_path is None:
            print(
                f"INFO: ref_model_path not set, defaulting to model_path: '{self.model_path}'"
            )
            self.ref_model_path = self.model_path

        if self.use_speculative_decoding:
            if self.draft_model_path is None:
                print(
                    f"INFO: draft_model_path not set, defaulting to ref_model_path: '{self.ref_model_path}' for speculative decoding."
                )
                self.draft_model_path = self.ref_model_path
            if self.num_draft_tokens < 1:
                 raise ValueError("num_draft_tokens must be >= 1 for speculative decoding.")
            if self.draft_model_path == self.model_path:
                 print("[yellow]Warning: Using the same model for actor and draft in speculative decoding (self-speculation).[/]")


        if self.kv_bits is not None and (self.kv_bits <= 0 or self.kv_bits > 8):
             raise ValueError("kv_bits must be None or between 1 and 8.")
        if self.kv_bits is not None and self.kv_group_size <= 0:
             raise ValueError("kv_group_size must be > 0 when kv_bits is set.")
        if self.kv_bits is not None and self.quantized_kv_start < 0:
             raise ValueError("quantized_kv_start must be >= 0 when kv_bits is set.")


        # Check reward content type is valid
        if self.reward_content_type.lower() not in CONTENT_REWARD_FUNCTIONS:
            valid_types = list(CONTENT_REWARD_FUNCTIONS.keys())
            raise ValueError(
                f"Invalid reward_content_type: '{self.reward_content_type}'. Must be one of: {valid_types}"
            )
        if self.reward_content_type.lower() == "math_eval":
            logger.warning(
                "[yellow bold]Using 'math_eval' content reward: This uses eval() and is a SECURITY RISK. Only use on trusted model outputs/datasets.[/]"
            )

        if self.system_prompt is None:
            # Default system prompt tailored to guide the model to use the tags
            default_sys = f"""You are a helpful assistant. Always use clear and concise language. Make sure the output is well-formatted, with proper indentation and structure. If the user asks for a solution, provide only the core components necessary to solve the task without any extraneous explanations. The focus should always be on functionality and efficiency.

            First think step-by-step – describe your plan, written out in great detail.. Minimize any other prose. Keep your answers short and impersonal. Use Markdown formatting in your answers. Always ensure clarity in your instructions and code, following a structured approach to solve the problem. Avoid unnecessary elaboration, focusing only on what is needed to complete the task.

            For tasks that involve multiple steps, ensure that each step is clearly outlined in the pseudocode before translating it into code. This helps avoid confusion and ensures that the final code matches the initial plan. Always use the most efficient methods and algorithms suited for the task.


            Respond with your step-by-step thinking process inside `{self.think_start_tag}` and `{self.think_end_tag}` tags. Then, provide the final answer content inside `{self.answer_start_tag}` and `{self.answer_end_tag}` tags."""

            self.system_prompt = default_sys
            print("INFO: Using default system prompt tailored to special tokens.")

        # Calculate effective batch size (total samples processed per optimizer step)
        # This is the number of generated sequences per update
        self.effective_batch_size = (
            self.ppo_batch_size * self.grad_accum_steps * self.num_rollout_samples
        )
        print(
            f"INFO: Effective batch size (samples per optimizer update): {self.effective_batch_size}"
        )


# --- Utility Functions ---


def apply_chat_template_wrapper(
    tokenizer: TokenizerWrapper, prompt: str, system_prompt: Optional[str]
) -> str:
    """Formats prompt using tokenizer.apply_chat_template."""
    messages = []
    if (
        system_prompt and system_prompt.strip()
    ):
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        logger.debug("Applied chat template successfully.")
        return formatted_prompt
    except Exception as e:
        logger.error(
            f"Error applying chat template: {e}. Falling back to simple format.",
            exc_info=False,
        )
        prefix = (
            f"System: {system_prompt.strip()}\n\n"
            if system_prompt and system_prompt.strip()
            else ""
        )
        return f"{prefix}User: {prompt.strip()}\n\nAssistant:"


def selective_softmax(logits: mx.array, tokens: mx.array) -> mx.array:
    """Calculates the log probability of specific target tokens given logits."""
    log_probs_all = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    tokens_expanded = (
        tokens[..., None] if tokens.ndim == log_probs_all.ndim - 1 else tokens
    )
    log_probs = mx.take_along_axis(
        log_probs_all, tokens_expanded.astype(mx.int64), axis=-1
    ).squeeze(-1)
    return log_probs.astype(mx.float32)


# A stream on the default device just for generation
generation_stream = mx.new_stream(mx.default_device())


def _create_4d_attention_mask(
    tokens: mx.array, pad_token_id: int, dtype: mx.Dtype = TARGET_FLOAT_DTYPE
) -> mx.array:
    """Creates a 4D attention mask combining causal and padding masks."""
    if tokens.ndim != 2:
        raise ValueError(
            f"Input tokens must be 2D (batch_size, sequence_length), got shape {tokens.shape}"
        )
    batch_size, sequence_length = tokens.shape

    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(
        sequence_length, dtype=dtype
    )

    padding_mask_2d = tokens == pad_token_id
    padding_mask_4d_keys = padding_mask_2d[:, None, None, :]

    neg_inf_val = mx.array(-1e9 if dtype != mx.bfloat16 else -65504.0, dtype=dtype)
    additive_padding_mask = mx.where(padding_mask_4d_keys, neg_inf_val, mx.zeros_like(neg_inf_val))

    combined_mask = mx.minimum(causal_mask[None, None, :, :], additive_padding_mask)

    return combined_mask


def build_rollout_batch(
    tokenizer: TokenizerWrapper,
    dataset: Dataset,
    indices: List[int],
    max_prompt_len: int,
    system_prompt: Optional[str],
    prompt_key: str,
    answer_key: str,
) -> Tuple[List[Dict], mx.array, int]:
    """
    Prepares a batch of unique tokenized prompts using chat template and corresponding answers.
    Adds basic validation for input text fields. Performs left-padding.
    """
    prompts_data = []
    max_len_in_batch = 0

    if tokenizer.pad_token_id is None:
        error_msg = "Tokenizer must have a pad_token_id for padding."
        logger.critical(error_msg)
        raise ValueError(error_msg)
    pad_id = tokenizer.pad_token_id

    for i in indices:
        try:
            sample_data = dataset[i]
            question = sample_data.get(prompt_key)
            answer_raw = sample_data.get(answer_key)

            if not isinstance(question, str) or not question.strip():
                logger.warning(f"Skipping dataset index {i}: Invalid/empty question ('{prompt_key}').")
                continue

            final_answer_str = None
            if isinstance(answer_raw, (str, int, float)):
                final_answer_str = str(answer_raw).strip()
                if not final_answer_str:
                    logger.debug(f"Dataset index {i}: Empty reference answer ('{answer_key}').")
                    final_answer_str = None
            elif answer_raw is not None:
                logger.debug(f"Dataset index {i}: Non-text reference answer type: {type(answer_raw)}.")

            try:
                prompt_text = apply_chat_template_wrapper(tokenizer, question, system_prompt)
            except Exception as e_template:
                logger.warning(f"Skipping index {i}: Failed applying chat template: {e_template}")
                continue

            try:
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            except Exception as e_tok:
                logger.warning(f"Skipping index {i}: Tokenization failed for '{prompt_text[:50]}...': {e_tok}")
                continue

            if len(prompt_tokens) > max_prompt_len:
                prompt_tokens = prompt_tokens[-max_prompt_len:]

            if not prompt_tokens:
                logger.warning(f"Skipping index {i}: Empty token list after tokenization/truncation.")
                continue

            prompts_data.append(
                {
                    "original_index": i,
                    "text": prompt_text,
                    "tokens": prompt_tokens,
                    "ref_answer_str": final_answer_str,
                }
            )
            max_len_in_batch = max(max_len_in_batch, len(prompt_tokens))

        except KeyError as e:
            logger.warning(f"Skipping index {i}: Missing key '{e}'.")
        except Exception as e:
            logger.warning(f"Skipping index {i}: Unexpected error: {e}", exc_info=False)

    if not prompts_data:
        logger.warning("No valid prompts found in the provided indices.")
        return [], mx.array([], dtype=mx.int32), 0

    padded_prompts_list = []
    for p_data in prompts_data:
        p_tokens = p_data["tokens"]
        pad_len = max_len_in_batch - len(p_tokens)
        padded_tokens = ([pad_id] * pad_len) + p_tokens
        padded_prompts_list.append(padded_tokens)

    try:
        prompts_mx = mx.array(padded_prompts_list, dtype=mx.int32)
    except Exception as e_pad:
        logger.error(f"Failed to create MLX array from padded prompts: {e_pad}", exc_info=True)
        return [], mx.array([], dtype=mx.int32), 0

    return prompts_data, prompts_mx, max_len_in_batch




def sample_logits(
    logits: mx.array,
    temp: float = 0.7,
    top_p: float = 0.9,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
) -> Tuple[mx.array, mx.array]:
    """
    Sample tokens from logits with log probability calculation.
    (Same as provided)
    """
    sampler = make_sampler(
        temp=temp, top_p=top_p, min_p=min_p, min_tokens_to_keep=min_tokens_to_keep
    )
    logits_f32 = logits.astype(mx.float32)
    tokens = sampler(logits_f32)
    log_probs_all = nn.log_softmax(logits_f32, axis=-1)
    tokens_expanded = (
        tokens[..., None] if tokens.ndim == log_probs_all.ndim - 1 else tokens
    )
    log_prob = mx.take_along_axis(
        log_probs_all, tokens_expanded.astype(mx.int64), axis=-1
    ).squeeze(-1)
    return tokens, log_prob


def selective_softmax(logits: mx.array, tokens: mx.array) -> mx.array:
    """Calculates the log probability of specific target tokens given logits."""
    # Ensure logits are float32 for stable softmax
    log_probs_all = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    tokens_expanded = (
        tokens[..., None] if tokens.ndim == log_probs_all.ndim - 1 else tokens
    )
    log_probs = mx.take_along_axis(
        log_probs_all, tokens_expanded.astype(mx.int64), axis=-1
    ).squeeze(-1)
    # Return in the original logits dtype or float32? Let's keep float32 for consistency downstream.
    return log_probs.astype(mx.float32)



# FIX START: generate_rollouts_for_batch method

def generate_rollouts_for_batch(
    model: nn.Module,
    ref_model: nn.Module,
    draft_model: Optional[nn.Module], # Added draft model
    tokenizer: TokenizerWrapper,
    prompts_data: List[Dict],
    prompts_mx: mx.array,
    max_prompt_len_batch: int,
    num_samples_per_prompt: int,
    max_gen_len: int,
    args: TrainingArgs,
) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:

    num_prompts = prompts_mx.shape[0]

    if num_prompts == 0:
        logger.warning("Received empty prompts_mx batch.")
        empty_batch = {
            "tokens": mx.array([], dtype=mx.int32),
            "response_mask": mx.array([], dtype=mx.float32),
            "advantages": mx.array([], dtype=mx.float32),
            "ref_log_probs": mx.array([], dtype=mx.float32),
        }
        return empty_batch, 0.0, {"raw_format": 0.0, "raw_content_combined": 0.0}

    total_samples = num_prompts * num_samples_per_prompt

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None: raise ValueError("Tokenizer must have a pad_token_id.")

    eos_token_str = tokenizer.decode([eos_id]) if eos_id is not None else ""
    pad_token_str = tokenizer.decode([pad_id])

    reward_config = RewardConfig(
        think_start_tag=args.think_start_tag, think_end_tag=args.think_end_tag,
        answer_start_tag=args.answer_start_tag, answer_end_tag=args.answer_end_tag,
    )

    # Use model's dtype for generation/cache
    model_dtype = TARGET_FLOAT_DTYPE # Use the target float dtype
    try:
        # Attempt to get model's actual parameter dtype if different
        model_dtype = next(tree_flatten(model.parameters()))[1].dtype
    except (StopIteration, Exception):
        logger.debug(f"Could not detect model dtype, using {model_dtype}.")

    # --- Repeat Prompts for GRPO Sampling ---
    prompts_mx_repeated = mx.repeat(prompts_mx, repeats=num_samples_per_prompt, axis=0)

    # --- Generation ---

    model.eval() # Set actor model to eval mode
    ref_model.eval() # Set ref model to eval mode
    if draft_model: draft_model.eval() # Set draft model to eval mode

    # Initialize caches for the batch of prompts
    validated_max_kv_size = None
    if args.max_kv_size is not None:
        try:
            if isinstance(args.max_kv_size, str):
                if args.max_kv_size.lower() == 'none':
                    validated_max_kv_size = None
                else:
                    validated_max_kv_size = int(args.max_kv_size)
            elif isinstance(args.max_kv_size, (int, float)):
                validated_max_kv_size = int(args.max_kv_size)
            else:
                logger.warning(f"Unexpected type for max_kv_size: {type(args.max_kv_size)}. Expected int or None. Using None.")
                validated_max_kv_size = None
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert max_kv_size '{args.max_kv_size}' to int: {e}. Using None.")
            validated_max_kv_size = None
    else:
        validated_max_kv_size = None

    model_caches = cache.make_prompt_cache(model, max_kv_size=validated_max_kv_size)
    ref_caches = cache.make_prompt_cache(ref_model, max_kv_size=validated_max_kv_size)
    if draft_model:
        draft_caches = cache.make_prompt_cache(draft_model, max_kv_size=validated_max_kv_size)
    else:
        draft_caches = None

    logger.debug(f"Initialized caches (will handle batch size {total_samples} dynamically). Max KV Size: {validated_max_kv_size if validated_max_kv_size is not None else 'Unlimited'}")

    sampler = make_sampler(
        temp=args.sampling_temperature,
        top_p=args.sampling_top_p,
        min_p=args.sampling_min_p,
        min_tokens_to_keep=1,
    )

    gen_start_time = time.monotonic()
    responses_tokens_list = [] # List to store generated token tensors for each step/segment
    response_log_probs_list = [] # List to store corresponding actor log probs

    # --- Initial Prompt Processing (Prefill) ---
    try:
        logger.debug(f"Processing prompts (Prefill). Input shape: {prompts_mx_repeated.shape}")
        prompt_attn_mask_4d = _create_4d_attention_mask(
            prompts_mx_repeated, pad_id, dtype=model_dtype
        )
        model_input_prompts = prompts_mx_repeated.astype(mx.int64)

        with mx.stream(generation_stream):
            model_output = model(model_input_prompts, mask=prompt_attn_mask_4d, cache=model_caches)
            ref_output = ref_model(model_input_prompts, mask=prompt_attn_mask_4d, cache=ref_caches)
            if draft_model:
                draft_output = draft_model(model_input_prompts, mask=prompt_attn_mask_4d, cache=draft_caches)

        next_token_logits = (model_output[0] if isinstance(model_output, tuple) else model_output)[:, -1, :]
        ref_next_token_logits = (ref_output[0] if isinstance(ref_output, tuple) else ref_output)[:, -1, :]
        if draft_model:
            draft_next_token_logits = (draft_output[0] if isinstance(draft_output, tuple) else draft_output)[:, -1, :]
        else:
            draft_next_token_logits = None # Define for eval list

        eval_list = [next_token_logits, ref_next_token_logits]
        if draft_next_token_logits is not None: eval_list.append(draft_next_token_logits)
        eval_list.extend([c.state for c in model_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)])
        eval_list.extend([c.state for c in ref_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)])
        if draft_model:
            eval_list.extend([c.state for c in draft_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)])
        mx.eval(eval_list)

        if args.kv_bits is not None:
            logger.debug(f"Applying KV cache quantization after prefill (threshold: {args.quantized_kv_start}).")
            maybe_quantize_kv_cache(model_caches, args.quantized_kv_start, args.kv_group_size, args.kv_bits)
            maybe_quantize_kv_cache(ref_caches, args.quantized_kv_start, args.kv_group_size, args.kv_bits)
            if draft_model:
                maybe_quantize_kv_cache(draft_caches, args.quantized_kv_start, args.kv_group_size, args.kv_bits)
            # Evaluate updated cache states
            eval_list_quant = [c.state for c in model_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)]
            eval_list_quant.extend([c.state for c in ref_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)])
            if draft_model:
                eval_list_quant.extend([c.state for c in draft_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)])
            if eval_list_quant: mx.eval(eval_list_quant)

    except Exception as e_prefill:
        logger.error(f"Initial prompt prefill failed: {e_prefill}", exc_info=True)
        model.train(); ref_model.train()
        if draft_model: draft_model.train()
        # Return structure consistent with success case but empty data
        return {
            "tokens": prompts_mx_repeated, # Return prompts to indicate context
            "response_mask": mx.zeros((total_samples, 0), dtype=mx.float32),
            "advantages": mx.zeros((total_samples, 1), dtype=mx.float32),
            "ref_log_probs": mx.zeros((total_samples, 0), dtype=mx.float32),
        }, 0.0, {"raw_format": 0.0, "raw_content_combined": 0.0}


    # --- Generation Loop (Batched) ---
    current_tokens = mx.zeros((total_samples,), dtype=mx.int32) # Placeholder
    ended = mx.full((total_samples,), False, dtype=mx.bool_)

    # Sample the first token
    try:
        current_tokens, current_log_probs = sample_logits(
            next_token_logits,
            temp=args.sampling_temperature, top_p=args.sampling_top_p,
            min_p=args.sampling_min_p, min_tokens_to_keep=1,
        )
        if eos_id is not None:
            ended = mx.logical_or(ended, mx.equal(current_tokens, eos_id))

        mx.eval(current_tokens, current_log_probs, ended)
        responses_tokens_list.append(current_tokens[:, None])
        response_log_probs_list.append(current_log_probs[:, None])

    except Exception as e_sample_first:
        logger.error(f"Initial token sampling failed: {e_sample_first}", exc_info=True)
        model.train(); ref_model.train()
        if draft_model: draft_model.train()
        # Return structure consistent with success case but empty data
        return {
            "tokens": prompts_mx_repeated,
            "response_mask": mx.zeros((total_samples, 0), dtype=mx.float32),
            "advantages": mx.zeros((total_samples, 1), dtype=mx.float32),
            "ref_log_probs": mx.zeros((total_samples, 0), dtype=mx.float32),
        }, 0.0, {"raw_format": 0.0, "raw_content_combined": 0.0}


    # Determine generation strategy
    use_speculative = args.use_speculative_decoding and draft_model is not None

    if not use_speculative:
        # --- Non-Speculative Batch Generation ---
        logger.debug("Using non-speculative batch generation.")
        try:
            for step in range(max_gen_len - 1): # Loop generates max_gen_len-1 more tokens
                if mx.all(ended).item():
                    logger.debug(f"All sequences ended generation at step {step+1}.")
                    break

                ended_before_step = ended # Capture state before this step's update

                try:
                    model_input_step = current_tokens[:, None].astype(mx.int64)
                    with mx.stream(generation_stream):
                        model_step_output = model(model_input_step, cache=model_caches)
                        ref_step_output = ref_model(model_input_step, cache=ref_caches) # Needed for ref_log_probs later? No, full pass later.

                    next_token_logits = (model_step_output[0] if isinstance(model_step_output, tuple) else model_step_output)[:, -1, :]
                    # ref_next_token_logits = (ref_step_output[0] if isinstance(ref_step_output, tuple) else ref_step_output)[:, -1, :] # Not needed here

                    sampled_tokens, current_log_probs = sample_logits(
                        next_token_logits, temp=args.sampling_temperature, top_p=args.sampling_top_p,
                        min_p=args.sampling_min_p, min_tokens_to_keep=1,
                    )

                    if eos_id is not None:
                        just_ended = mx.equal(sampled_tokens, eos_id)
                        ended = mx.logical_or(ended_before_step, just_ended)

                    # Select token: PAD if already ended before this step, else the new token
                    pad_val_mx = mx.array(pad_id, dtype=sampled_tokens.dtype)
                    tokens_to_add = mx.where(ended_before_step, pad_val_mx, sampled_tokens)

                    # Mask log probs: 0 if sequence was already ended
                    log_probs_to_add = mx.where(ended_before_step, mx.array(0.0, dtype=current_log_probs.dtype), current_log_probs)

                    # Evaluate async
                    eval_list = [tokens_to_add, log_probs_to_add, ended]
                    eval_list.extend([c.state for c in model_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)])
                    eval_list.extend([c.state for c in ref_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)])
                    mx.async_eval(eval_list)

                    responses_tokens_list.append(tokens_to_add[:, None])
                    response_log_probs_list.append(log_probs_to_add[:, None])

                    current_tokens = tokens_to_add # Input for the next step

                    # Apply KV cache quantization
                    if args.kv_bits is not None:
                        current_seq_len_after_step = model_caches[0].offset + 1
                        if current_seq_len_after_step > args.quantized_kv_start:
                            maybe_quantize_kv_cache(model_caches, args.quantized_kv_start, args.kv_group_size, args.kv_bits)
                            maybe_quantize_kv_cache(ref_caches, args.quantized_kv_start, args.kv_group_size, args.kv_bits)
                            eval_list_quant = [c.state for c in model_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)]
                            eval_list_quant.extend([c.state for c in ref_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)])
                            if eval_list_quant: mx.async_eval(eval_list_quant)

                except Exception as e_gen_step:
                    logger.error(f"Non-speculative generation failed at step {step+1}: {e_gen_step}", exc_info=True)
                    # Pad remaining steps robustly
                    current_len = sum(t.shape[1] for t in responses_tokens_list) if responses_tokens_list else 0
                    remaining_len = max_gen_len - current_len
                    if remaining_len > 0:
                        logger.warning(f"Padding remaining {remaining_len} steps due to error.")
                        if responses_tokens_list:
                             pad_dtype_tok = responses_tokens_list[0].dtype
                             pad_dtype_prob = response_log_probs_list[0].dtype
                             pad_array_tok = mx.full((total_samples, remaining_len), pad_id, dtype=pad_dtype_tok)
                             responses_tokens_list.append(pad_array_tok)
                             zero_array_prob = mx.full((total_samples, remaining_len), 0.0, dtype=pad_dtype_prob)
                             response_log_probs_list.append(zero_array_prob)
                        else: # Should not happen if first token succeeded, but for safety
                             pad_array_tok = mx.full((total_samples, remaining_len), pad_id, dtype=prompts_mx_repeated.dtype)
                             responses_tokens_list.append(pad_array_tok)
                             zero_array_prob = mx.full((total_samples, remaining_len), 0.0, dtype=mx.float32)
                             response_log_probs_list.append(zero_array_prob)
                    break # Exit loop on error

        except Exception as e_gen_loop:
            logger.error(f"Non-speculative generation loop failed: {e_gen_loop}", exc_info=True)
            # Error padding handled inside inner loop's exception block

    else: # use_speculative is True
        # --- Speculative Batch Generation ---
        logger.debug(f"Using speculative batch generation with {args.num_draft_tokens} draft tokens.")
        draft_input_dtype = mx.int64
        actor_input_dtype = mx.int64
        last_generated_tokens = prompts_mx_repeated[:, -1]
        generated_counts = mx.zeros((total_samples,), dtype=mx.int32)

        try:
            # Continue while any sequence needs more tokens and hasn't ended
            while mx.any(generated_counts < max_gen_len).item() and not mx.all(ended).item():
                if shutdown_requested:
                    logger.info("Shutdown requested during speculative generation loop.")
                    break

                # 1. Draft N tokens
                remaining_lens = max_gen_len - generated_counts
                # Draft at most num_draft_tokens, but no more than needed by any sequence
                num_draft = min(args.num_draft_tokens, mx.max(remaining_lens).item())
                if num_draft <= 0: break # Stop if all sequences have reached max length

                draft_tokens_batch = []
                draft_step_input = last_generated_tokens[:, None].astype(draft_input_dtype)

                for d_step in range(num_draft):
                    # Don't bother drafting if all sequences have ended (e.g., all generated EOS)
                    if mx.all(ended).item(): break

                    with mx.stream(generation_stream):
                        draft_output = draft_model(draft_step_input, cache=draft_caches)

                    draft_logits = (draft_output[0] if isinstance(draft_output, tuple) else draft_output)[:, -1, :]
                    # Use greedy sampling for draft model
                    draft_sampled_tokens, _ = sample_logits(draft_logits, temp=0.0, top_p=0.0, min_p=0.0)

                    # Draft PAD for sequences that have already ended
                    draft_sampled_tokens = mx.where(ended, mx.array(pad_id, dtype=draft_sampled_tokens.dtype), draft_sampled_tokens)

                    mx.eval(draft_sampled_tokens) # Evaluate each drafted token
                    draft_tokens_batch.append(draft_sampled_tokens[:, None])
                    draft_step_input = draft_sampled_tokens[:, None].astype(draft_input_dtype) # Use just drafted token for next input

                    # Apply KV cache quantization for draft model if enabled
                    if args.kv_bits is not None:
                        current_seq_len_draft = draft_caches[0].offset + 1
                        if current_seq_len_draft > args.quantized_kv_start:
                            maybe_quantize_kv_cache(draft_caches, args.quantized_kv_start, args.kv_group_size, args.kv_bits)
                            # Evaluate potentially updated cache states
                            eval_list_quant_draft = [c.state for c in draft_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)]
                            if eval_list_quant_draft: mx.async_eval(eval_list_quant_draft) # Async ok here

                if not draft_tokens_batch: break # No draft tokens generated (e.g., all ended)

                draft_tokens_batch_mx = mx.concatenate(draft_tokens_batch, axis=1) # Shape (B, num_draft)
                mx.eval(draft_tokens_batch_mx) # Ensure concatenated draft tokens are ready

                # 2. Evaluate main model on (last accepted token + drafted tokens)
                actor_input_sequence = mx.concatenate([last_generated_tokens[:, None], draft_tokens_batch_mx], axis=1).astype(actor_input_dtype)

                with mx.stream(generation_stream):
                    actor_output = model(actor_input_sequence, cache=model_caches)
                    ref_output = ref_model(actor_input_sequence, cache=ref_caches)

                actor_logits_draft = (actor_output[0] if isinstance(actor_output, tuple) else actor_output)[:, :-1, :]
                ref_logits_draft = (ref_output[0] if isinstance(ref_output, tuple) else ref_output)[:, :-1, :]
                actor_logits_after_draft = (actor_output[0] if isinstance(actor_output, tuple) else actor_output)[:, -1, :]
                ref_logits_after_draft = (ref_output[0] if isinstance(ref_output, tuple) else ref_output)[:, -1, :]

                # Calculate actor log probs for drafted tokens (needed for acceptance check)
                actor_log_probs_draft = selective_softmax(actor_logits_draft, draft_tokens_batch_mx)
                # Calculate ref log probs for drafted tokens (needed later for loss?) No, only need ref log probs for final sequence.

                mx.eval(actor_log_probs_draft, actor_logits_after_draft, ref_logits_after_draft) # Evaluate needed results
                mx.eval([c.state for c in model_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)] +
                        [c.state for c in ref_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)])

                # Apply KV cache quantization for actor/ref models
                if args.kv_bits is not None:
                    current_seq_len_actor_ref = model_caches[0].offset + 1
                    if current_seq_len_actor_ref > args.quantized_kv_start:
                        maybe_quantize_kv_cache(model_caches, args.quantized_kv_start, args.kv_group_size, args.kv_bits)
                        maybe_quantize_kv_cache(ref_caches, args.quantized_kv_start, args.kv_group_size, args.kv_bits)
                        eval_list_quant_actor = [c.state for c in model_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)]
                        eval_list_quant_actor.extend([c.state for c in ref_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)])
                        if eval_list_quant_actor: mx.async_eval(eval_list_quant_actor)

                # 3. Accept/Reject Tokens
                actor_sampled_draft_tokens = []
                actor_sampled_draft_log_probs = []
                for d_idx in range(num_draft):
                    actor_step_logits = actor_logits_draft[:, d_idx, :]
                    sampled_from_actor, sampled_from_actor_log_prob = sample_logits(
                        actor_step_logits, temp=args.sampling_temperature, top_p=args.sampling_top_p,
                        min_p=args.sampling_min_p, min_tokens_to_keep=1,
                    )
                    actor_sampled_draft_tokens.append(sampled_from_actor[:, None])
                    actor_sampled_draft_log_probs.append(sampled_from_actor_log_prob[:, None])

                if actor_sampled_draft_tokens: # Should always be true if num_draft > 0
                    actor_sampled_draft_tokens_mx = mx.concatenate(actor_sampled_draft_tokens, axis=1)
                    actor_sampled_draft_log_probs_mx = mx.concatenate(actor_sampled_draft_log_probs, axis=1)
                    mx.eval(actor_sampled_draft_tokens_mx, actor_sampled_draft_log_probs_mx)

                    mismatch_mask = draft_tokens_batch_mx != actor_sampled_draft_tokens_mx
                    # Mask where sequence already ended shouldn't affect mismatch, as draft=PAD, actor_sample likely not PAD
                    # Ensure cumprod works correctly with bool or requires int
                    cumulative_accepted_mask = mx.cumprod(mx.logical_not(mismatch_mask).astype(mx.int32), axis=1).astype(mx.bool_)
                    num_accepted_batch = mx.sum(cumulative_accepted_mask.astype(mx.int32), axis=1) # Shape (B,)
                    mx.eval(num_accepted_batch)

                    accepted_tokens_padded = draft_tokens_batch_mx * cumulative_accepted_mask.astype(draft_tokens_batch_mx.dtype)
                    accepted_log_probs_padded = actor_log_probs_draft * cumulative_accepted_mask.astype(actor_log_probs_draft.dtype)

                    all_accepted_mask = (num_accepted_batch == num_draft)

                    sampled_after_accepted_all, sampled_after_accepted_all_log_prob = sample_logits(
                        actor_logits_after_draft, temp=args.sampling_temperature, top_p=args.sampling_top_p,
                        min_p=args.sampling_min_p, min_tokens_to_keep=1,
                    )
                    ref_log_prob_after_accepted_all = selective_softmax(ref_logits_after_draft, sampled_after_accepted_all)

                    gather_indices = num_accepted_batch[:, None].astype(mx.int64) # Shape (B, 1)

                    dummy_token_batch = mx.full((total_samples, 1), pad_id, dtype=actor_sampled_draft_tokens_mx.dtype)
                    dummy_log_prob_batch = mx.full((total_samples, 1), 0.0, dtype=actor_sampled_draft_log_probs_mx.dtype)
                    actor_sampled_draft_tokens_padded_gather = mx.concatenate([actor_sampled_draft_tokens_mx, dummy_token_batch], axis=1)
                    actor_sampled_draft_log_probs_padded_gather = mx.concatenate([actor_sampled_draft_log_probs_mx, dummy_log_prob_batch], axis=1)

                    first_rejected_token = mx.take_along_axis(actor_sampled_draft_tokens_padded_gather, gather_indices, axis=1).squeeze(1)
                    first_rejected_log_prob = mx.take_along_axis(actor_sampled_draft_log_probs_padded_gather, gather_indices, axis=1).squeeze(1)

                    dummy_ref_logits = mx.zeros_like(ref_logits_draft[:, 0, :])
                    ref_logits_draft_padded_gather = mx.concatenate([ref_logits_draft, dummy_ref_logits[:, None, :]], axis=1)
                    # Use mx.repeat, assuming it's available based on earlier fix
                    ref_logits_at_mismatch = mx.take_along_axis(
                        ref_logits_draft_padded_gather,
                        mx.repeat(gather_indices[:, :, None], repeats=ref_logits_draft_padded_gather.shape[-1], axis=-1),
                        axis=1
                    ).squeeze(1)
                    first_rejected_ref_log_prob = selective_softmax(ref_logits_at_mismatch, first_rejected_token)

                    next_token_batch = mx.where(all_accepted_mask, sampled_after_accepted_all, first_rejected_token)
                    next_token_log_prob_batch = mx.where(all_accepted_mask, sampled_after_accepted_all_log_prob, first_rejected_log_prob)
                    next_token_ref_log_prob_batch = mx.where(all_accepted_mask, ref_log_prob_after_accepted_all, first_rejected_ref_log_prob) # Needed? No, ref log probs calculated later.

                    mx.eval(next_token_batch, next_token_log_prob_batch) # Evaluate results of this step

                    if eos_id is not None:
                        just_ended_next = mx.equal(next_token_batch, eos_id)
                        # Update ended state: consider sequences that ended *before* this spec dec step
                        # A sequence ends if it was already ended OR the new next_token is EOS
                        ended = mx.logical_or(ended, just_ended_next) # This updates based on the *newly determined* next token

                    # Append accepted + next token for this step
                    step_generated_tokens = mx.full((total_samples, num_draft + 1), pad_id, dtype=draft_tokens_batch_mx.dtype)
                    step_generated_log_probs = mx.full((total_samples, num_draft + 1), 0.0, dtype=actor_log_probs_draft.dtype)
                    # step_generated_ref_log_probs = mx.full((total_samples, num_draft + 1), 0.0, dtype=ref_log_probs_draft.dtype) # Not needed

                    step_generated_tokens[:, :num_draft] = accepted_tokens_padded
                    step_generated_log_probs[:, :num_draft] = accepted_log_probs_padded
                    # step_generated_ref_log_probs[:, :num_draft] = accepted_ref_log_probs_padded # Not needed

                    scatter_indices = num_accepted_batch[:, None].astype(mx.int64) # Shape (B, 1), Col index to update
                    scatter_updates_token = next_token_batch[:, None] # Shape (B, 1), Value to update with
                    scatter_updates_log_prob = next_token_log_prob_batch[:, None] # Shape (B, 1)
                    # scatter_updates_ref_log_prob = next_token_ref_log_prob_batch[:, None] # Not needed

                    # --- BEGIN mx.where based scatter update implementation ---
                    # Assuming mx.scatter, array.at[].set are unavailable

                    _squeezed_col_indices_for_scatter = scatter_indices.squeeze(-1) # Shape (B,)
                    _num_cols_in_target = step_generated_tokens.shape[1] # M = num_draft + 1

                    _col_range_broadcast = mx.arange(_num_cols_in_target)[None, :] # Shape (1, M)
                    _col_indices_to_match = _squeezed_col_indices_for_scatter[:, None] # Shape (B, 1)
                    _update_condition_mask = (_col_range_broadcast == _col_indices_to_match) # Shape (B, M)

                    # Update step_generated_tokens
                    _updates_tokens_squeezed = scatter_updates_token.squeeze(-1)
                    _updates_tokens_broadcast = mx.broadcast_to(_updates_tokens_squeezed[:, None], step_generated_tokens.shape)
                    step_generated_tokens = mx.where(
                        _update_condition_mask,
                        _updates_tokens_broadcast.astype(step_generated_tokens.dtype),
                        step_generated_tokens
                    )

                    # Update step_generated_log_probs
                    _updates_log_probs_squeezed = scatter_updates_log_prob.squeeze(-1)
                    _updates_log_probs_broadcast = mx.broadcast_to(_updates_log_probs_squeezed[:, None], step_generated_log_probs.shape)
                    step_generated_log_probs = mx.where(
                        _update_condition_mask,
                        _updates_log_probs_broadcast.astype(step_generated_log_probs.dtype),
                        step_generated_log_probs
                    )

                    # No need to update step_generated_ref_log_probs here

                    # --- END mx.where based scatter update implementation ---

                    # Append results for this speculative step
                    responses_tokens_list.append(step_generated_tokens)
                    response_log_probs_list.append(step_generated_log_probs)

                    last_generated_tokens = next_token_batch # Input for the next draft step

                    # Update counts - simple increment, will be clamped later
                    generated_counts += (num_accepted_batch + 1)

                    # Rewind cache handling (Warning only, no action)
                    # if mx.any(num_accepted_batch < num_draft).item():
                    #      logger.warning("Batched cache trimming not implemented. Cache size may grow beyond optimal if mismatches occur.")

                else: # No actor sampled tokens generated (should not happen if num_draft > 0)
                     break


        except Exception as e_spec_gen:
            logger.error(f"Speculative generation loop failed: {e_spec_gen}", exc_info=True)
            # Fallback to padding if generation fails
            current_len = sum(t.shape[1] for t in responses_tokens_list) if responses_tokens_list else 0
            remaining_len = max_gen_len - current_len
            if remaining_len > 0:
                logger.warning(f"Padding remaining {remaining_len} steps due to error.")
                if responses_tokens_list:
                     pad_dtype_tok = responses_tokens_list[0].dtype
                     pad_dtype_prob = response_log_probs_list[0].dtype
                     pad_array_tok = mx.full((total_samples, remaining_len), pad_id, dtype=pad_dtype_tok)
                     responses_tokens_list.append(pad_array_tok)
                     zero_array_prob = mx.full((total_samples, remaining_len), 0.0, dtype=pad_dtype_prob)
                     response_log_probs_list.append(zero_array_prob)
                else:
                     pad_array_tok = mx.full((total_samples, remaining_len), pad_id, dtype=prompts_mx_repeated.dtype)
                     responses_tokens_list.append(pad_array_tok)
                     zero_array_prob = mx.full((total_samples, remaining_len), 0.0, dtype=mx.float32)
                     response_log_probs_list.append(zero_array_prob)


    # --- Post-Generation Processing ---
    mx.synchronize() # Ensure all compute is finished
    model.train(); ref_model.train() # Switch back to train mode
    if draft_model: draft_model.train()

    gen_duration = time.monotonic() - gen_start_time
    total_gen_len_from_list = sum(t.shape[1] for t in responses_tokens_list) if responses_tokens_list else 0
    logger.debug(f"Generation loop finished ({total_gen_len_from_list} total generated token positions across batch) in {gen_duration:.2f}s.")

    # Combine generated tokens
    if not responses_tokens_list:
        logger.warning("No response tokens generated.")
        generated_seq_len = 0
        responses_mx = mx.zeros((total_samples, 0), dtype=mx.int32)
    else:
        try:
            responses_mx = mx.concatenate(responses_tokens_list, axis=1)
            generated_seq_len = responses_mx.shape[1]
            # Truncate if speculative decoding overshot max_gen_len
            if generated_seq_len > max_gen_len:
                 logger.warning(f"Generated sequence length ({generated_seq_len}) exceeded max_gen_len ({max_gen_len}). Truncating.")
                 responses_mx = responses_mx[:, :max_gen_len]
                 generated_seq_len = responses_mx.shape[1]
            mx.eval(responses_mx)
        except Exception as e_concat:
            logger.error(f"Failed to concatenate response tokens: {e_concat}", exc_info=True)
            generated_seq_len = 0
            responses_mx = mx.zeros((total_samples, 0), dtype=mx.int32)

    # Decode for reward calculation
    decoded_responses = ["[decode error]"] * total_samples # Default
    if generated_seq_len > 0:
        try:
            decoded_responses = tokenizer.batch_decode(
                responses_mx.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True,
            )
        except Exception as e_decode:
            logger.error(f"Failed to decode responses: {e_decode}", exc_info=True)
            # Keep default error message

    # Calculate Rewards
    rewards_list_total, rewards_list_format, rewards_list_content_combined = [], [], []
    for i in range(total_samples):
        resp_text = decoded_responses[i]
        prompt_index = i // num_samples_per_prompt
        ref_ans = ""
        if prompt_index < len(prompts_data):
            ref_ans = prompts_data[prompt_index].get("ref_answer_str", "")
        else:
            logger.error(f"Out of bounds prompt index {prompt_index} during reward calc for sample {i}.")

        cleaned_resp = resp_text
        if pad_token_str: cleaned_resp = re.sub(rf"(?:{re.escape(pad_token_str)})+$", "", cleaned_resp).rstrip()
        if eos_token_str:
            eos_pos = cleaned_resp.find(eos_token_str)
            if eos_pos != -1: cleaned_resp = cleaned_resp[:eos_pos].rstrip()

        total_reward, fmt_rew, content_rew_combined = 0.0, 0.0, 0.0
        try:
            if cleaned_resp.startswith("[") and "error]" in cleaned_resp: pass
            elif not ref_ans and args.reward_content_weight > 0:
                total_reward, fmt_rew, _ = calculate_total_reward(cleaned_resp, "", reward_config, args.reward_format_weight, 0.0, args.reward_content_type)
                content_rew_combined = 0.0
                total_reward = fmt_rew * args.reward_format_weight
            else:
                total_reward, fmt_rew, content_rew_combined = calculate_total_reward(
                    cleaned_resp, ref_ans or "", reward_config, args.reward_format_weight,
                    args.reward_content_weight, args.reward_content_type
                )
        except Exception as e_reward:
            logger.error(f"Reward calculation failed for sample {i} (prompt_idx={prompt_index}): {e_reward}", exc_info=False)

        rewards_list_total.append(total_reward)
        rewards_list_format.append(fmt_rew)
        rewards_list_content_combined.append(content_rew_combined)

    rewards = mx.array(rewards_list_total, dtype=mx.float32)
    rewards_raw_format = mx.array(rewards_list_format, dtype=mx.float32)
    rewards_raw_content_combined = mx.array(rewards_list_content_combined, dtype=mx.float32)
    mx.eval(rewards, rewards_raw_format, rewards_raw_content_combined)

    # Calculate Advantages
    advantages = mx.zeros_like(rewards[:, None])
    if num_prompts > 0 and num_samples_per_prompt > 1:
        try:
            rewards_per_prompt = rewards.reshape(num_prompts, num_samples_per_prompt)
            baseline = mx.mean(rewards_per_prompt, axis=1, keepdims=True)
            variance = mx.var(rewards_per_prompt, axis=1, keepdims=True)
            std_dev = mx.sqrt(variance + args.advantage_epsilon)
            advantages_per_prompt = (rewards_per_prompt - baseline) / (std_dev + args.advantage_epsilon)
            advantages = advantages_per_prompt.reshape(total_samples, 1)
            mx.eval(advantages)
        except Exception as e_adv:
            logger.error(f"Advantage calculation failed: {e_adv}. Using raw rewards as advantages.", exc_info=True)
            advantages = rewards[:, None] # Fallback
            mx.eval(advantages)
    elif num_prompts > 0:
        logger.debug("Using rewards as advantages (num_samples_per_prompt <= 1).")
        advantages = rewards[:, None]
        mx.eval(advantages)

    # --- Calculate Reference Log Probabilities and Response Mask ---
    full_sequence = mx.concatenate([prompts_mx_repeated, responses_mx], axis=1)
    full_seq_len = full_sequence.shape[1]
    mx.eval(full_sequence)

    ref_log_probs_response = mx.zeros((total_samples, generated_seq_len), dtype=mx.float32)
    response_mask_loss = mx.zeros((total_samples, generated_seq_len), dtype=mx.float32)

    if generated_seq_len > 0 and full_seq_len == (max_prompt_len_batch + generated_seq_len):
        try:
            # Determine ref_model dtype
            try: ref_model_param_dtype = next(tree_flatten(ref_model.parameters()))[1].dtype
            except: ref_model_param_dtype = mx.float32 # Fallback

            ref_attention_mask_4d = _create_4d_attention_mask(full_sequence, pad_id, dtype=mx.bfloat16)
            with mx.stream(generation_stream):
                ref_output_full = ref_model(full_sequence.astype(mx.int64), mask=ref_attention_mask_4d)

            ref_logits_full = ref_output_full[0] if isinstance(ref_output_full, tuple) else ref_output_full
            ref_logits_full = ref_logits_full.astype(mx.float32)
            mx.eval(ref_logits_full)

            logits_start_idx = max_prompt_len_batch - 1
            logits_end_idx = full_seq_len - 1 # Exclusive index for slicing
            logits_for_ref_response = ref_logits_full[:, logits_start_idx : logits_end_idx, :]

            target_tokens_for_ref = responses_mx

            if logits_for_ref_response.shape[1] == target_tokens_for_ref.shape[1]:
                ref_log_probs_response = selective_softmax(logits_for_ref_response, target_tokens_for_ref)
                response_mask_loss = (target_tokens_for_ref != pad_id).astype(mx.float32)
                mx.eval(ref_log_probs_response, response_mask_loss)
            else:
                logger.error(
                    f"Ref log prob shape mismatch: Logits shape {logits_for_ref_response.shape} (dim 1 length {logits_for_ref_response.shape[1]}) "
                    f"vs Targets shape {target_tokens_for_ref.shape} (dim 1 length {target_tokens_for_ref.shape[1]})."
                )
        except Exception as e_ref_full_pass:
            logger.error(f"Ref model full pass calculation failed: {e_ref_full_pass}", exc_info=True)
    elif generated_seq_len > 0:
        logger.error(
            f"Cannot calculate ref log probs due to sequence length mismatch: "
            f"full_seq_len={full_seq_len}, max_prompt_len_batch={max_prompt_len_batch}, generated_seq_len={generated_seq_len}"
        )

    mx.synchronize() # Final sync before returning data

    batch_data = {
        "tokens": full_sequence, # Shape (B, prompt_len + gen_len)
        "response_mask": response_mask_loss, # Shape (B, gen_len)
        "advantages": advantages, # Shape (B, 1)
        "ref_log_probs": ref_log_probs_response, # Shape (B, gen_len)
    }

    # Use .tolist() for safe conversion before np.mean, check size
    avg_reward_scalar = float(np.mean(rewards.tolist())) if rewards.size > 0 else 0.0
    avg_format_reward_raw = float(np.mean(rewards_raw_format.tolist())) if rewards_raw_format.size > 0 else 0.0
    avg_content_reward_raw_combined = float(np.mean(rewards_raw_content_combined.tolist())) if rewards_raw_content_combined.size > 0 else 0.0

    raw_reward_components = {
        "raw_format": avg_format_reward_raw,
        "raw_content_combined": avg_content_reward_raw_combined,
    }

    return batch_data, avg_reward_scalar, raw_reward_components
# FIX END: generate_rollouts_for_batch method


# --- Dataset Handling ---
# --- Dataset Handling (Same as provided, using datasets library) ---
def get_dataset(args: TrainingArgs) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    """Loads training and validation datasets from local path or Hugging Face Hub."""
    train_dset, val_dset = None, None
    logger = logging.getLogger(__name__)  # Use existing logger

    if not DATASETS_AVAILABLE:
        error_msg = "The 'datasets' library is required to load data. Please install it (`pip install datasets`)."
        logger.critical(error_msg)
        raise ImportError(error_msg)

    # --- Load Training Dataset ---
    if args.train_dataset_path:
        logger.info(
            f"Attempting to load train dataset from local path: {args.train_dataset_path}"
        )
        train_path = Path(args.train_dataset_path)
        if not train_path.is_file():
            logger.critical(f"Train dataset file not found: {train_path}")
            raise FileNotFoundError(f"Train dataset file not found: {train_path}")
        try:
            train_dset = load_dataset("json", data_files=str(train_path), split="train")
            logger.info(
                f"Successfully loaded local train dataset ({len(train_dset):,} examples) from {train_path}"
            )
        except Exception as e:
            logger.critical(
                f"Failed to load local train dataset '{args.train_dataset_path}': {e}",
                exc_info=True,
            )
            raise  # Halt execution
    elif args.dataset_name:
        logger.info(
            f"Attempting to load train dataset from Hub: {args.dataset_name}, config: {args.dataset_config}, split: {args.dataset_train_split}"
        )
        try:
            train_dset = load_dataset(
                args.dataset_name,
                args.dataset_config,
                split=args.dataset_train_split,
                trust_remote_code=True,
            )
            logger.info(
                f"Successfully loaded train dataset '{args.dataset_name}' split '{args.dataset_train_split}' ({len(train_dset):,} examples) from Hub."
            )
        except Exception as e:
            logger.critical(
                f"Failed to load train dataset '{args.dataset_name}' from Hub: {e}",
                exc_info=True,
            )
            raise  # Halt execution
    else:
        error_msg = "No training data source specified. Please provide --train-dataset-path or --dataset-name."
        logger.critical(error_msg)
        raise ValueError(error_msg)

    # --- Load Validation Dataset ---
    if args.val_dataset_path:
        logger.info(
            f"Attempting to load validation dataset from local path: {args.val_dataset_path}"
        )
        val_path = Path(args.val_dataset_path)
        if not val_path.is_file():
            logger.warning(
                f"Local validation dataset file not found: {val_path}. Skipping validation set."
            )
        else:
            try:
                # Note: 'train' split is commonly used for single-file local json datasets
                val_dset = load_dataset("json", data_files=str(val_path), split="train")
                logger.info(
                    f"Successfully loaded local validation dataset ({len(val_dset):,} examples) from {val_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load local validation dataset '{args.val_dataset_path}': {e}. Proceeding without validation set.",
                    exc_info=False,
                )
                val_dset = None
    elif args.dataset_name and args.dataset_val_split:
        logger.info(
            f"Attempting to load validation dataset from Hub: {args.dataset_name}, config: {args.dataset_config}, split: {args.dataset_val_split}"
        )
        try:
            val_dset = load_dataset(
                args.dataset_name,
                args.dataset_config,
                split=args.dataset_val_split,
                trust_remote_code=True,
            )
            logger.info(
                f"Successfully loaded validation dataset '{args.dataset_name}' split '{args.dataset_val_split}' ({len(val_dset):,} examples) from Hub."
            )
        except Exception as e:
            logger.warning(
                f"Failed to load validation dataset '{args.dataset_name}' split '{args.dataset_val_split}' from Hub: {e}. Proceeding without validation set.",
                exc_info=False,
            )
            val_dset = None
    else:
        logger.info(
            "No validation dataset path or Hub split specified. Skipping validation set."
        )

    # --- Validate Columns ---
    required_keys = {args.dataset_prompt_key, args.dataset_answer_key}
    if train_dset:
        missing_train_keys = required_keys - set(train_dset.column_names)
        if missing_train_keys:
            logger.critical(
                f"Train dataset missing required keys: {missing_train_keys}. Available columns: {train_dset.column_names}"
            )
            raise ValueError(
                f"Train dataset missing required keys: {missing_train_keys}"
            )

    if val_dset:
        missing_val_keys = required_keys - set(val_dset.column_names)
        if missing_val_keys:
            logger.warning(
                f"Validation dataset missing required keys for reward calculation: {missing_val_keys}. Available columns: {val_dset.column_names}. Evaluation rewards might be inaccurate."
            )

    return train_dset, val_dset



# --- Metrics Logger ---
class MetricsLogger:
    """Logs metrics to a CSV file with thread safety."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._file: Optional[Any] = None
        self._writer: Optional[csv.DictWriter] = None
        self._headers: List[str] = []
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._logged_file_closed_warning = False

        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            needs_header = not self.file_path.exists() or self.file_path.stat().st_size == 0
            self._file = open(self.file_path, "a", newline="", encoding="utf-8")
            self.logger.info(f"Metrics CSV {'created' if needs_header else 'opened (append mode)'}: {self.file_path}")
        except OSError as e:
            self.logger.error(f"Failed to open metrics file {self.file_path} for appending: {e}. CSV logging will be disabled.", exc_info=True)
            self._file = None

    def log(self, metrics: Dict[str, Any]):
        if self._file is None or self._file.closed:
            if not self._logged_file_closed_warning:
                self.logger.warning(f"Metrics file '{self.file_path.name}' is not open or has been closed. Cannot log metrics.")
                self._logged_file_closed_warning = True
            return

        loggable: Dict[str, Union[str, int, float, bool]] = {}
        for key, value in metrics.items():
            if isinstance(value, (mx.array, np.ndarray)):
                try:
                    if value.size == 1:
                        item = value.item()
                        loggable[key] = item if isinstance(item, (int, float, bool)) else str(item)
                    else:
                        loggable[key] = str(value.tolist())
                except Exception as e:
                    loggable[key] = f"[Array conv error: {e}]"
            elif isinstance(value, (int, float, bool, str)):
                loggable[key] = value
            elif value is None:
                loggable[key] = ""
            else:
                try:
                    loggable[key] = str(value)
                except Exception as e:
                    loggable[key] = f"[str conv error: {e}]"

        if not loggable:
            self.logger.debug("No loggable metrics provided.")
            return

        with self._lock:
            try:
                current_headers = sorted(loggable.keys())
                if self._writer is None or self._headers != current_headers:
                    if self._writer is not None:
                        self.logger.warning(f"CSV headers changed. Old: {self._headers}, New: {current_headers}. Rewriting header.")
                        self._file.flush()
                        self._file.close()
                        self._file = open(self.file_path, "w", newline="", encoding="utf-8") # TRUNCATE
                        self._file.close()
                        self._file = open(self.file_path, "a", newline="", encoding="utf-8") # APPEND

                    self._headers = current_headers
                    self._writer = csv.DictWriter(self._file, fieldnames=self._headers, extrasaction="ignore")
                    if self._file.tell() == 0:
                        self.logger.debug(f"Writing CSV header: {self._headers}")
                        self._writer.writeheader()

                self._writer.writerow(loggable)
                self._file.flush()
            except Exception as e:
                self.logger.error(f"Error writing metrics to CSV file '{self.file_path.name}': {e}", exc_info=True)

    def close(self):
        with self._lock:
            if self._file and not self._file.closed:
                try:
                    self._file.flush()
                    self._file.close()
                    self.logger.info(f"Closed metrics file: {self.file_path}")
                    self._file = None
                    self._writer = None
                    self._headers = []
                except Exception as e:
                    self.logger.error(f"Error closing metrics file '{self.file_path.name}': {e}")
            elif self._file is None:
                self.logger.debug("Metrics file was already closed or never opened.")


# --- Checkpoint Loading/Saving ---

def save_checkpoint(
    output_dir: Path,
    reason: str,
    step: int,
    num_updates: int,
    actor_model: nn.Module,
    optimizer: optim.Optimizer,
    tokenizer: Any,  # Use Any if TokenizerWrapper type hint isn't available
    training_args: Any,  # Use Any if TrainingArgs type hint isn't available
    model_config: Dict,
    create_latest_symlink: bool = True,
    keep_last: Optional[int] = 5,
) -> None:
    """
    Saves model/optimizer/tokenizer/state using safe commit utilities and rotates old checkpoints.
    Relies on utility functions like _save_and_commit being available.

    Args:
        output_dir: Base directory for checkpoints.
        reason: String identifier for the save reason.
        step: Current global training step.
        num_updates: Number of optimizer updates (used for naming and sorting).
        actor_model: The MLX model.
        optimizer: The MLX optimizer.
        tokenizer: Tokenizer object with save_pretrained method.
        training_args: Object containing training configuration (e.g., use_lora).
        model_config: Dictionary with model architecture config.
        create_latest_symlink: If True, create/update 'latest' symlink.
        keep_last: If > 0, keep only this many recent checkpoints. None or 0 disables.
    """
    global mlx_rng_key

    # --- Checkpoint Naming and Path ---
    safe_reason = re.sub(r"[^\w\-]+", "_", reason)
    # Add a timestamp for uniqueness in case multiple saves happen in the same second
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"checkpoint_{timestamp}_{safe_reason}_update_{num_updates}"
    save_path = output_dir / checkpoint_name
    logger.info(
        f"Attempting to save checkpoint: '{checkpoint_name}' (Reason: {reason}, Update: {num_updates})"
    )
    save_start_time = time.monotonic()
    save_successful = False

    # --- Ensure Utility Functions are Available (Optional Check) ---
    # Check if _save_and_commit and _save_directory_and_commit exist (part of llama_rl.utils)
    if "_save_and_commit" not in globals() or not callable(
        globals()["_save_and_commit"]
    ):
        logger.error(
            "Required utility function '_save_and_commit' not found or not callable. Cannot save checkpoint."
        )
        return  # Abort save
    if "_save_directory_and_commit" not in globals() or not callable(
        globals()["_save_directory_and_commit"]
    ):
        logger.error(
            "Required utility function '_save_directory_and_commit' not found or not callable. Cannot save checkpoint."
        )
        return  # Abort save

    try:
        # --- Create Checkpoint Directory ---
        save_path.mkdir(parents=True, exist_ok=True)

        # --- Evaluate Tensors ---
        # Evaluate model and optimizer state to ensure they are ready for saving
        logger.debug("Evaluating model parameters and optimizer state before saving...")
        mx.eval(actor_model.parameters())
        mx.eval(optimizer.state)
        logger.debug("Evaluation complete.")

        # --- Save Model Weights ---
        weights_target_path: Path
        use_lora = getattr(training_args, "use_lora", False)
        if use_lora:
            # Get only trainable parameters (LoRA weights)
            lora_params = dict(tree_flatten(actor_model.trainable_parameters()))
            if lora_params:
                weights_target_path = save_path / "adapters.safetensors"
                logger.debug(f"Saving LoRA adapters ({len(lora_params)} tensors)...")
                _save_and_commit(
                    temp_prefix="lora_adapter_",
                    target_path=str(weights_target_path),
                    save_fn=lambda tmp_p: mx.save_safetensors(tmp_p, lora_params),
                )
                logger.debug(f"Saved LoRA adapters to {weights_target_path.name}.")
            else:
                logger.warning(
                    "LoRA enabled, but no trainable parameters found. Skipping adapter save."
                )
        else:
            # Get all model parameters
            model_params = dict(tree_flatten(actor_model.parameters()))
            weights_target_path = save_path / "model.safetensors"
            logger.debug(f"Saving full model weights ({len(model_params)} tensors)...")
            _save_and_commit(
                temp_prefix="model_weights_",
                target_path=str(weights_target_path),
                save_fn=lambda tmp_p: mx.save_safetensors(tmp_p, model_params),
            )
            logger.debug(f"Saved full model weights to {weights_target_path.name}.")

        # --- Save Model Configuration ---
        config_target_path = save_path / "config.json"
        # Check if save_config from mlx_lm.utils is available, otherwise use json.dump
        if "save_config" in sys.modules.get("mlx_lm.utils", {}).__dict__:
            logger.debug(f"Saving model config using mlx_lm.utils.save_config...")
            try:
                _save_and_commit(
                    temp_prefix="config_",
                    target_path=str(config_target_path),
                    save_fn=lambda tmp_p: save_config(
                        model_config, tmp_p
                    ),  # Assumes mlx_lm.utils.save_config exists and takes path
                )
                logger.debug(
                    f"Saved model config using save_config to {config_target_path.name}."
                )
            except Exception as e_cfg_helper:
                logger.warning(
                    f"Using save_config helper failed ({e_cfg_helper}), falling back to manual JSON dump."
                )
                _save_and_commit(
                    temp_prefix="config_json_",
                    target_path=str(config_target_path),
                    save_fn=lambda tmp_p: json.dump(
                        model_config, open(tmp_p, "w"), indent=4
                    ),
                )
                logger.debug(
                    f"Saved model config using manual JSON dump to {config_target_path.name}."
                )
        else:
            logger.debug(
                "mlx_lm.utils.save_config function not found, using manual JSON dump."
            )
            _save_and_commit(
                temp_prefix="config_json_",
                target_path=str(config_target_path),
                save_fn=lambda tmp_p: json.dump(
                    model_config, open(tmp_p, "w"), indent=4
                ),
            )
            logger.debug(
                f"Saved model config using manual JSON dump to {config_target_path.name}."
            )

        # --- Save Optimizer State ---
        optimizer_target_path = save_path / "optimizer.safetensors"
        optimizer_state_flat = dict(tree_flatten(optimizer.state))
        logger.debug(f"Saving optimizer state ({len(optimizer_state_flat)} tensors)...")
        _save_and_commit(
            temp_prefix="optimizer_",
            target_path=str(optimizer_target_path),
            save_fn=lambda tmp_p: mx.save_safetensors(tmp_p, optimizer_state_flat),
        )
        logger.debug(f"Saved optimizer state to {optimizer_target_path.name}.")

        # --- Prepare and Save Training State (including RNG) ---
        if mlx_rng_key is not None:
            mx.eval(mlx_rng_key)  # Evaluate RNG key before converting to list
            # Convert MLX uint32 array to a list of Python ints for JSON serialization
            mlx_rng_list = [int(s) for s in mlx_rng_key.tolist()]
        else:
            mlx_rng_list = None
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "mlx": mlx_rng_list,
        }

        # Attempt to serialize training_args, skipping non-serializable parts if necessary
        training_args_dict = {}
        try:
            # Use asdict to get a dict representation of the dataclass
            args_dict_full = asdict(training_args)
            # Filter out potential non-serializable types if asdict fails or contains them
            # (e.g., file handles, complex objects unless handled by default=str)
            # Use json.dumps with default=str as a robust first step
            json.dumps(args_dict_full, default=str)  # Test serialization
            training_args_dict = args_dict_full
        except Exception as e_ser:
            logger.warning(
                f"Could not serialize training_args fully ({e_ser}). Saving a simplified snapshot."
            )
            # Fallback to a simpler representation
            if hasattr(training_args, "__dict__"):
                training_args_dict = vars(training_args)
            elif isinstance(training_args, dict):
                training_args_dict = training_args
            else:
                training_args_dict = {"info": "Could not serialize training_args"}

        training_state = {
            "global_step": step,
            "num_updates": num_updates,
            "use_lora": use_lora,
            "rng_state": rng_state,
            "training_args_snapshot": training_args_dict,
        }
        training_state_target_path = save_path / "training_state.json"
        logger.debug("Saving training state...")
        _save_and_commit(
            temp_prefix="training_state_",
            target_path=str(training_state_target_path),
            # Use default=str to handle potentially non-JSON serializable types (like Path)
            save_fn=lambda tmp_p: json.dump(
                training_state, open(tmp_p, "w"), indent=4, default=str
            ),
        )
        logger.debug(f"Saved training state to {training_state_target_path.name}.")

        # --- Save Tokenizer Files (using directory commit) ---
        if hasattr(tokenizer, "save_pretrained") and callable(
            tokenizer.save_pretrained
        ):
            logger.debug(f"Saving tokenizer files to {save_path.name}...")

            # With this direct approach:
            try:
                logger.debug(f"Saving tokenizer files directly to {save_path.name}...")
                tokenizer.save_pretrained(save_path)
                logger.debug(f"Saved tokenizer files directly into {save_path.name}.")
            except Exception as e_tok:
                logger.error(f"Failed to save tokenizer files: {e_tok}", exc_info=True)

            logger.debug(f"Saved tokenizer files into {save_path.name}.")
        else:
            logger.error(
                "Tokenizer object missing callable 'save_pretrained' method.",
                exc_info=False,
            )
            # Depending on requirements, you might want to raise an error here
            # raise AttributeError("Tokenizer missing save_pretrained method.")

        # --- Create/Update 'latest' Symlink ---
        if create_latest_symlink:
            latest_symlink_path = output_dir / "latest"
            try:
                # Use relative path for the symlink target
                target_path_str = os.path.relpath(save_path, output_dir)
                if latest_symlink_path.is_symlink():
                    logger.debug(
                        f"Removing existing 'latest' symlink: {latest_symlink_path}"
                    )
                    os.unlink(latest_symlink_path)
                elif latest_symlink_path.exists():
                    logger.warning(
                        f"Path '{latest_symlink_path}' exists but is not a symlink. Cannot create 'latest' symlink."
                    )
                # Only create symlink if it doesn't exist after checking/removing
                if not latest_symlink_path.exists():
                    os.symlink(
                        target_path_str, latest_symlink_path, target_is_directory=True
                    )
                    logger.info(
                        f"Updated 'latest' symlink to point to '{target_path_str}'"
                    )
            except OSError as e_symlink:
                logger.error(f"Failed to create/update 'latest' symlink: {e_symlink}")
            except Exception as e_symlink_other:
                logger.error(
                    f"Unexpected error during symlink creation: {e_symlink_other}"
                )

        # --- Final Success Log ---
        save_duration = time.monotonic() - save_start_time
        logger.info(
            f"Checkpoint saved successfully: '{checkpoint_name}' ({save_duration:.2f}s)"
        )
        save_successful = True

    except Exception as e:
        # --- General Error During Saving ---
        logger.error(
            f"Failed to save checkpoint '{checkpoint_name}': {e}", exc_info=True
        )
        if save_path.exists():
            logger.warning(
                f"Attempting to remove partially saved checkpoint directory: {save_path.name}"
            )
            try:
                shutil.rmtree(save_path)
                logger.info(
                    f"Successfully removed partial checkpoint: {save_path.name}"
                )
            except OSError as rm_err:
                logger.error(
                    f"Failed to remove partial checkpoint directory {save_path.name}: {rm_err}"
                )
        save_successful = False

    # --- Checkpoint Rotation Logic ---
    if save_successful and keep_last is not None and keep_last > 0:
        # (Rotation logic remains the same as before)
        logger.debug(f"Running checkpoint rotation: keep_last={keep_last}")
        try:
            # Update pattern to match the new naming format
            checkpoint_pattern = re.compile(r"^checkpoint_\d{8}_\d{6}_.+_update_(\d+)$")
            checkpoints: List[Tuple[int, Path]] = []
            for item in output_dir.iterdir():
                if item.is_dir():
                    match = checkpoint_pattern.match(item.name)
                    if match:
                        try:
                            checkpoints.append((int(match.group(1)), item))
                        except ValueError:
                            logger.warning(
                                f"Could not parse update number from: {item.name}"
                            )
            checkpoints.sort(
                key=lambda x: x[0], reverse=True
            )  # Sort by update number descending

            if len(checkpoints) > keep_last:
                checkpoints_to_delete = checkpoints[keep_last:]
                logger.info(
                    f"Found {len(checkpoints)} checkpoints. Keeping {keep_last}, deleting {len(checkpoints_to_delete)}."
                )
                # Ensure the 'latest' symlink target is not deleted if it's one of the older ones (shouldn't happen with correct save/symlink)
                latest_symlink_target = None
                latest_symlink_path = output_dir / "latest"
                if latest_symlink_path.is_symlink():
                    try:
                        latest_symlink_target = latest_symlink_path.resolve()
                    except OSError:
                        pass  # Ignore errors resolving symlink

                for update_num, path_to_delete in checkpoints_to_delete:
                    if (
                        latest_symlink_target
                        and path_to_delete.resolve() == latest_symlink_target
                    ):
                        logger.warning(
                            f"Skipping deletion of 'latest' checkpoint target: {path_to_delete.name} (Update: {update_num})"
                        )
                        continue
                    # Also ensure we don't delete the checkpoint we just saved if keep_last is very small (should be sorted out by logic, but safety check)
                    if path_to_delete.resolve() == save_path.resolve():
                        logger.warning(
                            f"Skipping deletion of current checkpoint: {path_to_delete.name} (Update: {update_num})"
                        )
                        continue

                    logger.info(
                        f"Deleting old checkpoint: {path_to_delete.name} (Update: {update_num})"
                    )
                    try:
                        shutil.rmtree(path_to_delete)
                    except OSError as e_rm:
                        logger.error(
                            f"Failed to delete checkpoint {path_to_delete.name}: {e_rm}"
                        )
            else:
                logger.debug(
                    f"Found {len(checkpoints)} checkpoints. No rotation needed."
                )
        except Exception as e_rotate:
            logger.error(f"Error during checkpoint rotation: {e_rotate}", exc_info=True)


def load_checkpoint(
    resume_path: Path,
    training_args: TrainingArgs,
    actor_model: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[int, int, bool]:
    """Loads state from a checkpoint directory. Modifies model and optimizer in-place.
    Returns start_step, start_updates, rng_loaded_flag.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Attempting to load checkpoint from: [link=file://{resume_path.resolve()}]{resume_path}[/]"
    )
    load_start_time = time.monotonic()
    rng_loaded = False  # Flag to indicate if RNG state was successfully restored

    # Define expected file paths
    state_path = resume_path / "training_state.json"
    config_path = resume_path / "config.json"  # Standard config name
    # Weights file path depends on LoRA state
    lora_weights_path = resume_path / "adapters.safetensors"
    full_weights_path = resume_path / "model.safetensors"
    optimizer_state_path = resume_path / "optimizer.safetensors"
    # Tokenizer should be loadable from the resume_path directory

    # --- Basic File Existence Checks ---
    required_files = [state_path, config_path]
    missing_files = [f.name for f in required_files if not f.exists()]
    if missing_files:
        logger.critical(
            f"Checkpoint load failed: Missing essential files in '{resume_path}': {', '.join(missing_files)}"
        )
        raise FileNotFoundError(f"Missing checkpoint files: {', '.join(missing_files)}")

    # Check for weights file based on expected LoRA state from training_args
    expected_weights_path = (
        lora_weights_path if training_args.use_lora else full_weights_path
    )
    if not expected_weights_path.exists():
        # Log the path checked
        logger.critical(
            f"Checkpoint load failed: Expected weights file '{expected_weights_path.name}' not found in '{resume_path}'."
        )
        raise FileNotFoundError(
            f"Expected weights file '{expected_weights_path.name}' not found."
        )

    try:
        # --- Load Training State First ---
        logger.debug(f"Loading training state from {state_path.name}...")
        with open(state_path, "r") as f:
            train_state = json.load(f)

        start_step = train_state.get("global_step", 0)
        start_updates = train_state.get("num_updates", 0)
        saved_use_lora = train_state.get("use_lora", False)  # Default False if missing
        logger.info(
            f"Checkpoint state loaded: Step={start_step}, Updates={start_updates}, Saved LoRA State={saved_use_lora}"
        )

        # --- Verify LoRA Compatibility ---
        if training_args.use_lora != saved_use_lora:
            logger.error(f"[bold red]LoRA Configuration Mismatch![/]")
            logger.error(f"  Checkpoint was saved with use_lora = {saved_use_lora}")
            logger.error(
                f"  Current run is configured with use_lora = {training_args.use_lora}"
            )
            raise ValueError(
                "Cannot resume: LoRA configuration mismatch between checkpoint and current settings."
            )
        logger.info(f"Checkpoint LoRA mode ({saved_use_lora}) matches current setting.")

        # Note: Applying LoRA layers (if use_lora=True) should happen *before* calling load_checkpoint.
        # The main() logic is responsible for creating the actor_model instance and applying LoRA
        # before passing it here. This function then *loads the weights* into the already structured model.

        # --- Load Model Weights ---
        weights_file = expected_weights_path  # Already checked existence above
        logger.debug(f"Loading model weights from {weights_file.name}...")
        weights = mx.load(str(weights_file))

        try:
            # Update the model parameters with the loaded weights
            # tree_unflatten expects a list of (path, array) tuples
            actor_model.update(tree_unflatten(list(weights.items())))
            logger.debug(
                f"Loaded and applied {len(weights)} weights from {weights_file.name}."
            )
        except Exception as e_weights_apply:
            logger.error(
                f"Failed to apply loaded weights: {e_weights_apply}. Check model structure and checkpoint format compatibility."
            )
            raise  # Re-raise as applying weights is critical

        mx.eval(actor_model.parameters())  # Evaluate model parameters after loading

        # --- Load Optimizer State ---
        if optimizer_state_path.exists():
            logger.debug(f"Loading optimizer state from {optimizer_state_path.name}...")
            try:
                opt_state_loaded_flat = mx.load(str(optimizer_state_path))
                # Unflatten the loaded state into the optimizer's state structure
                # This requires the optimizer's state structure to match what was saved.
                # If the model structure changed trainable parameters (e.g., different LoRA rank),
                # this might fail.
                optimizer.state = tree_unflatten(list(opt_state_loaded_flat.items()))
                mx.eval(optimizer.state)  # Evaluate loaded state
                logger.debug("Optimizer state loaded and evaluated.")
            except Exception as e_opt_load:
                # Key mismatches or other errors during unflattening
                logger.error(
                    f"Failed to load or apply optimizer state: {e_opt_load}. Optimizer state may be incorrect or reset.",
                    exc_info=True,
                )
                # Continue, but the optimizer state is not restored from checkpoint
        else:
            logger.warning(
                f"Optimizer state file not found ({optimizer_state_path.name}). Optimizer state will remain as initialized."
            )

        # --- Load RNG State ---
        global mlx_rng_key  # Ensure we modify the global key
        if "rng_state" in train_state and train_state["rng_state"]:
            logger.debug("Restoring RNG states...")
            try:
                rng_state_loaded = train_state["rng_state"]
                # Restore Python RNG state
                if (
                    "python" in rng_state_loaded
                    and rng_state_loaded["python"] is not None
                ):
                    random.setstate(tuple(rng_state_loaded["python"]))
                    logger.debug("Python RNG state restored.")
                else:
                    logger.debug("Python RNG state not found in checkpoint.")

                # Restore NumPy RNG state
                if (
                    "numpy" in rng_state_loaded
                    and rng_state_loaded["numpy"] is not None
                ):
                    np_state_list = rng_state_loaded["numpy"]
                    # NumPy state tuple format is complex and can change slightly,
                    # but typically involves a string name, a list/array of keys (uint32),
                    # a position, a has_gauss flag, and a cached_gaussian value.
                    # Need to ensure types match expected np.random.set_state format.
                    if (
                        isinstance(np_state_list, (list, tuple))
                        and len(np_state_list) >= 5
                    ):
                        try:
                            np_state_tuple = (
                                np_state_list[0],
                                np.array(
                                    np_state_list[1], dtype=np.uint32
                                ),  # Key array (ensure dtype)
                                int(np_state_list[2]),  # pos
                                int(
                                    np_state_list[3]
                                ),  # has_gauss (should be int 0 or 1)
                                float(np_state_list[4]),  # cached_gaussian
                            )
                            np.random.set_state(np_state_tuple)
                            logger.debug("NumPy RNG state restored.")
                        except Exception as e_np_set:
                            logger.warning(
                                f"Failed to set NumPy RNG state from saved format: {e_np_set}. Skipping NumPy RNG restoration.",
                                exc_info=False,
                            )
                            rng_loaded = False  # Indicate partial failure
                    else:
                        logger.warning(
                            f"NumPy RNG state format in checkpoint not recognized. Skipping restoration. State: {np_state_list}"
                        )
                        rng_loaded = False  # Indicate partial failure
                else:
                    logger.debug("NumPy RNG state not found in checkpoint.")

                # Restore MLX RNG state
                if "mlx" in rng_state_loaded and rng_state_loaded["mlx"] is not None:
                    mlx_state_list = rng_state_loaded["mlx"]
                    if isinstance(mlx_state_list, list):
                        # Convert list of ints back to MLX uint32 array key
                        mlx_rng_key = mx.array(mlx_state_list, dtype=mx.uint32)
                        mx.random.set_key(mlx_rng_key)  # Set the global key
                        logger.debug("MLX RNG state restored.")
                        rng_loaded = True  # MLX state restored successfully
                    else:
                        logger.warning(
                            f"Unrecognized MLX RNG state format: {type(mlx_state_list)}. Skipping MLX RNG restoration."
                        )
                        # If only MLX fails but others succeed, rng_loaded could be False depending on logic flow
                        # Let's ensure rng_loaded is True only if ALL standard states are restored
                        rng_loaded = False
                else:
                    logger.debug("MLX RNG state not found in checkpoint.")

                # Final check if *all* states were successfully loaded (Python, NumPy, MLX)
                # This simple check relies on the debug logs indicating success for each.
                # A more robust check would track success for each type explicitly.
                # For now, if MLX loaded, we assume other standard ones did too unless warnings occurred.
                if rng_loaded:
                    logger.info("Restored RNG states successfully.")

            except Exception as e_rng:
                logger.error(
                    f"Failed to restore RNG states: {e_rng}. Initial seeding may apply.",
                    exc_info=False,
                )
                rng_loaded = False
        else:
            logger.warning(
                "RNG state not found in checkpoint training_state.json. Initial seeding may apply."
            )
            rng_loaded = False

        # --- Finalize ---
        load_duration = time.monotonic() - load_start_time
        logger.info(
            f"[green]Checkpoint loaded successfully.[/] Resuming from Step: {start_step}, Update: {start_updates} ({load_duration:.2f}s)"
        )
        return start_step, start_updates, rng_loaded

    except Exception as e:
        logger.critical(
            f"Checkpoint loading from '{resume_path}' failed critically: {e}",
            exc_info=True,
        )
        # Re-raise exception to halt execution if a critical error occurred
        raise



# --- GRPO Loss Function ---
def grpo_loss(
    model: nn.Module,
    tokens: mx.array,
    response_mask: mx.array,
    advantages: mx.array,
    ref_log_probs: mx.array,
    beta: float,
    pad_token_id: int,
):
    logger = logging.getLogger(__name__)

    batch_size, full_seq_len = tokens.shape
    if full_seq_len <= 1:
        logger.debug(f"Loss Fn: Full sequence length is {full_seq_len}, requires > 1 for shifted loss calculation. Returning zero loss.")
        return mx.zeros((), dtype=mx.float32)
    if batch_size == 0:
        logger.debug("Loss Fn: Received empty batch. Returning zero loss.")
        return mx.zeros((), dtype=mx.float32)

    generated_seq_len = response_mask.shape[1]
    if generated_seq_len == 0:
        logger.debug("Loss Fn: Generated sequence length is 0. No response tokens to calculate loss on. Returning zero loss.")
        return mx.zeros((), dtype=mx.float32)

    model_dtype = TARGET_FLOAT_DTYPE # Use the target dtype for model input
    try:
        # Attempt to get model's actual parameter dtype if different
        model_dtype = next(tree_flatten(model.parameters()))[1].dtype
    except (StopIteration, Exception):
        logger.debug(f"Loss Fn: Could not detect model dtype from parameters, using {model_dtype}.")


    # Model forward pass
    try:
        model_input_tokens = tokens.astype(mx.int64) # Cast input tokens to model dtype
        attn_mask_4d = _create_4d_attention_mask(tokens, pad_token_id, dtype=model_dtype)

        model_output = model(model_input_tokens, mask=attn_mask_4d)
        logits = model_output[0] if isinstance(model_output, tuple) else model_output
        logits = logits.astype(mx.float32) # Use float32 for loss calculations

    except Exception as e:
        logger.error(f"Model forward pass failed in loss function: {e}", exc_info=True)
        return mx.zeros((), dtype=mx.float32)

    # Calculate current log probs (pi_theta(a|s)) for the generated response tokens
    # Logits at index K predict token K+1.
    # The first generated token is at index `full_seq_len - generated_seq_len` in `tokens`.
    # The logits predicting this token are at index `full_seq_len - generated_seq_len - 1` in `logits`.
    response_start_idx_in_full_sequence = full_seq_len - generated_seq_len
    logits_start_idx_for_response = response_start_idx_in_full_sequence - 1

    if logits_start_idx_for_response < 0:
         # This should only happen if full_seq_len <= 1, which is checked at the start
         logger.error(f"Calculated logits_start_idx_for_response ({logits_start_idx_for_response}) is negative but full_seq_len is > 1 ({full_seq_len}). Indexing error. Returning zero loss.")
         return mx.zeros((), dtype=mx.float32)

    # Slice logits to cover only the predictions for the generated tokens
    logits_for_response = logits[:, logits_start_idx_for_response : -1, :] # Shape (batch, generated_seq_len, vocab_size)

    # Target tokens for current log probs are the generated tokens
    target_tokens_for_current = tokens[:, response_start_idx_in_full_sequence:] # Shape (batch, generated_seq_len)

    if logits_for_response.shape[1] != target_tokens_for_current.shape[1]:
         logger.error(f"Current log prob shape mismatch: Logits {logits_for_response.shape}, Targets {target_tokens_for_current.shape}. Returning zero loss.")
         return mx.zeros((), dtype=mx.float32)

    current_log_probs_response = selective_softmax(logits_for_response, target_tokens_for_current) # Shape (batch, generated_seq_len), dtype float32

    # Calculate log ratio: log(pi_theta / pi_ref)
    # ref_log_probs shape: (batch, generated_seq_len)
    if current_log_probs_response.shape != ref_log_probs.shape:
         logger.error(f"Log prob shape mismatch between current ({current_log_probs_response.shape}) and reference ({ref_log_probs.shape}). Returning zero loss.")
         return mx.zeros((), dtype=mx.float32)

    log_ratio = current_log_probs_response - ref_log_probs # Shape: (batch, generated_seq_len)

    # Policy gradient term: - advantage * log_prob(action)
    # advantages shape: (batch, 1) -> broadcast
    # Use masked current_log_probs for PG term
    # Mask is (batch, generated_seq_len)
    pg_term = -advantages * (current_log_probs_response * response_mask) # Shape: (batch, generated_seq_len)

    # KL divergence term: beta * (ratio - 1 - log_ratio) approx beta * KL(pi || pi_ref)
    ratio_unmasked = mx.exp(log_ratio)
    kl_div_approx_unmasked = (ratio_unmasked - 1 - log_ratio) # Shape: (batch, generated_seq_len)
    kl_term = beta * kl_div_approx_unmasked * response_mask # Shape: (batch, generated_seq_len)

    # Combine terms: Loss = - [ PG - KL ] = -PG + KL
    # GRPO Loss = -E[log(pi_theta/pi_ref) * Advantage] + beta * E[KL(pi_theta || pi_ref)]
    # Per-token loss: (-log_ratio * Advantage + beta * KL_approx) * response_mask
    per_token_loss = (-log_ratio * advantages + beta * kl_div_approx_unmasked) * response_mask # Shape: (batch, generated_seq_len)

    # Sum over sequence dimension, normalize by number of valid response tokens
    token_counts = mx.sum(response_mask, axis=1) # (batch,)
    sequence_loss_sum = mx.sum(per_token_loss, axis=1) # (batch,)

    # Avoid division by zero and ensure loss is 0 for samples with no generated tokens
    non_zero_count_mask = (token_counts > 0).astype(sequence_loss_sum.dtype)
    # Add a small epsilon to the denominator only where count is zero, then mask out the result
    # This prevents NaN/Inf from 0/0 while allowing division where count > 0
    sequence_loss = sequence_loss_sum / (token_counts + (1 - non_zero_count_mask) * 1e-8) # Add epsilon where count is 0
    sequence_loss = sequence_loss * non_zero_count_mask # Zero out loss where count was 0

    # Average loss over the batch
    loss = mx.mean(sequence_loss)

    # Check for NaN/Inf
    if mx.isnan(loss).any() or mx.isinf(loss).any():
        logger.error(f"NaN or Inf detected in GRPO loss! Loss: {loss.item()}")
        # Log component stats for debugging NaN/Inf
        if generated_seq_len > 0:
            # Calculate masked sums/means for logging
            num_masked_tokens = mx.sum(response_mask).item()
            num_masked_tokens = max(num_masked_tokens, 1.0) # Avoid div by zero for logging stats

            masked_pg_sum = mx.sum(pg_term * response_mask).item() # Sum over batch and sequence
            masked_kl_sum = mx.sum(kl_term * response_mask).item()

            avg_pg_loss_per_token = masked_pg_sum / num_masked_tokens
            avg_kl_loss_per_token = masked_kl_sum / num_masked_tokens

            logger.error(f"Component Stats (Avg per masked token): PG={avg_pg_loss_per_token:.4f}, KL={avg_kl_loss_per_token:.4f}")
            logger.error(f"Advantage Stats: mean={mx.mean(advantages).item():.4f}, std={mx.std(advantages).item():.4f}")

            # LogRatio stats over masked tokens
            masked_log_ratio = log_ratio * response_mask
            # Use mx.mean with 'where' if available, otherwise flatten and filter
            if hasattr(mx, 'mean') and 'where' in mx.mean.__code__.co_varnames: # Check for 'where' arg
                 mean_log_ratio = mx.mean(masked_log_ratio, where=response_mask).item() if num_masked_tokens > 0 else 0.0
                 std_log_ratio = mx.std(masked_log_ratio, where=response_mask).item() if num_masked_tokens > 1 else 0.0
            else: # Fallback for older MLX versions
                 flat_masked_log_ratio = masked_log_ratio.flatten()
                 relevant_log_ratios = flat_masked_log_ratio[flat_masked_log_ratio != 0] # Assumes 0 is only from mask
                 mean_log_ratio = mx.mean(relevant_log_ratios).item() if relevant_log_ratios.size > 0 else 0.0
                 std_log_ratio = mx.std(relevant_log_ratios).item() if relevant_log_ratios.size > 1 else 0.0

            logger.error(f"LogRatio Stats (masked tokens): mean={mean_log_ratio:.4f}, std={std_log_ratio:.4f}")


        raise ValueError("NaN/Inf loss detected")

    return loss


# --- Evaluation Function ---
# FIX START: evaluate_grpo method

def evaluate_grpo(
    model: nn.Module,
    dataset: Dataset,
    tokenizer: TokenizerWrapper,
    args: TrainingArgs,
    model_config: Dict,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> Dict[str, float]:
    """
    Evaluates the model on the validation set using greedy generation and reward calculation.
    Uses explicit indexing for correct reward alignment.
    """
    eval_logger = logging.getLogger(__name__)
    if dataset is None:
        eval_logger.info("No validation dataset provided. Skipping evaluation.")
        return {}
    max_iters = len(dataset)
    if max_iters <= 0:
        eval_logger.info("Validation dataset is empty. Skipping evaluation.")
        return {}
    eval_logger.info(f"Starting evaluation on {max_iters} samples...")

    total_rewards, total_format_rewards, total_content_rewards_combined = [], [], []
    samples_processed = 0
    rollout_examples = []
    reward_config = RewardConfig(
        think_start_tag=args.think_start_tag,
        think_end_tag=args.think_end_tag,
        answer_start_tag=args.answer_start_tag,
        answer_end_tag=args.answer_end_tag,
    )
    try:
        get_content_reward_fn(args.reward_content_type)
    except ValueError:
        eval_logger.error(f"Invalid reward_content_type '{args.reward_content_type}'. Evaluation aborted.")
        return {}

    eval_batch_size = args.ppo_batch_size
    indices = list(range(max_iters))
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        error_msg = "Tokenizer needs pad_token_id for evaluation padding."
        eval_logger.critical(error_msg)
        return {}

    eos_token_str = tokenizer.decode([eos_id], skip_special_tokens=False, clean_up_tokenization_spaces=True) if eos_id is not None else ""
    pad_token_str = tokenizer.decode([pad_id], skip_special_tokens=False, clean_up_tokenization_spaces=True) if pad_id is not None else ""

    model_dtype = TARGET_FLOAT_DTYPE
    try:
        model_dtype = next(tree_flatten(model.parameters()))[1].dtype
    except (StopIteration, Exception):
        eval_logger.warning(f"Eval: Could not detect model dtype from parameters, using {model_dtype}.")


    model.eval()
    if progress and task_id:
        progress.reset(task_id, total=max_iters, description="Evaluating...")

    # Greedy sampler (temp=0.0, top_p=0.0, min_p=0.0)
    eval_sampler = make_sampler(temp=0.0, top_p=0.0, min_p=0.0)

    for i in range(0, max_iters, eval_batch_size):
        if shutdown_requested:
            eval_logger.info("Shutdown requested. Stopping evaluation.")
            break
        batch_indices = indices[i : i + eval_batch_size]
        if not batch_indices:
            continue

        # Build batch for evaluation (num_samples_per_prompt is 1 for greedy eval)
        prompts_data_batch, prompts_mx, max_prompt_len_batch = build_rollout_batch(
            tokenizer,
            dataset,
            batch_indices,
            args.max_prompt_len,
            args.system_prompt,
            args.dataset_prompt_key,
            args.dataset_answer_key,
        )
        if not prompts_data_batch:
            eval_logger.warning(f"Skipping evaluation batch starting at index {i}: No valid prompts found.")
            if progress and task_id: progress.update(task_id, advance=len(batch_indices))
            continue

        batch_size_actual = prompts_mx.shape[0]
        if batch_size_actual == 0:
            eval_logger.warning(f"Skipping evaluation batch starting at index {i}: prompts_mx is empty after build_rollout_batch.")
            if progress and task_id: progress.update(task_id, advance=len(batch_indices))
            continue

        # --- Generate Responses (Greedy, Batch) ---
        batch_responses_tokens = []
        # Initialize caches for the batch of prompts for evaluation
        # FIX: Remove batch_size argument from make_prompt_cache call
        eval_caches = cache.make_prompt_cache(model, max_kv_size=args.max_kv_size)

        try:
            # Prefill the model
            prompt_attn_mask_4d = _create_4d_attention_mask(prompts_mx, pad_id, dtype=model_dtype)
            model_input_prompts = prompts_mx.astype(model_dtype)
            with mx.stream(generation_stream):
                 model_output = model(model_input_prompts, mask=prompt_attn_mask_4d, cache=eval_caches)

            next_token_logits = (model_output[0] if isinstance(model_output, tuple) else model_output)[:, -1, :]
            mx.eval(next_token_logits, [c.state for c in eval_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)]) # Evaluate cache states

            # Apply KV cache quantization after prefill if enabled and threshold met
            if args.kv_bits is not None:
                 eval_logger.debug(f"Applying KV cache quantization after eval prefill (threshold: {args.quantized_kv_start}).")
                 maybe_quantize_kv_cache(eval_caches, args.quantized_kv_start, args.kv_group_size, args.kv_bits) # <-- CORRECTED CALL
                 mx.eval([c.state for c in eval_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)]) # Evaluate updated cache states


            current_tokens = eval_sampler(next_token_logits)
            batch_responses_tokens.append(current_tokens[:, None])

            ended = mx.zeros_like(current_tokens)
            if eos_id is not None:
                ended = mx.equal(current_tokens, eos_id)

            for step in range(args.max_gen_len - 1):
                if mx.all(ended).item():
                    eval_logger.debug(f"All sequences in eval batch ended generation at step {step+1}.")
                    break

                ended_before_step = ended

                try:
                    model_input_step = current_tokens[:, None].astype(model_dtype)
                    with mx.stream(generation_stream):
                         model_step_output = model(model_input_step, cache=eval_caches)

                    next_token_logits = (model_step_output[0] if isinstance(model_step_output, tuple) else model_step_output)[:, -1, :]
                    sampled_tokens = eval_sampler(next_token_logits)

                    just_ended = mx.equal(sampled_tokens, eos_id) if eos_id is not None else mx.zeros_like(ended)
                    ended = mx.logical_or(ended_before_step, just_ended)

                    # Select token: PAD if already ended before this step, else the new token
                    pad_val_mx = mx.array(pad_id, dtype=sampled_tokens.dtype)
                    tokens_to_add = mx.where(ended_before_step, pad_val_mx, sampled_tokens)

                    eval_list = [tokens_to_add, ended]
                    eval_list.extend([c.state for c in eval_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)])
                    mx.async_eval(eval_list)

                    batch_responses_tokens.append(tokens_to_add[:, None])
                    current_tokens = tokens_to_add

                    # Apply KV cache quantization
                    if args.kv_bits is not None:
                         # Current sequence length for cache is prompt_len + tokens generated so far + current token
                         # The cache object's offset should track this correctly after the forward pass
                         # Check if *any* sequence in the batch exceeds the threshold
                         current_seq_len_after_step = eval_caches[0].offset + 1 # Assuming all layer caches have the same offset
                         if current_seq_len_after_step > args.quantized_kv_start:
                              maybe_quantize_kv_cache(eval_caches, args.quantized_kv_start, args.kv_group_size, args.kv_bits) # <-- CORRECTED CALL
                              mx.async_eval([c.state for c in eval_caches if hasattr(c, 'state') and isinstance(c.state, mx.array)]) # Evaluate updated cache states

                except Exception as e_eval_gen_step:
                    eval_logger.error(f"Evaluation generation failed at step {step+1}: {e_eval_gen_step}", exc_info=True)
                    current_len = len(batch_responses_tokens)
                    remaining_len = args.max_gen_len - current_len
                    if remaining_len > 0:
                        eval_logger.warning(f"Padding remaining {remaining_len} steps due to error.")
                        pad_dtype = batch_responses_tokens[0].dtype
                        pad_array = mx.full((batch_size_actual, remaining_len), pad_id, dtype=pad_dtype)
                        batch_responses_tokens.extend([pad_array[:, k:k+1] for k in range(remaining_len)])
                    break

        except Exception as e_eval_gen_loop:
             eval_logger.error(f"Evaluation generation loop failed: {e_eval_gen_loop}", exc_info=True)


        if batch_responses_tokens:
            try:
                responses_mx = mx.concatenate(batch_responses_tokens, axis=1)
                mx.eval(responses_mx)
                decoded_responses = tokenizer.batch_decode(
                    responses_mx.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            except Exception as e_decode:
                eval_logger.error(f"Evaluation batch decoding failed: {e_decode}")
                decoded_responses = ["[decoding error]"] * batch_size_actual
        else:
            eval_logger.warning(f"No tokens generated for evaluation batch starting at index {i}. Using empty responses.")
            decoded_responses = [""] * batch_size_actual

        for j in range(batch_size_actual):
            resp_text = decoded_responses[j]
            ref_ans = prompts_data_batch[j]["ref_answer_str"]
            prompt_text_for_log = prompts_data_batch[j]["text"]

            cleaned_resp = resp_text
            if pad_token_str:
                cleaned_resp = re.sub(rf"(?:\s*{re.escape(pad_token_str)})+$", "", cleaned_resp).rstrip()
            if eos_token_str:
                eos_pos = cleaned_resp.find(eos_token_str)
                if eos_pos != -1: cleaned_resp = cleaned_resp[:eos_pos].strip()

            if cleaned_resp.startswith("[") and "error]" in cleaned_resp:
                total_rew, fmt_rew, content_rew_combined = 0.0, 0.0, 0.0
            else:
                total_rew, fmt_rew, content_rew_combined = calculate_total_reward(
                    cleaned_resp,
                    ref_ans,
                    reward_config,
                    args.reward_format_weight,
                    args.reward_content_weight,
                    args.reward_content_type,
                )

            total_rewards.append(total_rew)
            total_format_rewards.append(fmt_rew)
            total_content_rewards_combined.append(content_rew_combined)
            samples_processed += batch_indices[j] - batch_indices[0] + 1 # Correctly count samples processed in this batch

        if progress and task_id:
            progress.update(task_id, advance=batch_size_actual)

    model.train()

    mean_total_reward = np.mean(total_rewards) if total_rewards else 0.0
    mean_format_reward_raw = np.mean(total_format_rewards) if total_format_rewards else 0.0
    mean_content_reward_raw_combined = np.mean(total_content_rewards_combined) if total_content_rewards_combined else 0.0

    eval_logger.info(f"Evaluation finished. Processed: {samples_processed}/{max_iters} samples.")
    eval_logger.info(f"  [bold]Mean Total Reward (Weighted): {mean_total_reward:.4f}[/]")
    eval_logger.info(f"  Mean Format Reward (Raw): {mean_format_reward_raw:.4f}")
    eval_logger.info(f"  Mean Content Reward (Raw/Combined): {mean_content_reward_raw_combined:.4f} (Type: {args.reward_content_type})")

    if rollout_examples:
        table = Table(
            title=f"Evaluation Samples (First {len(rollout_examples)})",
            show_header=True,
            header_style="bold magenta",
            box=None,
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Prompt End", style="dim cyan", max_width=40, overflow="fold")
        table.add_column("Generated", style="white", max_width=60, overflow="fold")
        table.add_column("Reference", style="dim green", max_width=30, overflow="fold")
        table.add_column("Reward Details", style="bold yellow", justify="right", min_width=25)
        for ex in rollout_examples:
            table.add_row(ex["prompt"], ex["generated"], ex["reference"], ex["reward"])
        console.print(table)

    final_metrics = {
        "eval/mean_reward_weighted": mean_total_reward,
        "eval/mean_format_reward_raw": mean_format_reward_raw,
        "eval/mean_content_reward_raw_combined": mean_content_reward_raw_combined,
        "eval/samples_processed": float(samples_processed),
    }
    return final_metrics

# FIX END: evaluate_grpo method


# --- Training Orchestration ---
def train(
    args: TrainingArgs,
    actor_model: nn.Module,
    ref_model: nn.Module,
    draft_model: Optional[nn.Module], # Added draft model
    model_config: dict,
    tokenizer: TokenizerWrapper,
    train_dset: Dataset,
    val_dset: Optional[Dataset],
    optimizer: optim.Optimizer,
    lr_scheduler: Callable[[int], float], # Added LR scheduler
    start_step: int,
    start_updates: int,
):
    """Main training loop for GRPO."""
    global shutdown_requested
    train_logger = logging.getLogger(__name__)
    wandb_run = None

    if args.use_wandb:
        if WANDB_AVAILABLE:
            try:
                wandb_config_dict = asdict(args)
                for k, v in wandb_config_dict.items():
                    if isinstance(v, Path):
                        wandb_config_dict[k] = str(v)
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name,
                    config=wandb_config_dict,
                    resume="allow",
                )
                train_logger.info(f"WandB logging enabled. Run: {wandb.run.name} ({wandb.run.url})")
            except Exception as e:
                train_logger.error(f"WandB initialization failed: {e}. Disabling WandB.", exc_info=True)
                args.use_wandb = False
        else:
            train_logger.warning("WandB requested but not available. Disabling WandB logging.")
            args.use_wandb = False

    # --- Gradient Checkpointing Setup ---
    if args.use_grad_checkpointing:
        train_logger.info("Gradient checkpointing enabled.")
        # Attempt to find common layer types and wrap them
        try:
            # Define common layer types to checkpoint (e.g., TransformerDecoderLayer)
            # This list might need adjustment based on the specific model architecture
            # Look for modules that contain 'attn' and 'mlp' submodules, common in transformer blocks
            checkpointable_layer_types = []
            # Example: Find LlamaDecoderLayer if LlamaModel is available
            try:
                from mlx_lm.models.llama import LlamaDecoderLayer
                checkpointable_layer_types.append(LlamaDecoderLayer)
            except ImportError:
                pass # LlamaDecoderLayer not available

            # Add other common layer types if needed (e.g., from GPT, Mistral, etc.)
            # try:
            #     from mlx_lm.models.mistral import MistralDecoderLayer
            #     checkpointable_layer_types.append(MistralDecoderLayer)
            # except ImportError: pass

            if not checkpointable_layer_types:
                 train_logger.warning("No known checkpointable layer types found. Gradient checkpointing will not be applied.")
            else:
                all_layers = []
                # Traverse the model to find instances of checkpointable types
                for name, module in actor_model.named_modules():
                    if any(isinstance(module, t) for t in checkpointable_layer_types):
                        all_layers.append((name, module))

                if all_layers:
                    num_layers_total = len(all_layers)
                    num_ckpt = args.grad_checkpoint_layers if args.grad_checkpoint_layers is not None and args.grad_checkpoint_layers > 0 else num_layers_total
                    num_ckpt = min(num_ckpt, num_layers_total)

                    # Apply to the top `num_ckpt` layers
                    layers_to_checkpoint = all_layers[-num_ckpt:]
                    train_logger.info(f"Applying grad checkpoint to {len(layers_to_checkpoint)}/{num_layers_total} layers (top {num_ckpt}).")

                    applied_count = 0
                    for name, layer in layers_to_checkpoint:
                        try:
                            # grad_checkpoint wraps the module in-place
                            # Check if it's already checkpointed to avoid double wrapping
                            if not hasattr(layer, "_checkpointed_wrapped"):
                                grad_checkpoint(layer)
                                applied_count += 1
                                train_logger.debug(f"Applied grad checkpoint to layer: {name}")
                            else:
                                train_logger.debug(f"Layer {name} already appears to be checkpointed.")

                        except Exception as e_gc_apply:
                            train_logger.error(f"Failed applying grad checkpoint to layer {name}: {e_gc_apply}")

                    train_logger.info(f"Successfully applied grad checkpointing to {applied_count} layers.")
                else:
                    train_logger.warning("No suitable layers found for gradient checkpointing.")

        except Exception as e_gc_setup:
            train_logger.error(f"Error during gradient checkpointing setup: {e_gc_setup}")


    global_step = start_step
    num_updates = start_updates

    mx.eval(actor_model.parameters())
    mx.eval(optimizer.state)
    mx.eval(ref_model.parameters())
    if draft_model: mx.eval(draft_model.parameters())


    start_time = time.monotonic()
    last_save_update = num_updates
    last_eval_update = num_updates
    peak_mem_gb = 0.0

    output_dir = Path(args.output_dir)
    metrics_logger = MetricsLogger(output_dir / "training_metrics.csv")
    reward_config = RewardConfig(
        think_start_tag=args.think_start_tag,
        think_end_tag=args.think_end_tag,
        answer_start_tag=args.answer_start_tag,
        answer_end_tag=args.answer_end_tag,
    )

    train_logger.info(f"Training outputs will be saved to: [link=file://{output_dir.resolve()}]{output_dir}[/]")

    # Prepare loss function and gradient calculation
    value_and_grad_fn = nn.value_and_grad(
        actor_model,
        lambda model, tokens, response_mask, advantages, ref_log_probs: grpo_loss(
            model,
            tokens,
            response_mask,
            advantages,
            ref_log_probs,
            args.grpo_beta,
            tokenizer.pad_token_id,
        ),
    )

    train_indices = list(range(len(train_dset)))
    total_updates_target = args.num_training_steps
    train_logger.info(f"Starting GRPO training from update {num_updates}. Target updates: {total_updates_target}. Effective batch size (samples per update): {args.effective_batch_size}")

    progress_cols = (
        TextColumn("[b blue]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("• Upd: [b]{task.completed:.0f}[/]/[dim]{task.total:.0f}"),
        TextColumn("• Loss:[red]{task.fields[loss]:.3f}"),
        TextColumn("• RollRew:[yellow]{task.fields[roll_rew]:.2f}"),
        TextColumn("• FmtRew:[dim green]{task.fields[fmt_rew]:.2f}"),
        TextColumn("• ContRew:[dim yellow]{task.fields[cont_rew]:.2f}"),
        TextColumn("• LR:[cyan]{task.fields[lr]:.1e}"),
        TextColumn("• GradN:[magenta]{task.fields[grad_norm]:.2f}"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        TimeElapsedColumn(),
    )

    exit_code = 0 # Default exit code is success

    try:
        with Progress(*progress_cols, console=console, transient=False, refresh_per_second=2) as progress:
            main_task = progress.add_task(
                "Training",
                total=total_updates_target,
                completed=num_updates,
                loss=math.nan,
                roll_rew=math.nan,
                fmt_rew=math.nan,
                cont_rew=math.nan,
                lr=args.learning_rate,
                grad_norm=math.nan,
            )

            current_indices_pos = 0

            while num_updates < total_updates_target and not shutdown_requested:

                if args.shuffle_data and current_indices_pos == 0:
                    random.shuffle(train_indices)
                    train_logger.debug("Shuffled training indices.")

                accumulated_grads = None
                accum_count = 0
                effective_batch_losses, effective_batch_rewards_weighted = [], []
                effective_batch_rewards_fmt, effective_batch_rewards_content_combined = [], []

                num_prompts_for_effective_batch = args.ppo_batch_size * args.grad_accum_steps
                effective_batch_prompt_indices = []
                start_pos = current_indices_pos
                for k in range(num_prompts_for_effective_batch):
                    idx_pos = (start_pos + k) % len(train_indices)
                    effective_batch_prompt_indices.append(train_indices[idx_pos])
                current_indices_pos = (start_pos + num_prompts_for_effective_batch) % len(train_indices)

                micro_batch_indices_list = [
                    effective_batch_prompt_indices[k : k + args.ppo_batch_size]
                    for k in range(0, num_prompts_for_effective_batch, args.ppo_batch_size)
                ]

                for micro_batch_indices in micro_batch_indices_list:
                    if shutdown_requested: break
                    if not micro_batch_indices: continue

                    rollout_start = time.monotonic()
                    try:
                        (
                            prompts_data_mb,
                            prompts_mx_mb,
                            max_prompt_len_mb,
                        ) = build_rollout_batch(
                            tokenizer,
                            train_dset,
                            micro_batch_indices,
                            args.max_prompt_len,
                            args.system_prompt,
                            args.dataset_prompt_key,
                            args.dataset_answer_key,
                        )
                        if not prompts_data_mb:
                            train_logger.warning(f"Skipping micro-batch (Update {num_updates+1}, accum {accum_count+1}): No valid prompts from indices {micro_batch_indices}.")
                            continue

                        (
                            rollout_data,
                            avg_rollout_rew_w,
                            raw_rew_comp,
                        ) = generate_rollouts_for_batch(
                            actor_model,
                            ref_model,
                            draft_model if args.use_speculative_decoding else None, # Pass draft model conditionally
                            tokenizer,
                            prompts_data_mb,
                            prompts_mx_mb,
                            max_prompt_len_mb,
                            args.num_rollout_samples,
                            args.max_gen_len,
                            args,
                        )

                        if rollout_data["tokens"].shape[0] == 0 or rollout_data["response_mask"].shape[1] == 0:
                            train_logger.warning(f"Skipping micro-batch (Update {num_updates+1}, accum {accum_count+1}): Rollout generation resulted in empty sequences.")
                            continue

                        rollout_dur = time.monotonic() - rollout_start
                        effective_batch_rewards_weighted.append(avg_rollout_rew_w)
                        effective_batch_rewards_fmt.append(raw_rew_comp.get("raw_format", math.nan))
                        effective_batch_rewards_content_combined.append(raw_rew_comp.get("raw_content_combined", math.nan))

                        progress.update(
                            main_task,
                            roll_rew=np.nanmean(effective_batch_rewards_weighted) if effective_batch_rewards_weighted else math.nan,
                            fmt_rew=np.nanmean(effective_batch_rewards_fmt) if effective_batch_rewards_fmt else math.nan,
                            cont_rew=np.nanmean(effective_batch_rewards_content_combined) if effective_batch_rewards_content_combined else math.nan,
                        )

                    except Exception as e_rollout:
                        train_logger.error(f"Rollout failed for micro-batch (Update {num_updates+1}, accum {accum_count+1}): {e_rollout}", exc_info=True)
                        continue

                    loss_calc_start = time.monotonic()
                    try:
                        loss, grads_tree = value_and_grad_fn(actor_model, **rollout_data)
                        mx.eval(loss, grads_tree)
                        loss_item = loss.item()
                        if math.isnan(loss_item) or math.isinf(loss_item):
                            train_logger.error(f"NaN/Inf loss ({loss_item:.4f}) detected in micro-batch {accum_count+1}. Skipping grad accumulation for this micro-batch.")
                            continue

                        effective_batch_losses.append(loss_item)

                        scaled_grads_tree = tree_map(lambda g: g / args.grad_accum_steps, grads_tree)

                        if accumulated_grads is None:
                            accumulated_grads = scaled_grads_tree
                        else:
                            accumulated_grads = tree_map(
                                lambda acc, new: acc + new if acc is not None and new is not None else acc or new,
                                accumulated_grads,
                                scaled_grads_tree,
                            )

                        accum_count += 1
                        loss_calc_dur = time.monotonic() - loss_calc_start
                        train_logger.debug(f"Loss/Grad (Upd {num_updates}, MB {accum_count}/{args.grad_accum_steps}): Loss: {loss_item:.4f}. Time: {loss_calc_dur:.2f}s.")

                    except ValueError as e_loss_val: # Catch NaN/Inf from loss function raise
                        train_logger.error(f"Loss/Grad calculation failed (NaN/Inf) for micro-batch {accum_count+1} (Update {num_updates+1}): {e_loss_val}. Discarding effective batch gradients.", exc_info=True)
                        accumulated_grads = None
                        effective_batch_losses = []
                        break
                    except Exception as e_loss:
                        train_logger.error(f"Loss/Grad calculation failed for micro-batch (Update {num_updates+1}, accum {accum_count+1}): {e_loss}", exc_info=True)
                        accumulated_grads = None
                        effective_batch_losses = []
                        break

                if shutdown_requested: break

                # --- Optimizer Step ---
                if accum_count == args.grad_accum_steps and accumulated_grads is not None:
                    update_start_time = time.monotonic()
                    num_updates += 1

                    try:
                        mx.eval(accumulated_grads)

                        # Get current learning rate from scheduler
                        current_lr = lr_scheduler(num_updates)
                        optimizer.learning_rate = current_lr # Manually update LR

                        final_grads = accumulated_grads
                        grad_norm_mx = mx.array(math.nan)
                        grad_norm = math.nan
                        if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                            final_grads, grad_norm_mx = optim.clip_grad_norm(final_grads, args.grad_clip_norm)
                            mx.eval(final_grads, grad_norm_mx)
                            grad_norm = grad_norm_mx.item()
                        else:
                            try:
                                grad_norm_mx = optim.compute_grad_norm(accumulated_grads)
                                mx.eval(grad_norm_mx)
                                grad_norm = grad_norm_mx.item()
                            except Exception: pass

                        trainable_params = actor_model.trainable_parameters()
                        optimizer.apply_gradients(final_grads, trainable_params)

                        mx.eval(trainable_params, optimizer.state)

                        avg_loss_step = np.mean(effective_batch_losses) if effective_batch_losses else math.nan
                        avg_reward_step_weighted = np.nanmean(effective_batch_rewards_weighted) if effective_batch_rewards_weighted else math.nan
                        avg_reward_step_fmt = np.nanmean(effective_batch_rewards_fmt) if effective_batch_rewards_fmt else math.nan
                        avg_reward_step_content_combined = np.nanmean(effective_batch_rewards_content_combined) if effective_batch_rewards_content_combined else math.nan


                        update_duration = time.monotonic() - update_start_time
                        progress.update(
                            main_task,
                            advance=1,
                            completed=num_updates,
                            loss=avg_loss_step,
                            roll_rew=avg_reward_step_weighted,
                            fmt_rew=avg_reward_step_fmt,
                            cont_rew=avg_reward_step_content_combined,
                            lr=current_lr,
                            grad_norm=grad_norm,
                        )

                        mem_metric = {}
                        if PSUTIL_AVAILABLE:
                            try:
                                process = psutil.Process(os.getpid())
                                mem_info = process.memory_info()
                                current_mem_gb = mem_info.rss / (1024**3)
                                peak_mem_gb = max(peak_mem_gb, current_mem_gb)
                                mlx_mem_peak_gb = mx.metal.get_peak_memory() / (1024**3) if hasattr(mx, "metal") and mx.metal.is_available() else 0.0
                                mlx_mem_current_gb = mx.metal.get_active_memory() / (1024**3) if hasattr(mx, "metal") and mx.metal.is_available() else 0.0
                                mem_metric = {
                                    "memory/process_peak_gb": peak_mem_gb,
                                    "memory/process_current_gb": current_mem_gb,
                                    "memory/mlx_peak_gb": mlx_mem_peak_gb,
                                    "memory/mlx_current_gb": mlx_mem_current_gb,
                                }
                            except Exception: pass

                        metrics_to_log = {
                            "train/loss": avg_loss_step,
                            "train/mean_rollout_reward_weighted": avg_reward_step_weighted,
                            "train/mean_format_reward_raw": avg_reward_step_fmt,
                            "train/mean_content_reward_raw_combined": avg_reward_step_content_combined,
                            "train/learning_rate": current_lr, # Log actual LR used
                            "train/grad_norm": grad_norm if not math.isnan(grad_norm) else 0.0,
                            "timers/update_duration_sec": update_duration,
                            "timers/rollout_duration_sec": rollout_dur,
                            "train/accum_count": accum_count,
                            **mem_metric,
                        }
                        file_log = {
                            "update_step": num_updates,
                            **metrics_to_log,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        metrics_logger.log(file_log)
                        if args.use_wandb and wandb_run:
                            try:
                                wandb.log(metrics_to_log, step=num_updates)
                            except Exception as e_wandb:
                                train_logger.error(f"WandB logging failed: {e_wandb}")

                        if (num_updates % args.eval_every == 0 or num_updates == 1 or num_updates == total_updates_target or num_updates % 10 == 0):
                            train_logger.info(
                                f"Update {num_updates}/{total_updates_target} | "
                                f"Loss: {avg_loss_step:.4f} | RollRew(Wt): {avg_reward_step_weighted:.3f} | "
                                f"RawFmt: {avg_reward_step_fmt:.2f} | RawCont(Comb): {avg_reward_step_content_combined:.2f} ({args.reward_content_type}) | "
                                f"GradNorm: {grad_norm:.2f} | LR: {current_lr:.1e} | "
                                f"Time: {update_duration:.2f}s"
                            )
                        accumulated_grads = None
                        accum_count = 0
                        effective_batch_losses, effective_batch_rewards_weighted = [], []
                        effective_batch_rewards_fmt, effective_batch_rewards_content_combined = [], []

                    except Exception as e_update:
                        train_logger.error(f"Optimizer step failed for update {num_updates}: {e_update}", exc_info=True)

                if (num_updates > last_save_update or num_updates > last_eval_update):
                    perform_eval = (val_dset is not None) and ((num_updates % args.eval_every == 0) or (num_updates >= total_updates_target))
                    if perform_eval and num_updates > last_eval_update:
                        eval_start = time.monotonic()
                        train_logger.info(f"--- Starting Evaluation @ Update {num_updates} ---")
                        try:
                            with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console, transient=True) as eval_prog:
                                eval_task = eval_prog.add_task(f"Evaluating", total=len(val_dset))
                                eval_metrics = evaluate_grpo(
                                    actor_model,
                                    val_dset,
                                    tokenizer,
                                    args,
                                    model_config,
                                    progress=eval_prog,
                                    task_id=eval_task,
                                )
                            eval_dur = time.monotonic() - eval_start
                            train_logger.info(f"--- Evaluation Finished ({eval_dur:.2f}s) ---")
                            if eval_metrics:
                                fm = {k: f"{v:.4f}" for k, v in eval_metrics.items()}
                                train_logger.info(f"Eval Metrics @ Update {num_updates}: {fm}")
                                eval_log = {
                                    "update_step": num_updates,
                                    **eval_metrics,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                }
                                metrics_logger.log(eval_log)
                                if args.use_wandb and wandb_run:
                                    try:
                                        wandb.log(eval_metrics, step=num_updates)
                                    except Exception as e_wandb_eval:
                                        train_logger.error(f"WandB eval logging failed: {e_wandb_eval}")
                            last_eval_update = num_updates
                        except Exception as e_eval:
                            train_logger.error(f"Evaluation failed at update {num_updates}: {e_eval}", exc_info=True)

                    should_save, save_reason = False, ""
                    if SAVE_ON_EXIT_FLAG_PATH.exists():
                        should_save, save_reason = True, "exit_request"
                        logger.warning("Save-on-exit flag detected, initiating save...")
                        try: SAVE_ON_EXIT_FLAG_PATH.unlink()
                        except OSError: pass
                    elif shutdown_requested:
                        should_save, save_reason = True, "shutdown_signal"
                    elif num_updates % args.save_every == 0:
                        should_save, save_reason = True, "periodic"
                    elif num_updates >= total_updates_target:
                        should_save, save_reason = True, "final_step"

                    if should_save and num_updates > last_save_update:
                        save_checkpoint(
                            output_dir,
                            save_reason,
                            global_step,
                            num_updates,
                            actor_model,
                            optimizer,
                            tokenizer,
                            args,
                            model_config,
                        )
                        last_save_update = num_updates

                if num_updates >= total_updates_target:
                    train_logger.info(f"Target number of updates ({total_updates_target}) reached.")
                    break
                if shutdown_requested:
                    train_logger.info("Shutdown requested, exiting training loop.")
                    break

            final_loss = progress.tasks[main_task].fields.get("loss", math.nan) if progress.tasks else math.nan
            progress.update(main_task, description=f"Training Finished (Loss: {final_loss:.3f})", refresh=True)

    except KeyboardInterrupt:
        train_logger.warning("\nTraining interrupted by user (KeyboardInterrupt).")
        shutdown_requested = True
    except Exception as train_err:
        train_logger.critical("Critical error during training loop!", exc_info=True)
        console.print_exception(show_locals=args.verbose)
        exit_code = 1
        if num_updates > 0:
            try:
                train_logger.warning("Attempting emergency checkpoint save due to critical error...")
                save_checkpoint(
                    output_dir,
                    "crash",
                    global_step,
                    num_updates,
                    actor_model,
                    optimizer,
                    tokenizer,
                    args,
                    model_config,
                )
            except Exception as final_save_err:
                train_logger.error(f"Emergency checkpoint save FAILED: {final_save_err}")
    finally:
        total_training_duration = time.monotonic() - start_time
        final_status = "Completed" if num_updates >= total_updates_target and not shutdown_requested else "Interrupted"
        train_logger.info(f"--- Training {final_status} ---")
        train_logger.info(f" Total Updates Performed: {num_updates}/{total_updates_target}")
        train_logger.info(f" Total Duration: {total_training_duration / 3600:.2f} hrs ({total_training_duration:.1f} sec)")
        if PSUTIL_AVAILABLE:
            train_logger.info(f" Peak Process Memory Usage: {peak_mem_gb:.2f} GB")
        if hasattr(mx, "metal") and mx.metal.is_available():
            final_mlx_peak_gb = mx.metal.get_peak_memory() / (1024**3)
            final_mlx_active_gb = mx.metal.get_active_memory() / (1024**3)
            train_logger.info(f" Peak MLX Metal Memory: {final_mlx_peak_gb:.2f} GB")
            train_logger.info(f" Final MLX Metal Active Memory: {final_mlx_active_gb:.2f} GB")

        save_reason_final = locals().get("save_reason", "N/A")
        train_logger.info(f" Final Checkpoint Reason: {save_reason_final}")
        train_logger.info(f" Log files and checkpoints saved in: [link=file://{output_dir.resolve()}]{output_dir}[/]")

        saved_at_final_update = last_save_update == num_updates

        if shutdown_requested and not saved_at_final_update:
            train_logger.info("Performing final save due to interruption...")
            save_checkpoint(
                output_dir,
                "interrupted_final",
                global_step,
                num_updates,
                actor_model,
                optimizer,
                tokenizer,
                args,
                model_config,
            )
        elif num_updates >= total_updates_target and not saved_at_final_update:
            train_logger.info("Performing final save upon completion...")
            save_checkpoint(
                output_dir,
                "final_step",
                global_step,
                num_updates,
                actor_model,
                optimizer,
                tokenizer,
                args,
                model_config,
            )

        metrics_logger.close()
        if wandb_run:
            train_logger.info("Finishing WandB run...")
            try:
                wandb.finish(exit_code=exit_code)
            except Exception as e_wb_finish:
                logger.error(f"WandB finish failed: {e_wb_finish}")

        if SAVE_ON_EXIT_FLAG_PATH.exists():
            train_logger.info(f"Cleaning up final save flag: {SAVE_ON_EXIT_FLAG_PATH}")
            try: SAVE_ON_EXIT_FLAG_PATH.unlink()
            except OSError: pass

        # Explicitly clear MLX cache at the very end
        mx.clear_cache()
        logger.info("MLX cache cleared.")


# --- Embedding Initialization Helper ---
def initialize_new_token_embeddings(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    new_token_ids: List[int],
    init_with_mean: bool,
):
    """Initializes embeddings for newly added tokens."""
    logger = logging.getLogger(__name__)
    if not new_token_ids:
        logger.info("No new token IDs provided for embedding initialization.")
        return

    try:
        # Find the embedding layer (common patterns)
        embeddings = None
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embeddings = model.model.embed_tokens
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            embeddings = model.transformer.wte
        elif hasattr(model, "embed_tokens"):
            embeddings = model.embed_tokens
        else:
            logger.error("Could not find embedding layer ('embed_tokens' or 'transformer.wte') in the model structure.")
            raise ValueError("Embedding layer not found.")

        if not isinstance(embeddings, (nn.Embedding, QuantizedEmbedding)):
            logger.error(f"Found embedding attribute is not an nn.Embedding or QuantizedEmbedding layer: {type(embeddings)}")
            raise ValueError("Attribute is not an Embedding layer.")

        # If quantized, convert to float for calculation, then maybe re-quantize later if needed (though resize handles this)
        embedding_matrix = embeddings.weight
        if isinstance(embeddings, QuantizedEmbedding):
             # Dequantize for calculation
             embedding_matrix = mx.dequantize(
                 embedding_matrix,
                 embeddings.scales,
                 embeddings.biases,
                 embeddings.group_size,
                 embeddings.bits,
             ).astype(TARGET_FLOAT_DTYPE) # Use target float dtype

        num_existing_tokens = embedding_matrix.shape[0] - len(new_token_ids)
        embedding_dim = embedding_matrix.shape[1]
        logger.info(f"Initializing embeddings for {len(new_token_ids)} new tokens (IDs: {new_token_ids}). Existing vocab: {num_existing_tokens}, Dim: {embedding_dim}.")

        if num_existing_tokens <= 0:
            logger.warning("No existing tokens to compute mean embedding from. Using default random initialization (already done by resize).")
            init_with_mean = False

        if init_with_mean:
            logger.info("Calculating mean of existing embeddings...")
            mean_embedding = mx.mean(embedding_matrix[:num_existing_tokens], axis=0, keepdims=False)
            mx.eval(mean_embedding)
            if mean_embedding.shape != (embedding_dim,):
                raise ValueError(f"Mean embedding shape incorrect: Expected ({embedding_dim},), got {mean_embedding.shape}")

            logger.info("Assigning mean embedding to new tokens...")
            # Need to update the *original* embedding layer's weight matrix
            # If it's quantized, this is tricky. The resize function handles re-quantization.
            # Let's assume resize has already allocated space and done default init.
            # We need to update the weights *after* resize but *before* training.
            # If the layer is quantized, we can't directly assign a float vector to a quantized weight row.
            # The resize function should handle the initialization of new rows.
            # The mean initialization should ideally be part of the resize logic or applied to the float weights *before* re-quantization.
            # Given the provided resize logic, it seems it handles new rows.
            # Let's modify the resize function to accept the mean initialization strategy.

            # --- Revised Approach ---
            # The resize function should handle the initialization of new rows.
            # The mean initialization logic should be integrated into resize_model_embeddings.
            # Calling initialize_new_token_embeddings *after* resize is redundant/incorrect for quantized layers.
            # Let's remove this separate function and integrate mean init into resize_model_embeddings.
            pass # This function will be removed.

    except Exception as e:
        logger.error(f"Failed to initialize new token embeddings: {e}", exc_info=True)
        raise


# ======================================================================
# Model Resizing Logic (Integrated Mean Init)
# ======================================================================
def resize_model_embeddings(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    newly_added_token_ids: List[int], # Pass the IDs that were newly added
    init_new_embeddings_with_mean: bool, # Pass the flag
) -> nn.Module:
    """
    Resize the model’s vocabulary‑dependent layers to match ``len(tokenizer)``.
    Works for both float and MLX‑quantised layers.
    Initializes newly added token embeddings based on the mean of existing embeddings if requested.

    * Input embedding  : model.model.embed_tokens
    * Output projection: lm_head / output
    """
    new_vocab = len(tokenizer)
    if new_vocab <= 0:
        raise ValueError(f"Invalid tokenizer length: {new_vocab}")

    logger = logging.getLogger(__name__)

    # simple helper for random init
    def _rand(shape, dtype):
        return mx.random.normal(scale=0.02, shape=shape, dtype=dtype)

    # ------------------------------------------------------------------ #
    # 1. embed_tokens ----------------------------------------------------
    # ------------------------------------------------------------------ #
    old_embed = None
    # Find the embedding layer
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        old_embed = model.model.embed_tokens
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        old_embed = model.transformer.wte
    elif hasattr(model, "embed_tokens"):
        old_embed = model.embed_tokens

    if old_embed is not None and old_embed.weight.shape[0] != new_vocab:
        is_q = isinstance(old_embed, QuantizedEmbedding)
        old_vocab_size = old_embed.weight.shape[0]
        emb_dim = (
            getattr(old_embed, "dims", None) # For QuantizedEmbedding
            or getattr(model.config, "hidden_size", None)
            or old_embed.weight.shape[1]
        )
        if emb_dim is None:
             raise ValueError("Could not determine embedding dimension for resizing.")

        # Get weights in float format for manipulation
        if is_q:
            if not hasattr(old_embed, "to_float"):
                 raise AttributeError(f"Quantized Embedding layer {type(old_embed).__name__} does not have a 'to_float' method required for resizing.")
            float_w = old_embed.to_float().weight.astype(TARGET_FLOAT_DTYPE)
            old_group_size = getattr(old_embed, "group_size", 64)
            old_bits = getattr(old_embed, "bits", 4)
        else:
            float_w = old_embed.weight.astype(TARGET_FLOAT_DTYPE)

        # Handle compressed rows in quantized weights if necessary (logic from provided code)
        if float_w.shape[1] != emb_dim:
            pad_cols = emb_dim - float_w.shape[1]
            logger.info(
                f"Quantised embed_tokens rows are compressed ({float_w.shape[1]} cols); padding with {pad_cols} random cols."
            )
            # Pad with random columns using the target float dtype
            float_w = mx.concatenate((float_w, _rand((float_w.shape[0], pad_cols), TARGET_FLOAT_DTYPE)), axis=1)


        # Create new weight matrix
        new_w = mx.zeros((new_vocab, emb_dim), dtype=TARGET_FLOAT_DTYPE)

        # Copy existing weights
        copy_rows = min(old_vocab_size, new_vocab)
        new_w[:copy_rows] = float_w[:copy_rows]

        # Initialize new rows (beyond old_vocab_size)
        new_rows_start_idx = old_vocab_size
        num_new_rows_needed = new_vocab - old_vocab_size

        if num_new_rows_needed > 0:
            # Default initialization for new rows (random)
            new_w[new_rows_start_idx:] = _rand((num_new_rows_needed, emb_dim), TARGET_FLOAT_DTYPE)
            logger.debug(f"Initialized {num_new_rows_needed} new embedding rows randomly.")

            # Apply mean initialization for specific newly added tokens if requested
            if init_new_embeddings_with_mean and newly_added_token_ids:
                 logger.info(f"Calculating mean of existing embeddings for mean initialization...")
                 if old_vocab_size <= 0:
                      logger.warning("No existing tokens to compute mean embedding from. Skipping mean initialization.")
                 else:
                    try:
                        mean_embedding = mx.mean(float_w[:old_vocab_size], axis=0, keepdims=False)
                        mx.eval(mean_embedding)
                        if mean_embedding.shape != (emb_dim,):
                             raise ValueError(f"Mean embedding shape incorrect: Expected ({emb_dim},), got {mean_embedding.shape}")

                        logger.info(f"Assigning mean embedding to {len(newly_added_token_ids)} newly added tokens...")
                        for token_id in newly_added_token_ids:
                            if 0 <= token_id < new_vocab:
                                new_w[token_id] = mean_embedding # Assign the 1D mean vector
                            else:
                                logger.warning(f"Newly added token ID {token_id} out of bounds for new embedding matrix shape {new_w.shape}. Skipping mean initialization for this ID.")
                        mx.eval(new_w[newly_added_token_ids]) # Evaluate the changes for the specific rows
                        logger.info("Mean embedding initialization complete for specified new tokens.")
                    except Exception as e_mean_init:
                         logger.error(f"Failed during mean embedding initialization: {e_mean_init}", exc_info=True)
                         # Continue with random init for these tokens


        # Create the new embedding layer
        if is_q:
            # Create a temporary float embedding layer to quantize from
            tmp = nn.Embedding(new_vocab, emb_dim)
            tmp.weight = new_w
            mx.eval(tmp.parameters()) # Evaluate before quantization
            new_embed = QuantizedEmbedding.from_embedding(
                tmp,
                group_size=old_group_size,
                bits=old_bits,
            )
        else:
            new_embed = nn.Embedding(new_vocab, emb_dim)
            new_embed.weight = new_w
            mx.eval(new_embed.parameters())

        # Update the model's embedding layer
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
             model.model.embed_tokens = new_embed
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
             model.transformer.wte = new_embed
        elif hasattr(model, "embed_tokens"):
             model.embed_tokens = new_embed

        logger.info(f"Resized embed_tokens: {old_vocab_size} → {new_vocab}")

    # ------------------------------------------------------------------ #
    # 2. lm_head / output ------------------------------------------------
    # ------------------------------------------------------------------ #
    # Find the output projection layer (lm_head or output)
    output_layer = None
    output_parent = None
    output_name = None
    for parent, name in [
        (model, "lm_head"),
        (model, "output"),
        (getattr(model, "model", None), "output"),
    ]:
        if parent is not None and hasattr(parent, name):
            layer = getattr(parent, name)
            if isinstance(layer, (nn.Linear, QuantizedLinear)):
                 output_layer = layer
                 output_parent = parent
                 output_name = name
                 break

    if output_layer is not None:
        is_q = isinstance(output_layer, QuantizedLinear)
        old_vocab_size, hidden = output_layer.weight.shape
        hidden_size = getattr(model.config, "hidden_size", hidden) # Get hidden size from config, fallback to weight shape
        if hidden_size is None:
             raise ValueError("Could not determine hidden size for output layer resizing.")


        if old_vocab_size != new_vocab:
            if is_q:
                 if not hasattr(output_layer, "to_float"):
                      raise AttributeError(f"Quantized Linear layer {type(output_layer).__name__} does not have a 'to_float' method required for resizing.")
                 float_head = output_layer.to_float()
                 old_group_size = getattr(output_layer, "group_size", 64)
                 old_bits = getattr(output_layer, "bits", 4)
            else:
                 float_head = output_layer

            base_w = float_head.weight.astype(TARGET_FLOAT_DTYPE)
            base_b = float_head.bias.astype(TARGET_FLOAT_DTYPE) if getattr(float_head, "bias", None) is not None else None

            # Create new weight matrix and bias vector
            new_w = mx.zeros((new_vocab, hidden_size), dtype=TARGET_FLOAT_DTYPE)
            new_b = mx.zeros((new_vocab,), dtype=TARGET_FLOAT_DTYPE) if base_b is not None else None

            # Copy existing weights and biases
            copy_rows = min(old_vocab_size, new_vocab)
            new_w[:copy_rows] = base_w[:copy_rows]
            if new_b is not None:
                 new_b[:copy_rows] = base_b[:copy_rows]

            # Initialize new rows (beyond old_vocab_size) with random values for weights and zeros for biases
            num_new_rows_needed = new_vocab - old_vocab_size
            if num_new_rows_needed > 0:
                 new_rows_start_idx = old_vocab_size
                 new_w[new_rows_start_idx:] = _rand((num_new_rows_needed, hidden_size), TARGET_FLOAT_DTYPE)
                 logger.debug(f"Initialized {num_new_rows_needed} new output layer rows randomly.")
                 # Biases are already initialized to zeros

                 # No special initialization needed for output layer biases/weights for specific tokens usually.
                 # The mean embedding initialization only applies to the input embeddings.


            # Create the new output layer
            if is_q:
                tmp = nn.Linear(hidden_size, new_vocab, bias=base_b is not None)
                tmp.weight = new_w
                if new_b is not None:
                    tmp.bias = new_b
                mx.eval(tmp.parameters()) # Evaluate before quantization
                new_head = QuantizedLinear.from_linear(
                    tmp,
                    group_size=old_group_size,
                    bits=old_bits,
                )
            else:
                new_head = nn.Linear(hidden_size, new_vocab, bias=base_b is not None)
                new_head.weight = new_w
                if new_b is not None:
                    new_head.bias = new_b
                mx.eval(new_head.parameters())

            # Update the model's output layer
            setattr(output_parent, output_name, new_head)
            logger.info(f"Resized {output_name}: {old_vocab_size} → {new_vocab}")
        else:
             logger.info(f"Output layer '{output_name}' already matches new vocabulary size ({new_vocab}). No resize needed.")

    else:
        logger.warning("Could not find a suitable output projection layer (lm_head or output) to resize.")


    logger.info("Embedding & head resize complete.")
    return model


def check_and_set_resume_path(args: TrainingArgs):
    """
    Checks if resuming is requested and potentially uses the 'latest' symlink
    in the output directory if no specific checkpoint is given.
    Modifies args.resume_from_checkpoint in-place.
    """
    logger = logging.getLogger(__name__)

    if args.output_dir:
        output_dir_path = Path(args.output_dir)

        if not args.resume_from_checkpoint:
            logger.info("No specific --resume-from-checkpoint provided.")
            if output_dir_path.is_dir():
                latest_symlink_path = output_dir_path / "latest"
                if latest_symlink_path.is_symlink():
                    try:
                        resolved_checkpoint_path = latest_symlink_path.resolve()
                        if resolved_checkpoint_path.is_dir():
                            args.resume_from_checkpoint = str(resolved_checkpoint_path)
                            logger.info(f"Found 'latest' symlink pointing to existing directory.")
                            logger.info(f"Setting resume_from_checkpoint to: [cyan]{args.resume_from_checkpoint}[/cyan]")
                        else:
                            logger.warning(f"'latest' symlink exists at {latest_symlink_path} but points to a non-directory: {resolved_checkpoint_path}. Skipping automatic resume.")
                    except OSError as e:
                        logger.warning(f"Could not resolve 'latest' symlink at {latest_symlink_path}: {e}. Skipping automatic resume.")
                    except Exception as e:
                        logger.error(f"An unexpected error occurred while resolving 'latest' symlink at {latest_symlink_path}: {e}", exc_info=True)
                else:
                    logger.info(f"'latest' symlink not found in existing output directory {args.output_dir}. Starting new run.")
            else:
                logger.info(f"Output directory {args.output_dir} does not exist. Starting new run.")
        else:
            logger.info(f"Resuming from explicitly provided checkpoint: [cyan]{args.resume_from_checkpoint}[/cyan]")
    else:
        logger.error("args.output_dir is not set. Cannot check for 'latest' symlink. This should not happen with current args validation.")


def parse_arguments() -> TrainingArgs:
    """Parses command-line arguments using argparse, driven by the TrainingArgs dataclass fields."""
    parser = argparse.ArgumentParser(
        description="Fine-tunes a language model using GRPO with MLX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if not is_dataclass(TrainingArgs):
        raise TypeError("TrainingArgs must be a dataclass.")

    for f in fields(TrainingArgs):
        if f.init is False:
            continue
        cliname = f"--{f.name.replace('_', '-')}"
        kwargs: Dict[str, Any] = {
            "dest": f.name,
            "help": f.metadata.get("help", f.name),
        }
        field_type = f.type
        origin_type = get_origin(field_type)
        type_args = get_args(field_type)
        is_optional_type = origin_type is Union and type(None) in type_args
        has_default = f.default is not MISSING or f.default_factory is not MISSING
        default_value = (
            f.default_factory() if f.default_factory is not MISSING else f.default
        )

        if field_type is bool:
            if has_default and default_value is True:
                cliname = f"--no-{f.name.replace('-', '_')}" # Use underscore for consistency
                kwargs["action"] = "store_false"
                kwargs["help"] += " (disables this option)"
            else:
                kwargs["action"] = "store_true"
        elif is_optional_type:
            actual_type = str # Default to str for optional types
            for arg_t in type_args:
                if arg_t is not type(None) and arg_t in [int, float, str, dict]: # Add dict as a possible type
                    actual_type = arg_t
                    break
            kwargs["type"] = actual_type
            kwargs["default"] = default_value if has_default else None
            kwargs["required"] = False
        else:
            kwargs["type"] = field_type
            if has_default:
                kwargs["default"] = default_value
                kwargs["required"] = False
            else:
                kwargs["required"] = True
                kwargs["help"] = f"(Required) {kwargs['help']}" if kwargs["help"] else "(Required)"

        if (kwargs.get("required", False) and f.default is MISSING and f.default_factory is MISSING):
             kwargs.pop("default", None)

        # Handle dict type for lr_schedule_config
        if field_type is Dict:
             kwargs["type"] = json.loads # Parse JSON string into dict

        parser.add_argument(cliname, **kwargs)

    parsed_args = parser.parse_args()
    args_dict = vars(parsed_args)

    try:
        training_args_instance = TrainingArgs(**args_dict)
        return training_args_instance
    except (TypeError, ValueError) as e:
        console.print(f"\n[bold red]ERROR:[/bold red] Argument validation failed: {e}")
        parser.print_help()
        sys.exit(1)


# --- Main Execution ---
def main():
    """Main function to parse arguments, set up, and run training."""
    # Memory limit attempt (keep as is)
    # try: limit_memory(60 * 1024) # limit_memory expects MB
    # except Exception as e: print(f"Warning: Failed to set memory limit: {e}", file=sys.stderr)

    args = parse_arguments()
    check_and_set_resume_path(args)

    # --- Setup Logging ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    rich_handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        show_path=log_level <= logging.DEBUG,
        log_time_format="[%X]",
        console=console,
        level=log_level,
    )
    handlers: List[logging.Handler] = [rich_handler]
    log_file_msg = "[dim]File logging disabled.[/]"
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "training_debug.log"
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        log_fmt = "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        file_handler.setFormatter(logging.Formatter(log_fmt, date_fmt))
        handlers.append(file_handler)
        log_file_msg = f"Debug Log: [link=file://{log_file.resolve()}]{log_file.name}[/]"
    except Exception as e:
        print(f"Warning: Could not set up file logging in '{args.output_dir}': {e}")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    mlx_log_level = logging.INFO if not args.verbose else logging.DEBUG
    if log_level > logging.DEBUG: mlx_log_level = logging.INFO
    logging.getLogger("mlx").setLevel(mlx_log_level)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logger.info(f"Logging Level: Console={logging.getLevelName(log_level)}, File=DEBUG. {log_file_msg}")

    if args.use_wandb and not WANDB_AVAILABLE:
        logger.warning("WandB requested but not installed (`pip install wandb`). Disabling.")
        args.use_wandb = False
    if args.train_dataset_path is None and args.dataset_name is None:
        logger.critical("No training dataset source specified. Please provide --train-dataset-path or --dataset-name. Exiting.")
        sys.exit(1)
    if args.dataset_name is not None and not DATASETS_AVAILABLE:
        logger.critical("Hugging Face dataset specified (--dataset-name) but `datasets` library is not installed (`pip install datasets`). Exiting.")
        sys.exit(1)
    if args.reward_content_type.lower() == "math_eval":
         console.print("[red bold]WARNING: 'math_eval' reward uses eval(), posing a SEVERE SECURITY RISK. Only use on trusted model outputs/datasets.[/]")


    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    logger.info("Registered signal handlers for SIGINT and SIGTERM.")
    if SAVE_ON_EXIT_FLAG_PATH.exists():
        logger.warning(f"Removing stale save flag file from previous run: {SAVE_ON_EXIT_FLAG_PATH}")
        try: SAVE_ON_EXIT_FLAG_PATH.unlink()
        except OSError as e: logger.error(f"Failed to remove stale save flag: {e}", exc_info=False)

    logger.info("[bold green]--- Effective Training Configuration ---[/]")
    config_table = Table(show_header=False, box=None, padding=(0, 1), title="Arguments")
    config_table.add_column("Parameter", style="dim cyan", justify="right")
    config_table.add_column("Value", style="white")
    args_dict_display = asdict(args)
    for key, value in sorted(args_dict_display.items()):
        display_value = str(value)
        if ("path" in key.lower() or "token" in key.lower()) and isinstance(value, str) and len(value) > 10:
             display_value = f"{value[:5]}...{value[-5:]}" # Basic masking for long paths/tokens
        if key == "lr_schedule_config":
             display_value = json.dumps(value) # Show LR config as JSON string
        config_table.add_row(key, display_value)
    console.print(config_table)
    logger.info("[bold green]------------------------------------[/]")

    try:
        train_dset, val_dset = get_dataset(args)
    except Exception as e:
        logger.critical(f"Failed to load dataset(s): {e}", exc_info=True)
        sys.exit(1)
    if train_dset is None:
        logger.critical("Training dataset could not be loaded. Exiting.")
        sys.exit(1)
    logger.info(f"Training dataset loaded: {len(train_dset)} samples.")
    if val_dset:
        logger.info(f"Validation dataset loaded: {len(val_dset)} samples.")
    else:
        logger.info("No validation dataset loaded.")

    actor_model: Optional[nn.Module] = None
    tokenizer: Optional[TokenizerWrapper] = None
    ref_model: Optional[nn.Module] = None
    draft_model: Optional[nn.Module] = None # Initialize draft model
    model_config: Optional[Dict] = None
    optimizer: Optional[optim.Optimizer] = None
    lr_scheduler: Optional[Callable[[int], float]] = None # Initialize LR scheduler
    start_step = 0
    start_updates = 0
    rng_restored_from_checkpoint = False
    newly_added_token_ids = []

    # --- Load Tokenizer First (Needed for Model Loading/Resizing) ---
    tokenizer_path_to_load = Path(args.model_path) # Default to model path
    if args.resume_from_checkpoint:
         tokenizer_path_to_load = Path(args.resume_from_checkpoint) # Load from checkpoint if resuming

    try:
        tokenizer = load_tokenizer(tokenizer_path_to_load)
        logger.info(f"Tokenizer loaded from {tokenizer_path_to_load}: Vocab size {tokenizer.vocab_size}")

        # --- Add Special Tokens & Identify Newly Added IDs ---
        special_tokens_to_add = {
            "additional_special_tokens": sorted(list(set([
                args.think_start_tag.strip(),
                args.think_end_tag.strip(),
                args.answer_start_tag.strip(),
                args.answer_end_tag.strip(),
            ])))
        }
        logger.info(f"Adding special tokens to tokenizer: {special_tokens_to_add['additional_special_tokens']}")
        num_tokens_before = tokenizer.vocab_size
        num_added = tokenizer.add_special_tokens(special_tokens_to_add)
        num_tokens_after = tokenizer.vocab_size
        num_new_tokens = num_tokens_after - num_tokens_before

        if num_new_tokens > 0:
            logger.info(f"Added {num_new_tokens} new special tokens. New vocab size: {num_tokens_after}.")
            newly_added_token_ids = list(range(num_tokens_before, num_tokens_after))
        else:
            logger.info("No new special tokens were added (they might exist already in the tokenizer).")

        # --- Check PAD Token ---
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                logger.warning(f"Tokenizer missing PAD token. Using EOS token '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id}) as PAD.")
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.pad_token = tokenizer.eos_token
                # Update underlying tokenizer if possible (best effort)
                if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "pad_token_id"):
                    tokenizer.tokenizer.pad_token_id = tokenizer.eos_token_id
                if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "pad_token"):
                    tokenizer.tokenizer.pad_token = tokenizer.eos_token
            else:
                logger.critical("Tokenizer has no PAD token and no EOS token! Cannot proceed.")
                sys.exit(1)
        else:
            logger.info(f"Using Tokenizer PAD token ID: {tokenizer.pad_token_id} ('{tokenizer.pad_token}')")

        # Add PAD token to special_tokens_map if it's not there after setting it
        if tokenizer.pad_token and (tokenizer.pad_token not in tokenizer.special_tokens_map or tokenizer.special_tokens_map.get("pad_token") != tokenizer.pad_token):
            tokenizer.special_tokens_map["pad_token"] = tokenizer.pad_token
            logger.debug("Added PAD token to tokenizer.special_tokens_map.")


        # --- Conditional Seeding ---
        global mlx_rng_key  # Use global key
        if not rng_restored_from_checkpoint:
            logger.info(
                f"Setting random seed: {args.seed} (RNG state not loaded from checkpoint)."
            )
            random.seed(args.seed)
            np.random.seed(args.seed)
            # mlx_rng_key = mx.random.key(args.seed) # This creates a *new* key, better to set the global state
            mlx.core.random.seed(args.seed)  # Set the global MLX RNG state using the seed
            mlx_rng_key = mx.random.key(0)  # Get the initial key state after seeding
            mx.eval(mlx_rng_key)  # Evaluate the initial key
        else:
            logger.info(
                "Skipping initial seeding as RNG state was restored from checkpoint."
            )
            # The load_checkpoint function should have already updated mlx_rng_key and potentially called mx.random.set_key().
            # Ensure the global key is still the one loaded if needed downstream, though mx.random functions use the internal state.
            # The state was set by mx.random.set_key() inside load_checkpoint.

        # Evaluate all parameters one last time before training starts, especially after init/loading
        logger.info("Evaluating final model state before starting training loop...")

    except Exception as e:
        logger.critical(f"Failed to load or configure tokenizer: {e}", exc_info=True)
        sys.exit(1)


    # --- Load Models ---
    try:
        # Load base actor model structure and weights (or just structure if resuming)
        model_path_to_load = Path(args.model_path)
        if args.resume_from_checkpoint:
             # When resuming, load the model structure from the checkpoint path
             # The weights will be loaded by load_checkpoint later
             model_path_to_load = Path(args.resume_from_checkpoint)
             logger.info(f"Loading actor model structure from checkpoint path: [cyan]{model_path_to_load}[/cyan]...")
             # Use load with strict=False to load structure without requiring all weights match
             actor_model, _ = load(model_path_to_load, strict=False)
             logger.info(f"Actor model structure loaded from checkpoint path: {type(actor_model).__name__}")
             # Load config from the checkpoint path
             model_config = load_config(model_path_to_load)
             actor_model.config = model_config
             logger.info(f"Model config loaded from checkpoint for {model_config.get('model_type', 'Unknown Type')}")

        else:
             # Load base actor model and tokenizer (tokenizer is ignored as we loaded it already)
             logger.info(f"Loading base actor model from: [cyan]{model_path_to_load}[/cyan]...")
             actor_model, _ = load(model_path_to_load)
             if not isinstance(actor_model, nn.Module):
                 raise TypeError(f"Loaded actor model is not an nn.Module: {type(actor_model)}")
             logger.info(f"Base actor model loaded: {type(actor_model).__name__}")
             # Load config from the base model path
             model_config = load_config(model_path_to_load)
             actor_model.config = model_config
             logger.info(f"Model config loaded for {model_config.get('model_type', 'Unknown Type')}")

        print(actor_model)
        # --- Resize Actor Model Embeddings (AFTER loading structure, BEFORE loading checkpoint weights) ---
        if actor_model.model.embed_tokens.weight.shape[0] != tokenizer.vocab_size:
             logger.info(f"Resizing actor model embeddings from {actor_model.model.embed_tokens.weight.shape[0]} to {tokenizer.vocab_size}...")
             actor_model.model = resize_model_embeddings(
                 actor_model.model,
                 tokenizer._tokenizer,
                 newly_added_token_ids,
                 args.init_new_embeddings_with_mean,
             )
             mx.eval(actor_model.parameters()) # Evaluate after resize
             logger.info(f"Actor model embeddings resized to {tokenizer.vocab_size}.")
        elif newly_added_token_ids and args.init_new_embeddings_with_mean:
             # If vocab size matches but we added tokens (meaning they were already there)
             # and mean init is requested, we could potentially re-initialize them here.
             # However, the resize function handles this now if the vocab size *changes*.
             # If vocab size doesn't change, their embeddings are already loaded with the base model.
             # Skipping mean init if vocab size didn't change.
             logger.debug("Vocab size did not change. Skipping mean embedding initialization for tokens that might have been newly added to tokenizer but already existed in model vocab.")


        # Load reference model
        logger.info(f"Loading reference model from: [cyan]{args.ref_model_path}[/cyan]...")
        ref_model, ref_tokenizer = load(Path(args.ref_model_path))
        if not isinstance(ref_model, nn.Module):
             raise TypeError(f"Loaded reference model is not an nn.Module: {type(ref_model)}")

        # Resize Reference Model Embeddings if vocab size differs
        if ref_tokenizer.vocab_size != tokenizer.vocab_size:
            logger.warning(f"Reference model tokenizer vocab size ({ref_tokenizer.vocab_size}) does not match actor tokenizer vocab size ({tokenizer.vocab_size}). Resizing ref model embeddings.")
            # Pass the *same* newly added token IDs and mean init flag to ref model resize
            ref_model = resize_model_embeddings(
                ref_model,
                ref_tokenizer, # Use ref_tokenizer for its internal state if needed by resize
                newly_added_token_ids, # Use the IDs added to the actor tokenizer
                args.init_new_embeddings_with_mean,
            )
            mx.eval(ref_model.parameters())
            logger.info(f"Reference model embeddings resized to {tokenizer.vocab_size}.")
        elif newly_added_token_ids and args.init_new_embeddings_with_mean:
             # Same logic as actor model: skip mean init if vocab size didn't change
             logger.debug("Ref model vocab size did not change. Skipping mean embedding initialization.")


        ref_model.freeze()
        ref_model.eval()
        mx.eval(ref_model.parameters())
        logger.info(f"Reference model loaded ({type(ref_model).__name__}) and frozen.")

        # Load draft model if speculative decoding is enabled
        if args.use_speculative_decoding:
            logger.info(f"Loading draft model from: [cyan]{args.draft_model_path}[/cyan] for speculative decoding...")
            draft_model, draft_tokenizer = load(Path(args.draft_model_path))
            if not isinstance(draft_model, nn.Module):
                 raise TypeError(f"Loaded draft model is not an nn.Module: {type(draft_model)}")

            # Resize Draft Model Embeddings if vocab size differs
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                logger.warning(f"Draft model tokenizer vocab size ({draft_tokenizer.vocab_size}) does not match actor tokenizer vocab size ({tokenizer.vocab_size}). Resizing draft model embeddings.")
                # Pass the *same* newly added token IDs and mean init flag to draft model resize
                draft_model = resize_model_embeddings(
                    draft_model,
                    draft_tokenizer, # Use draft_tokenizer for its internal state if needed by resize
                    newly_added_token_ids, # Use the IDs added to the actor tokenizer
                    args.init_new_embeddings_with_mean,
                )
                mx.eval(draft_model.parameters())
                logger.info(f"Draft model embeddings resized to {tokenizer.vocab_size}.")
            elif newly_added_token_ids and args.init_new_embeddings_with_mean:
                 # Same logic as actor model: skip mean init if vocab size didn't change
                 logger.debug("Draft model vocab size did not change. Skipping mean embedding initialization.")

            draft_model.eval() # Draft model is typically in eval mode
            mx.eval(draft_model.parameters())
            logger.info(f"Draft model loaded ({type(draft_model).__name__}).")
        else:
            logger.info("Speculative decoding is disabled.")


    except Exception as e:
        logger.critical(f"Failed to load models or resize embeddings: {e}", exc_info=True)
        sys.exit(1)


    # --- Initialize Optimizer ---
    try:
        trainable_params_dict = actor_model.trainable_parameters()
        trainable_params_list = tree_flatten(trainable_params_dict)
        trainable_param_arrays = [p for _, p in trainable_params_list]

        if not trainable_param_arrays:
            logger.critical("No trainable parameters found! Check model freezing. Cannot train. Exiting.")
            raise ValueError("No trainable parameters found.")
        else:
            num_trainable = sum(p.size for p in trainable_param_arrays)
            logger.info(f"Creating optimizer for {len(trainable_param_arrays)} trainable tensors ({num_trainable:,} parameters).")

        optimizer = optim.AdamW(
            learning_rate=args.learning_rate, # Initial LR, will be updated by scheduler
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            weight_decay=args.optimizer_weight_decay,
        )
        logger.info(f"Initialized AdamW optimizer.")

    except Exception as e:
        logger.critical(f"Failed to initialize optimizer: {e}", exc_info=True)
        sys.exit(1)

    # --- Build Learning Rate Scheduler ---
    try:
        # Ensure total_steps is set in the schedule config if using total_steps based schedulers
        if args.lr_schedule_config.get("name") in ["linear_schedule", "cosine_decay"] and len(args.lr_schedule_config.get("arguments", [])) < 3:
             # Assume the last argument should be total_steps
             if len(args.lr_schedule_config.get("arguments", [])) == 2:
                  args.lr_schedule_config["arguments"].append(args.num_training_steps)
                  logger.info(f"Added num_training_steps ({args.num_training_steps}) to LR schedule arguments.")
             else:
                  logger.warning(f"LR schedule '{args.lr_schedule_config.get('name')}' requires 3 arguments (start_lr, end_lr, total_steps). Using default LR.")
                  # Fallback to constant LR
                  lr_scheduler = lambda step: args.learning_rate
        else:
             # Use the provided config
             pass # Config is assumed correct or build_schedule will raise error

        lr_scheduler = build_schedule(args.lr_schedule_config)
        logger.info(f"Built LR scheduler: {args.lr_schedule_config.get('name')}")

    except Exception as e:
        logger.error(f"Failed to build LR scheduler from config {args.lr_schedule_config}: {e}. Using constant learning rate.", exc_info=True)
        lr_scheduler = lambda step: args.learning_rate # Fallback to constant LR


    # --- Load Checkpoint State (AFTER models/optimizer are initialized and model resized) ---
    if args.resume_from_checkpoint:
        try:
            # This loads weights into actor_model and state into optimizer
            start_step, start_updates, rng_restored_from_checkpoint = load_checkpoint(
                Path(args.resume_from_checkpoint), actor_model, optimizer
            )
        except Exception as e:
            logger.critical(f"Failed to load checkpoint state: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("Starting new training run (not resuming from checkpoint).")


    # --- Conditional Seeding ---
    if not rng_restored_from_checkpoint:
        logger.info(f"Setting random seed: {args.seed} (RNG state not loaded from checkpoint).")
        random.seed(args.seed)
        np.random.seed(args.seed)
        mlx.core.random.seed(args.seed) # Set the global MLX RNG state
        # No need to manage a separate mlx_rng_key variable unless needed for specific ops

    else:
        logger.info("Skipping initial seeding as RNG state was restored from checkpoint.")
        # The load_checkpoint function should have already called mx.random.set_state().

    # Evaluate all parameters one last time before training starts
    logger.info("Evaluating final model state before starting training loop...")
    mx.eval(actor_model.parameters(), ref_model.parameters(), optimizer.state)
    if draft_model: mx.eval(draft_model.parameters())
    logger.info("Initial state evaluation complete.")


    # --- Start Training Process ---
    logger.info("[bold blue]>>> Starting GRPO Training Process <<<[/]")
    training_start_time = time.monotonic()
    exit_code = 0
    try:
        train(
            args,
            actor_model,
            ref_model,
            draft_model, # Pass draft model
            model_config,
            tokenizer,
            train_dset,
            val_dset,
            optimizer,
            lr_scheduler, # Pass LR scheduler
            start_step,
            start_updates,
        )
    except Exception as train_err:
        logger.critical("Training process terminated due to an unhandled exception.", exc_info=True)
        console.print_exception(show_locals=args.verbose)
        exit_code = 1
    finally:
        total_duration_mins = (time.monotonic() - training_start_time) / 60
        logger.info(f"Total script execution time: {total_duration_mins:.2f} minutes.")
        logger.info("[bold blue]>>> Training Script Finished <<<[/]")
        logging.shutdown()
        sys.exit(exit_code)


# --- Entry Point ---
if __name__ == "__main__":
    rprint("\n[bold underline]GRPO MLX Trainer (Aligned)[/]")
    rprint("[dim] - Uses `tokenizer.apply_chat_template`.")
    rprint("[dim] - Adds special tokens, resizes embeddings, and initializes them.")
    rprint(f"[red bold] - WARNING: 'math_eval' reward uses `eval()`. Use --reward-content-type jaccard unless input is fully trusted.[/]")
    rprint("[dim] - [green]Using GRPO loss with group-based advantage normalization.[/]")
    rprint("[dim] - Checkpoints include training state, optimizer state, tokenizer, and weights.")
    rprint("[dim] - Supports resuming from last checkpoint via 'latest' symlink.")
    rprint("[dim] - Includes support for Speculative Decoding (if draft model provided).")
    rprint("[dim] - Includes support for KV Cache Quantization.")
    rprint("[dim] - Uses LR Scheduling from config.")
    rprint("[dim] - Includes Gradient Checkpointing (requires model-specific layer types).")
    rprint("-" * 30 + "\n")
    main()
# ```

# **Documentation for the GRPO MLX Trainer Script**

# This documentation describes how to use the `mlx_grpo_trainer_aligned.py` script for fine-tuning a language model using the GRPO algorithm with MLX.

# **Process: Fine-tuning a Language Model using GRPO**

# This script implements a training loop to fine-tune an MLX language model using the Grouped Reinforcement Learning with Policy Optimization (GRPO) algorithm. It generates multiple responses per prompt (rollouts), calculates rewards based on a defined reward function, computes advantages, and updates the model using a policy gradient loss with a KL penalty against a reference model.

# **1. System Requirements**

# *   **Operating System:** macOS (MLX is currently macOS-only).
# *   **Hardware:** Apple Silicon (M1, M2, M3, etc.) CPU or GPU. Training performance is significantly better with a GPU. Sufficient RAM/Unified Memory is required to load the model(s) and process batch data.
# *   **Software:**
#     *   Python 3.8 or higher.
#     *   MLX library (`pip install mlx`).
#     *   `mlx-lm` library (`pip install -U mlx-lm`).
#     *   `datasets` library (`pip install datasets`) - Required for loading data from Hugging Face Hub or local JSONL files.
#     *   `rich` library (`pip install rich`) - Used for enhanced console output and progress bars.
#     *   `psutil` library (`pip install psutil`) - Optional, for memory monitoring.
#     *   `wandb` library (`pip install wandb`) - Optional, for logging metrics to Weights & Biases.
#     *   `llama_rl` library (`pip install llama_rl`) - Optional, required for the `_save_and_commit` utility for atomic checkpoint saving and the external reward function if used. If not installed, fallback dummy functions are used, and checkpointing may not be atomic.

# **2. Access Needs**

# *   Read access to the base model files (`--model-path`).
# *   Read access to the reference model files (`--ref-model-path`).
# *   Read access to the draft model files (`--draft-model-path`) if using speculative decoding.
# *   Read access to the dataset files (local JSONL or Hugging Face Hub access).
# *   Write access to the output directory (`--output-dir`) for saving checkpoints, logs, and metrics.
# *   Network access to Hugging Face Hub if loading models or datasets from there.
# *   Network access to Weights & Biases if `--use-wandb` is enabled.

# **3. Step Sequences**

# 1.  **Install Dependencies:** Ensure all required Python libraries are installed (`pip install -r requirements.txt` or install individually).
# 2.  **Prepare Data:** Have your training and validation data ready. This can be a local JSONL file or a dataset available on the Hugging Face Hub. The dataset must contain columns for the input prompt and the reference answer, specified by `--dataset-prompt-key` and `--dataset-answer-key`.
# 3.  **Choose Models:** Select a base model for training (`--model-path`), a reference model for the KL penalty (`--ref-model-path`), and optionally a draft model for speculative decoding (`--draft-model-path`). These can be local paths or Hugging Face Hub IDs. The models must be compatible with MLX and ideally use the same tokenizer or have compatible architectures for resizing.
# 4.  **Configure Training:** Define training parameters using command-line arguments (see Configuration Settings below). Pay close attention to `--output-dir`, `--model-path`, data paths/names, batch sizes (`--ppo-batch-size`, `--num-rollout-samples`, `--grad-accum-steps`), learning rate and schedule (`--learning-rate`, `--lr-schedule-config`), reward configuration (`--reward-format-weight`, `--reward-content-weight`, `--reward-content-type`), and advanced features like speculative decoding (`--use-speculative-decoding`, `--draft-model-path`, `--num-draft-tokens`), KV cache quantization (`--kv-bits`, `--kv-group-size`, `--quantized-kv-start`), and gradient checkpointing (`--use-grad-checkpointing`, `--grad-checkpoint-layers`).
# 5.  **Run the Script:** Execute the Python script from your terminal.
#     ```bash
#     python mlx_grpo_trainer_aligned.py --output-dir /path/to/output --model-path mlx-community/Llama-3-8B-Instruct-4bit --dataset-name your_dataset/name --num-training-steps 1000 ... [other args]
#     ```
# 6.  **Monitor Training:** Observe the console output for progress updates, loss, reward metrics, and gradient norms. If using WandB, monitor the run in the WandB UI. Check the debug log file in the output directory for detailed information.
# 7.  **Interrupt Gracefully (Optional):** Press `Ctrl+C` once to request a graceful shutdown. The script will attempt to save a final checkpoint before exiting. Press `Ctrl+C` again to force an immediate exit without saving.
# 8.  **Resume Training (Optional):** To resume training from a checkpoint, either specify the checkpoint directory path using `--resume-from-checkpoint /path/to/checkpoint_dir` or ensure the output directory contains a `latest` symlink pointing to the desired checkpoint and run the script with the same `--output-dir`.
# 9.  **Analyze Results:** After training completes or is interrupted, analyze the metrics logged in the console, WandB, and the `training_metrics.csv` file in the output directory. Inspect the generated samples logged during evaluation.
# 10. **Use Fine-tuned Model:** The final model weights (or adapter weights if LoRA was used, though LoRA is removed in this version) are saved in the checkpoint directory. You can load these weights with the model structure for inference or further use.

# **4. Error Handling**

# *   **Argument Validation:** The script uses `argparse` and dataclass `__post_init__` for initial validation of command-line arguments. Invalid arguments will result in an error message and script exit.
# *   **File/Directory Not Found:** Checks are performed for the existence of the output directory, dataset files, and checkpoint files. Missing required files will cause the script to exit with a `FileNotFoundError` or a critical log message.
# *   **Import Errors:** Checks are performed for optional dependencies (`datasets`, `psutil`, `wandb`, `llama_rl`). If a required dependency for a requested feature is missing, a warning or critical error is logged, and the feature might be disabled or the script might exit.
# *   **Model/Tokenizer Loading Errors:** Errors during model or tokenizer loading (e.g., incompatible format, network issues) are caught, logged as critical errors, and the script exits.
# *   **Dataset Loading Errors:** Errors during dataset loading from local files or Hugging Face Hub are caught, logged as critical errors, and the script exits.
# *   **NaN/Inf Loss:** The `grpo_loss` function explicitly checks for `NaN` or `Inf` values in the calculated loss. If detected, an error is logged with component statistics, and a `ValueError` is raised, which is caught in the training loop to skip the current update step and log the issue. If NaN/Inf persists, it might indicate instability requiring hyperparameter tuning or model/data inspection.
# *   **Optimizer/Gradient Errors:** Errors during gradient calculation or optimizer application (e.g., shape mismatches, numerical instability) are caught, logged, and the script attempts to continue, skipping the problematic update step.
# *   **Generation Errors:** Errors during rollout generation (either non-speculative or speculative) are caught, logged, and the script attempts to pad the remaining sequence length with PAD tokens to allow the batch to proceed through the loss calculation, although the resulting loss/reward for that sample might be zero or inaccurate.
# *   **Reward Calculation Errors:** Errors during reward calculation for a specific sample are caught, logged, and the reward for that sample is set to 0.0 to prevent halting the training.
# *   **Checkpoint Saving/Loading Errors:** Errors during checkpoint operations are logged. Saving attempts to clean up partial saves. Loading logs warnings for non-critical issues (like missing optimizer state) and critical errors for essential missing files or compatibility issues, leading to script exit.
# *   **Signal Handling:** `SIGINT` (Ctrl+C) and `SIGTERM` signals are caught to trigger a graceful shutdown, attempting to save a final checkpoint. A second signal forces immediate exit.
# *   **General Exceptions:** A broad `except Exception` block in `main` and `train` catches unexpected errors, logs them critically, prints a traceback (if verbose), sets an error exit code, and attempts an emergency checkpoint save before the script finishes.

# **5. Security Protocols**

# *   **`math_eval` Warning:** The script includes prominent warnings in the console output and documentation about the severe security risk of using the `math_eval` reward function due to its use of `eval()`. It is strongly recommended to use the `jaccard` reward type unless the model outputs are guaranteed to be safe.
# *   **Limited `eval` Scope:** The `math_eval_reward` function attempts to limit the scope of `eval()` by providing restricted `globals` and `locals` dictionaries and performing basic input cleaning. However, this is not a foolproof security measure.
# *   **Trusted Inputs:** The script assumes that the input dataset prompts and reference answers are trusted.
# *   **Checkpoint Integrity:** The script relies on the `_save_and_commit` utility (assumed from `llama_rl.utils` or a fallback) for atomic checkpoint saving to reduce the risk of corrupted files during interruptions.

# **6. Backup Procedures**

# *   **Periodic Checkpointing:** The script automatically saves checkpoints periodically based on the `--save-every` argument (number of optimizer updates).
# *   **End-of-Training Checkpoint:** A final checkpoint is saved upon reaching the target number of training steps.
# *   **Interruption Checkpoint:** The script attempts to save a checkpoint when a graceful shutdown signal (SIGINT, SIGTERM) is received or when a critical error occurs.
# *   **Checkpoint Rotation:** The script can automatically delete older checkpoints, keeping only the most recent `--keep-last` checkpoints to manage disk space.
# *   **Manual Backup:** Users should manually back up the entire output directory periodically to an external storage location for disaster recovery, especially if the training run is long or critical.

# **7. Recovery Steps**

# *   **Resume from Checkpoint:** If training is interrupted (gracefully or due to an error/crash) and a checkpoint was saved, training can be resumed from that point.
#     *   Ensure the same script version and environment are used.
#     *   Run the script again with the *same* `--output-dir` and other training parameters.
#     *   Specify the path to the checkpoint directory using `--resume-from-checkpoint /path/to/checkpoint_dir`.
#     *   Alternatively, if the output directory contains a `latest` symlink pointing to the desired checkpoint, simply running the script with the same `--output-dir` will automatically attempt to resume from the `latest` checkpoint.
#     *   The script will load the model weights, optimizer state, training progress (step, updates), and RNG state from the checkpoint.
# *   **Handling Corrupted Checkpoints:** If a checkpoint is corrupted and cannot be loaded, the script will log a critical error and exit. You may need to attempt resuming from an older checkpoint or start a new training run.
# *   **Handling Data Corruption:** If the dataset files become corrupted, training will fail during data loading or batch processing. You will need to restore the dataset from a backup or re-download it.

# **8. Include (Screenshots/Diagrams, Command Sequences, Configuration Settings, Validation Steps, Common Errors/Fixes)**

# *   **Screenshots/Diagrams:** Not applicable in this text-based format. A diagram would typically show the GRPO loop (Generate Rollouts -> Calculate Rewards -> Calculate Advantages -> Calculate Loss -> Optimizer Step) and the components involved (Actor Model, Reference Model, Draft Model, Dataset, Tokenizer, Optimizer).
# *   **Command Sequences:** See Step 5 above for a basic example. A more detailed example would include specific arguments:
#     ```bash
#     # Example: Start a new training run
#     python mlx_grpo_trainer_aligned.py \
#       --output-dir ./grpo_output \
#       --model-path mlx-community/Llama-3-8B-Instruct-4bit \
#       --ref-model-path mlx-community/Llama-3-8B-Instruct-4bit \
#       --dataset-name gsm8k \
#       --dataset-config main \
#       --dataset-train-split train \
#       --dataset-val-split test \
#       --ppo-batch-size 4 \
#       --num-rollout-samples 4 \
#       --grad-accum-steps 2 \
#       --num-training-steps 1000 \
#       --save-every 50 \
#       --eval-every 100 \
#       --learning-rate 3e-5 \
#       --lr-schedule-config '{"name": "cosine_decay", "arguments": [3e-5, 1e-7, 1000], "warmup": 50}' \
#       --reward-content-type jaccard \
#       --use-wandb \
#       --use-grad-checkpointing \
#       --grad-checkpoint-layers 8 \
#       --use-speculative-decoding \
#       --draft-model-path mlx-community/Llama-3-8B-Instruct-4bit \
#       --num-draft-tokens 4 \
#       --kv-bits 4 \
#       --kv-group-size 64 \
#       --quantized-kv-start 1000

#     # Example: Resume training from the latest checkpoint in the output directory
