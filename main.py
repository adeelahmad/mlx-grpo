# mlx_grpo_trainer_aligned.py
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
import threading
from typing import (
    Tuple,
    Dict,
    Any,
    List,
    Optional,
    Union,
    Generator,
    Callable,
    Type,
    get_origin,
    get_args,
    Tuple,
)
from dataclasses import dataclass, field, asdict, fields, is_dataclass, MISSING
from pathlib import Path
import string  # Needed for Jaccard preprocessing
import shutil  # <-- Import shutil for rmtree
from mlx.nn.utils import average_gradients  # Keep for potential future distributed use

# Import MLX components
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from llama_rl.reward import (
    SimpleRewardFunction,
    RewardConfig as SimpleRewardConfig,
)  # Assuming llama_rl package structure
import os
import re
import time
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Union, Callable  # Added Callable
from dataclasses import asdict  # Assuming TrainingArgs is a dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from llama_rl.utils import (
    _save_and_commit,
    _save_directory_and_commit,  # (Needs definition if not in utils)
    _check_disk_space,
    MIN_REQUIRED_BYTES,
    limit_memory,
    # save_config # (Or a similar function for saving model config JSON)
)


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
from rich import print as rprint

# Numpy
import numpy as np

# --- Optional Dependencies ---
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Minimal logging if psutil is missing (logger not fully configured yet)
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
    from datasets import load_dataset, Dataset

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
    from mlx_lm.utils import (
        load_config,
        get_model_path,
        load,
        _get_classes,
        save_config,
        make_shards,
    )
    from mlx_lm.tuner.utils import (
        linear_to_lora_layers,
        load_adapters,
        print_trainable_parameters,
    )
    from mlx_lm.tuner.trainer import grad_checkpoint  # Keep if using grad checkpointing
    from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
    from mlx_lm.sample_utils import (
        make_sampler,
        make_logits_processors,
    )  # Keep if using logits processors
except ImportError as e:
    print(
        f"[bold red]Import error (mlx_lm, tuner utils, or sample_utils):[/] {e}. Please install/update mlx-lm: [code]pip install -U mlx-lm[/]"
    )
    traceback.print_exc()
    sys.exit(1)

# --- Global Variables ---
logger = logging.getLogger(__name__)  # Configured in main()
console = Console(stderr=True, force_terminal=True)
shutdown_requested = False
SAVE_ON_EXIT_FLAG_PATH = Path(".save_on_exit_request")
mlx_rng_key = mx.random.key(0)  # Default key, potentially updated by seeding/checkpoint

# Define EXRERNAL_REWARD_FN if it's used globally. Ensure llama_rl is installed.
try:
    from llama_rl.reward import SimpleRewardFunction, RewardConfig as SimpleRewardConfig

    EXRERNAL_REWARD_FN = SimpleRewardFunction(
        SimpleRewardConfig()
    )  # Assuming default config is fine
except ImportError:
    logger.warning(
        "llama_rl.reward not found. External reward function will not be available."
    )
    EXRERNAL_REWARD_FN = None


# --- Reward Logic ---
@dataclass
class RewardConfig:
    """Configuration for reward calculation tags."""

    think_start_tag: str = "<thinking>"  # Use args tags
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
            # Both thinking and answer have non-empty content after stripping whitespace
            # logger.debug("Format Reward: Correct structure and non-empty content found.")
            return 0.5  # Reward for correct structure and non-empty content
        else:
            # Structure is correct, but one or both content parts are empty
            logger.debug("Format Reward: Correct structure but empty content found.")
            return 0.0  # Neutral reward if tags okay but content missing
    else:
        # The required pattern <thinking>...</thinking>...<answer>...</answer> was not found
        logger.debug("Format Reward: Required tag structure not found.")
        return -0.5  # Penalty for missing structure or wrong order


def math_eval_reward(
    text: str, reference_answer_str: Optional[str], config: RewardConfig
) -> float:
    """
    Assigns reward based on evaluating the expression in answer tags
    and comparing to a numeric reference answer. Gives 1.0 for correct math eval.
    *** WARNING: Uses eval(), posing a SECURITY RISK if inputs are not trusted. ***
    """
    if reference_answer_str is None:
        logger.debug("Math Eval Reward: No reference answer provided.")
        return 0.0  # Cannot evaluate if no reference

    # Escape tags for regex, strip them first
    start_tag_esc = re.escape(config.answer_start_tag.strip())
    end_tag_esc = re.escape(config.answer_end_tag.strip())

    # Regex to find content between tags, allowing for whitespace
    pattern = rf"{start_tag_esc}\s*(.*?)\s*{end_tag_esc}"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        # If answer tags are missing, accuracy reward should be 0, not interfere with format penalty
        logger.debug("Math Eval Reward: <answer> tags not found or pattern mismatch.")
        # logger.debug(f"Text: {text[:100]}...\nReference: {reference_answer_str[:100]}...") # Optional: Log snippet
        return 0.0  # No answer tags found

    extracted_expr = match.group(1).strip()
    if not extracted_expr:
        logger.debug("Math Eval Reward: Extracted expression content is empty.")
        return 0.0  # Empty expression -> incorrect answer

    try:
        # Clean the reference answer to get a numeric value
        cleaned_reference = reference_answer_str
        # Common pattern in math datasets like GSM8K
        if "####" in cleaned_reference:
            try:
                cleaned_reference = cleaned_reference.split("####", 1)[1].strip()
            except IndexError:
                logger.debug(
                    f"Math Eval Reward: '####' found but no content after it in reference: '{reference_answer_str[:50]}...'"
                )
                return 0.0
        # Remove non-numeric, non-decimal point, non-sign chars from the potential number part
        # Allows for scientific notation (e.g., 1e-3)
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
        # Replace common math power syntax ^ with **
        safe_expr = extracted_expr.replace("^", "**")
        # Remove any potentially malicious patterns (basic check) - be very careful here
        if re.search(r'[;\'"`!@#$%^&*<>/\\|~]', safe_expr):
            logger.warning(
                f"Math Eval Reward: Detected potentially unsafe characters in expression '{safe_expr}'. Skipping evaluation."
            )
            return 0.0  # Treat as incorrect if potentially unsafe

        # Limited eval scope - explicitly list allowed functions/objects
        # Import specific math functions you want to allow
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

        # Allow basic numeric types and comparison operators
        allowed_globals = {
            "__builtins__": {"abs": abs, "round": round, "min": min, "max": max},
            "math": allowed_math,
        }
        allowed_locals = {}  # No extra locals

        evaluated_answer = eval(safe_expr, allowed_globals, allowed_locals)

        if not isinstance(evaluated_answer, (int, float)):
            logger.debug(
                f"Math Eval Reward: Evaluated expression '{extracted_expr}' did not result in a number ({type(evaluated_answer).__name__})."
            )
            return 0.0

        # Use math.isclose for float comparison (gives 1.0 if correct, 0.0 otherwise)
        # Adjust tolerances based on expected precision of answers
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
        return 0.0  # Treat evaluation errors as incorrect


def jaccard_reward(
    text: str, reference_completion: Optional[str], config: RewardConfig
) -> float:
    """
    Assigns reward based on the Jaccard Similarity between the token sets
    of the text within <answer> tags and the reference completion string. Gives score 0-1.
    """
    if reference_completion is None:
        logger.debug("Jaccard Reward: No reference completion provided.")
        return 0.0  # Cannot evaluate if no reference

    # Escape tags for regex, strip them first
    start_tag_esc = re.escape(config.answer_start_tag.strip())
    end_tag_esc = re.escape(config.answer_end_tag.strip())

    # Regex to find content between tags, allowing for whitespace
    pattern = rf"{start_tag_esc}\s*(.*?)\s*{end_tag_esc}"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        # If answer tags are missing, accuracy reward should be 0
        logger.debug("Jaccard Reward: <answer> tags not found or pattern mismatch.")
        # logger.debug(f"Text: {text[:100]}...\n=====\nCompletion: {reference_completion[:100]}...") # Optional: Log snippet
        return 0.0  # No answer tags found

    extracted_answer = match.group(1).strip()

    # --- Jaccard Similarity Calculation ---
    def preprocess_text(input_text: str) -> set:
        if not isinstance(input_text, str):
            logger.debug(
                f"Jaccard preprocess_text: Input is not a string ({type(input_text)}). Returning empty set."
            )
            return set()
        # Use regex for robustness against various whitespace and multiple punctuation marks
        text_lower = input_text.lower()
        # Replace punctuation with spaces and split
        text_no_punct = re.sub(
            r"[%s]+" % re.escape(string.punctuation), " ", text_lower
        )
        tokens = set(text_no_punct.split())
        # Optional: Add stop word removal if desired
        return tokens

    generated_tokens = preprocess_text(extracted_answer)
    reference_tokens = preprocess_text(reference_completion)

    if not generated_tokens and not reference_tokens:
        return 1.0  # Both empty after preprocessing -> perfect match
    if not generated_tokens or not reference_tokens:
        return 0.0  # One is empty, the other is not

    intersection = generated_tokens.intersection(reference_tokens)
    union = generated_tokens.union(reference_tokens)

    if not union:  # Should only happen if both are empty, handled above
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
    # Add other reward functions here
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
    reward_format_weight: float,  # Weight for format (e.g., 0.5)
    reward_content_weight: float,  # Weight for content (e.g., 0.5)
    reward_content_type: str,
) -> Tuple[float, float, float]:
    """
    Calculates the total weighted reward based on format (0.5/-0.5) and content (0-1) components.
    Total reward = w_fmt * fmt_rew + w_cont * cont_rew.
    Returns (total_weighted_reward, raw_format_reward, raw_content_reward).
    """
    # Check weights sum to 1 or adjust logging/interpretation
    if not math.isclose(reward_format_weight + reward_content_weight, 1.0):
        logger.warning(
            f"Reward weights ({reward_format_weight} + {reward_content_weight}) do not sum to 1.0. Total reward is a weighted sum, not a simple average."
        )

    # Raw format reward (e.g., 0.5, 0.0, -0.5)
    format_rew = format_reward(generated_text, reward_config)
    logger.debug(
        f"Format Reward: {format_rew:.2f} (Text: {generated_text[:100]}...)"
    )  # Log snippet of text

    # Raw content reward (e.0.0 to 1.0 for jaccard/math_eval)
    content_rew = 0.0
    external_rew = 0.0
    external_rew_weight = 0.0  # Default external reward weight

    try:
        content_reward_fn = get_content_reward_fn(reward_content_type)
        content_rew = content_reward_fn(
            generated_text, reference_answer_str, reward_config
        )
        logger.debug(f"Content Reward ({reward_content_type}): {content_rew:.4f}")

        # Apply external reward if available
        if EXRERNAL_REWARD_FN is not None:
            try:
                # Assuming calculate_reward returns a tuple (reward, weight)
                external_rew, external_rew_weight = EXRERNAL_REWARD_FN.calculate_reward(
                    generated_text, reference_answer_str
                )
                logger.debug(
                    f"External Reward: {external_rew:.4f} (Weight: {external_rew_weight:.2f})"
                )
                # Combine internal and external content rewards
                # This logic was potentially problematic before. A common way is:
                # final_content_rew = content_rew * (1 - external_rew_weight) + external_rew * external_rew_weight
                # OR if external_rew is a multiplier/bonus:
                # final_content_rew = content_rew * (1 + external_rew * external_rew_weight)
                # Let's stick to the previous logic for now, assuming external_rew is a bonus added to content_rew
                content_rew = content_rew + (
                    external_rew * external_rew_weight
                )  # Assuming external_rew_weight is used as a multiplier here
                logger.debug(
                    f"Combined Content Reward (Internal + External): {content_rew:.4f}"
                )

            except Exception as e_external:
                logger.error(
                    f"Error calculating external reward: {e_external}", exc_info=False
                )
                # Continue with just internal content_rew
        # else: logger.debug("External reward function not available.")

    except ValueError:
        logger.error(
            f"Invalid reward_content_type '{reward_content_type}' during reward calculation. Setting content reward to 0.",
            exc_info=True,
        )
        content_rew = 0.0
    except Exception as e:
        logger.error(
            f"Error calculating content reward ({reward_content_type}): {e}",
            exc_info=True,
        )
        content_rew = 0.0  # Treat calculation errors as zero reward

    # Calculate total weighted reward
    # Example: If perfect format (0.5) and perfect content (1.0) with 0.5 weights each:
    # total_rew = (0.5 * 0.5) + (0.5 * 1.0) = 0.25 + 0.5 = 0.75
    # Example: If bad format (-0.5) and okay content (0.6):
    # total_rew = (0.5 * -0.5) + (0.5 * 0.6) = -0.25 + 0.3 = 0.05
    # The previous code had `content_rew = content_rew + (EXRERNAL_REWARD_FN.calculate_reward(generated_text,reference_answer_str)[0] * content_rew)`
    # This means `content_rew` was being *multiplied* by the external reward value.
    # The current code adds `external_rew * external_rew_weight` which seems more standard.
    # Reverting to the previous logic if that was intended:
    # if EXRERNAL_REWARD_FN is not None:
    #      try:
    #          external_rew_val, _ = EXRERNAL_REWARD_FN.calculate_reward(generated_text,reference_answer_str)
    #          content_rew = content_rew * (1 + external_rew_val) # Example: content_rew * (1 + bonus)
    #          logger.debug(f"Combined Content Reward (Internal * (1+External)): {content_rew:.4f}")
    #      except Exception as e_external:
    #          logger.error(f"Error calculating external reward for multiplication: {e_external}", exc_info=False)
    #          # Continue with just internal content_rew

    # Let's clarify the external reward interaction:
    # If EXRERNAL_REWARD_FN.calculate_reward returns a tuple (bonus_value, weight),
    # a simple approach is: final_content_rew = content_rew + (bonus_value * weight)
    # Or if weight is a factor: final_content_rew = (content_rew * (1-weight)) + (bonus_value * weight)
    # Sticking to the simple additive bonus scaled by its weight seems most reasonable for now.
    # The logger.info statements below seem to reflect the simple weighted sum structure, not the external bonus calculation.
    # Let's add logging reflecting the combined content reward before final calculation.

    final_content_reward_for_sum = content_rew  # Use the potentially combined value

    total_rew = (reward_format_weight * format_rew) + (
        reward_content_weight * final_content_reward_for_sum
    )

    # INFO logging (these seem redundant with DEBUG logs above, maybe keep only for final calculation?)
    # logger.info(f"Content Rewarded: {final_content_reward_for_sum}") # This is the combined internal+external
    # logger.info(f"Final Content Rewarded With External: {final_content_reward_for_sum}") # Same as above
    # logger.info(f"Final Reward: {total_rew}") # Final weighted sum

    return (
        float(total_rew),
        float(format_rew),
        float(final_content_reward_for_sum),
    )  # Return raw values for logging


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
        logging.shutdown()  # Attempt to flush logs
        sys.exit(1)  # Force exit


# --- Configuration Dataclass ---
@dataclass
class TrainingArgs:
    """Configuration arguments for the GRPO training script."""

    # --- Paths & Setup ---
    output_dir: str = field(
        metadata={"help": "MANDATORY: Directory for checkpoints, logs, final model."}
    )
    model_path: str = field(
        metadata={"help": "MANDATORY: Path or ID of the base MLX model to train."}
    )  # Made mandatory
    ref_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path or ID of the reference model for KL penalty (defaults to model_path if None)."
        },
    )
    train_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local training JSONL path (overrides HF dataset)."},
    )  # Changed default to None
    val_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local validation JSONL path (overrides HF dataset)."},
    )  # Changed default to None
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
    )  # Changed default to None
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
    )  # Adjusted default for gsm8k
    dataset_answer_key: str = field(
        default="completion",
        metadata={"help": "Dataset dictionary key for the reference answer."},
    )  # Adjusted default for gsm8k
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to load pre-trained LoRA adapters from."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a checkpoint directory to resume training from (contains weights, optimizer, etc.). If not set, tries to find 'latest' symlink in output_dir."
        },
    )  # Updated help text

    # --- Model & Tokenizer ---
    max_prompt_len: int = field(
        default=512,
        metadata={
            "help": "Maximum number of tokens for the input prompt (truncates longer prompts)."
        },
    )
    max_gen_len: int = field(
        default=512,
        metadata={"help": "Maximum number of tokens to generate during rollout."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": "System prompt to use in chat format (uses default if None)."
        },
    )
    # Special Tokens (added for Task 3) - ensure defaults match RewardConfig
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
    # Add mean embedding initialization flag
    init_new_embeddings_with_mean: bool = field(
        default=True,
        metadata={
            "help": "Initialize embeddings for new special tokens with the mean of existing embeddings."
        },
    )

    # --- Optimizer & Scheduling ---
    learning_rate: float = field(
        default=1e-5, metadata={"help": "Peak learning rate for the AdamW optimizer."}
    )
    lr_decay_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Steps for one LR decay cycle (MLX requires manual implementation)."
        },
    )
    lr_decay_rate: float = field(default=0.1, metadata={"help": "LR decay factor."})
    lr_decay_min_lr: float = field(
        default=1e-7, metadata={"help": "Minimum LR after decay."}
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
    )  # Default 8 for GRPO
    ppo_batch_size: int = field(
        default=2,
        metadata={
            "help": "Number of prompts processed in one rollout/gradient calculation step (micro-batch size)."
        },
    )  # Increased default
    sampling_temperature: float = field(
        default=0.7,
        metadata={
            "help": "Temperature for sampling during rollouts (higher is more random)."
        },
    )
    sampling_top_p: float = field(
        default=0.9,
        metadata={
            "help": "Top-p (nucleus) sampling probability during rollouts (0 to disable)."
        },
    )
    grpo_beta: float = field(
        default=0.1,
        metadata={"help": "Beta hyperparameter for GRPO KL penalty strength."},
    )
    advantage_epsilon: float = field(
        default=1e-8,
        metadata={
            "help": "Epsilon added to advantage normalization denominator for stability."
        },
    )

    # --- LoRA Configuration ---
    use_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA fine-tuning instead of full parameter tuning."},
    )
    lora_layers: int = field(
        default=16,
        metadata={"help": "Number of layers to apply LoRA to (from the top)."},
    )
    lora_rank: int = field(default=16, metadata={"help": "Rank of the LoRA matrices."})
    lora_alpha: float = field(
        default=32.0, metadata={"help": "LoRA alpha scaling parameter (often 2*rank)."}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "Dropout probability for LoRA layers."}
    )
    lora_scale: float = field(
        default=1.0, metadata={"help": "LoRA scaling factor (applied to the output)."}
    )

    # --- Training Control & Techniques ---
    num_training_steps: int = field(
        default=5000,
        metadata={"help": "Total number of training update steps (optimizer steps)."},
    )
    save_every: int = field(
        default=5, metadata={"help": "Save checkpoint every N update steps."}
    )  # Increased default
    eval_every: int = field(
        default=10, metadata={"help": "Evaluate every N update steps."}
    )  # Increased default
    # Removed generate_samples_every, covered by eval logging
    seed: int = field(default=42, metadata={"help": "Random seed."})
    shuffle_data: bool = field(
        default=True,
        metadata={"help": "Shuffle training data indices each epoch/pass."},
    )
    grad_accum_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation steps."}
    )  # Increased default
    use_grad_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "Enable gradient checkpointing (requires manual layer wrapping)."
        },
    )
    grad_checkpoint_layers: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of layers for grad checkpointing (default: all applicable if enabled)."
        },
    )

    # --- Reward Configuration ---
    reward_format_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for the format/structure reward component."},
    )  # Adjusted default to 0.5
    reward_content_weight: float = field(
        default=0.7, metadata={"help": "Weight for the content reward component."}
    )  # Adjusted default to 0.5
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
        default=False, metadata={"help": "Log metrics to Weights & Biases."}
    )  # Default False
    wandb_project: Optional[str] = field(
        default="mlx-grpo-finetune", metadata={"help": "WandB project name."}
    )  # Adjusted default
    wandb_entity: Optional[str] = field(
        default=None, metadata={"help": "WandB entity (user or team name)."}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "Custom name for the WandB run."}
    )

    # --- Calculated Field ---
    effective_batch_size: int = field(init=False)  # Calculated in post_init

    def __post_init__(self):
        """Perform validation checks and calculate derived fields."""
        if not self.output_dir:
            raise ValueError("--output-dir is mandatory.")
        if not self.model_path:
            raise ValueError("--model-path is mandatory.")
        if self.use_lora and self.lora_rank <= 0:
            raise ValueError("If use_lora is True, lora_rank must be > 0.")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1.")
        if self.ppo_batch_size < 1:
            raise ValueError("ppo_batch_size must be >= 1.")
        # if self.num_rollout_samples < 1: raise ValueError("num_rollout_samples must be >= 1.") # Allow 1 for REINFORCE, but GRPO is >1
        if self.num_rollout_samples > 1 and not math.isclose(
            self.reward_format_weight + self.reward_content_weight, 1.0
        ):
            logger.warning(
                f"Reward weights ({self.reward_format_weight} + {self.reward_content_weight}) do not sum to 1.0. Total reward is a weighted sum, not a simple average."
            )
        if self.adapter_path and self.resume_from_checkpoint:
            logger.warning(
                "Both --adapter-path and --resume-from-checkpoint provided. Checkpoint takes precedence for loading weights."
            )
        if self.adapter_path and not self.use_lora:
            logger.warning(
                "--adapter-path provided but --no-use-lora set. Adapter will not be loaded."
            )

        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            print("INFO: grad_clip_norm <= 0, disabling gradient clipping.")
            self.grad_clip_norm = None

        if self.ref_model_path is None:
            print(
                f"INFO: ref_model_path not set, defaulting to model_path: '{self.model_path}'"
            )
            self.ref_model_path = self.model_path

        if self.lr_decay_steps:
            print(
                "WARNING: LR decay args set, but MLX optimizers lack built-in schedulers. Manual implementation needed if required."
            )

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
            # Default system prompt using the *actual* special tokens defined in args
            # Example prompt template from Llama-3-Instruct
            # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            #
            # {your_system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
            #
            # {your_user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            #
            # Example tailored for thinking/answer tags:
            # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            #
            # You are a helpful assistant. First, think step-by-step using {think_start}...{think_end}. Then, provide the answer using {answer_start}...{answer_end}.
            # <|eot_id|><|start_header_id|>user<|end_header_id|>
            #
            # {your_user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            #
            # {thinking_and_answer_generation_starts_here}
            #
            # Note: The specific chat template structure might vary by model.
            # Using a generic format here, assuming apply_chat_template handles model specifics.
            # The default system prompt should ideally guide the model to use the tags.

            default_sys = f"""You are a helpful assistant. Respond with your step-by-step thinking process inside `{self.think_start_tag}` and `{self.think_end_tag}` tags. Then, provide the final answer content inside `{self.answer_start_tag}` and `{self.answer_end_tag}` tags."""

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
    ):  # Only add if system prompt is provided and not empty
        messages.append(
            {"role": "system", "content": system_prompt.strip()}
        )  # Strip system prompt whitespace
    messages.append(
        {"role": "user", "content": prompt.strip()}
    )  # Strip user prompt whitespace

    try:
        # Add add_generation_prompt=True to ensure the template ends correctly
        # for the model to start generating (e.g., includes "<|assistant|>\n")
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # Return string, not tokens
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
        # Ensure fallback ends appropriately for generation
        return f"{prefix}User: {prompt.strip()}\n\nAssistant:"


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


def _create_4d_attention_masdk(
    tokens: mx.array, pad_token_id: int, dtype: mx.Dtype = mx.bfloat16
) -> mx.array:
    """Creates a 4D attention mask combining causal and padding masks."""
    if tokens.ndim != 2:
        raise ValueError(
            f"Input tokens must be 2D (batch_size, sequence_length), got shape {tokens.shape}"
        )
    batch_size, sequence_length = tokens.shape
    if sequence_length == 0:
        # Handle empty sequence case
        return mx.zeros(
            (batch_size, 1, 0, 0), dtype=dtype
        )  # Or raise error if empty batch invalid

    # 1. Causal Mask (Use MLX utility)
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(
        sequence_length, dtype=dtype
    )
    # Shape: (sequence_length, sequence_length)
    # 2. Padding Mask (True where token is pad_token_id)
    padding_mask_2d = tokens == pad_token_id
    # Shape: (batch_size, sequence_length)
    # 3. Expand masks for broadcasting
    # Causal mask expands to (1, 1, L, L) - applies to all batches/heads
    causal_mask_4d = causal_mask[None, None, :, :]
    # Padding mask expands to (B, 1, 1, L) - applies per batch/head along the key/value sequence dim
    padding_mask_4d = padding_mask_2d[:, None, None, :]
    # 4. Combine masks
    # Broadcast padding mask to (B, 1, L, L)
    # Where padding_mask_4d is True (meaning the *key* token is padding), set mask value to -inf
    # Where padding_mask_4d is False, use the causal_mask value.
    broadcastable_padding_mask = mx.broadcast_to(
        padding_mask_4d, (batch_size, 1, sequence_length, sequence_length)
    )
    # Where the key token is padding, mask is -inf. Otherwise, use the causal mask.
    # Note: MLX attention expects 0 for attended positions, negative infinity for masked positions.
    # The causal mask already has 0s and -infs. The padding mask needs to introduce -infs.
    neg_inf_val = (
        -1e9 if dtype == mx.float32 else -65504.0
    )  # Use appropriate neg inf for dtype (bfloat16 max is ~3.38e38, so need negative infinity)
    neg_inf_array = mx.full((), neg_inf_val, dtype=dtype)  # Scalar for broadcasting
    combined_mask = mx.where(
        broadcastable_padding_mask,
        neg_inf_array,  # Mask out keys that are padding (along the Key/Value sequence axis)
        causal_mask_4d,  # Otherwise, use the causal mask (which already handles future positions)
    )
    # Shape: (batch_size, 1, sequence_length, sequence_length)
    return combined_mask


def _create_4d_attention_mask(
    tokens: mx.array, pad_token_id: int, dtype: mx.Dtype = mx.bfloat16
) -> mx.array:
    """Creates a 4D attention mask combining causal and padding masks."""
    if tokens.ndim != 2:
        raise ValueError(
            f"Input tokens must be 2D (batch_size, sequence_length), got shape {tokens.shape}"
        )
    batch_size, sequence_length = tokens.shape
    # 1. Causal Mask (Use MLX utility)
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(
        sequence_length, dtype=dtype
    )
    # Shape: (sequence_length, sequence_length)
    # 2. Padding Mask (True where token is pad_token_id)
    padding_mask_2d = tokens == pad_token_id
    # Shape: (batch_size, sequence_length)
    # 3. Expand masks for broadcasting
    # Causal mask expands to (1, 1, L, L) - applies to all batches/heads
    causal_mask_4d = causal_mask[None, None, :, :]
    # Padding mask expands to (B, 1, 1, L) - applies per batch/head along the key/value sequence dim
    padding_mask_4d = padding_mask_2d[:, None, None, :]
    # 4. Combine masks
    # Broadcast padding mask to (B, 1, L, L)
    # Where padding_mask_4d is True (meaning the *key* token is padding), set mask value to -inf
    # Where padding_mask_4d is False, use the causal_mask value.
    broadcastable_padding_mask = mx.broadcast_to(
        padding_mask_4d, (batch_size, 1, sequence_length, sequence_length)
    )
    # Where the key token is padding, mask is -inf. Otherwise, use the causal mask.
    # Note: MLX attention expects 0 for attended positions, negative infinity for masked positions.
    # The causal mask already has 0s and -infs. The padding mask needs to introduce -infs.
    neg_inf_val = (
        -1e9 if dtype == mx.float32 else -65504.0
    )  # Use appropriate neg inf for dtype
    neg_inf_array = mx.full((), neg_inf_val, dtype=dtype)  # Scalar for broadcasting
    combined_mask = mx.where(
        broadcastable_padding_mask,
        neg_inf_array,  # Mask out keys that are padding
        causal_mask_4d,  # Otherwise, use the causal mask (which already handles future positions)
    )
    # Shape: (batch_size, 1, sequence_length, sequence_length)
    return combined_mask


# --- REVISED build_rollout_batch ---
def build_rollout_batch(
    tokenizer: TokenizerWrapper,
    dataset: Dataset,
    indices: List[int],
    num_samples_per_prompt: int,
    max_prompt_len: int,
    system_prompt: Optional[str],
    prompt_key: str,
    answer_key: str,
) -> Tuple[List[Dict], mx.array, int]:
    """
    Prepares a batch of tokenized prompts using chat template and corresponding answers.
    Adds basic validation for input text fields.
    Returns prompts_data, prompts_mx (padded tokens), max_prompt_len_batch.
    Reference answers are included in prompts_data.
    """
    prompts_data = (
        []
    )  # List of dicts {'text': str, 'tokens': List[int], 'ref_answer_str': str or None}
    max_len_in_batch = 0
    logger = logging.getLogger(__name__)

    for i in indices:
        try:
            sample_data = dataset[i]
            question = sample_data.get(prompt_key)
            answer_raw = sample_data.get(answer_key)

            # --- Basic Validation ---
            if not isinstance(question, str) or not question.strip():
                logger.warning(
                    f"Skipping dataset index {i}: Invalid or empty question field ('{prompt_key}')."
                )
                continue

            # Process the reference answer (keep as string or None)
            final_answer_str = None
            if isinstance(answer_raw, (str, int, float)):
                final_answer_str = str(answer_raw)
                if not final_answer_str.strip():
                    logger.debug(
                        f"Dataset index {i}: Empty reference answer field ('{answer_key}'). Reward calculation for content may be affected."
                    )
                    final_answer_str = None  # Treat empty string as None

            if final_answer_str is None and answer_raw is not None:
                logger.debug(
                    f"Dataset index {i}: Reference answer is not string/numeric: {type(answer_raw)}. Setting reference to None."
                )

            # Create the formatted prompt text using the tokenizer's chat template
            # This now includes model-specific formatting and the generation prompt.
            prompt_text = apply_chat_template_wrapper(
                tokenizer, question, system_prompt
            )

            # Tokenize the formatted prompt
            try:
                # Do NOT add special tokens again, apply_chat_template handles them.
                # Added return_attention_mask=False as it's not needed here, mask created later
                encoded = tokenizer.encode_plus(
                    prompt_text,
                    add_special_tokens=False,
                    return_token_type_ids=False,
                    return_attention_mask=False,
                )
                prompt_tokens = encoded["input_ids"]

            except Exception as e_tok:
                logger.warning(
                    f"Skipping dataset index {i}: Tokenization failed for formatted prompt '{prompt_text[:50]}...'. Error: {e_tok}"
                )
                continue

            # Truncate if necessary (from the left)
            if len(prompt_tokens) > max_prompt_len:
                prompt_tokens = prompt_tokens[-max_prompt_len:]
                # logger.debug(f"Truncated prompt for index {i} to {max_prompt_len} tokens (keeping end).")

            if not prompt_tokens:
                logger.warning(
                    f"Skipping dataset index {i}: Tokenization resulted in empty token list for formatted prompt '{prompt_text[:50]}...'."
                )
                continue

            # Store prompt data
            prompts_data.append(
                {
                    "original_index": i,  # Keep track of original dataset index
                    "text": prompt_text,  # Store the formatted text (useful for logging/debugging)
                    "tokens": prompt_tokens,
                    "ref_answer_str": final_answer_str,  # Store the cleaned reference string
                }
            )
            max_len_in_batch = max(max_len_in_batch, len(prompt_tokens))

        except KeyError as e:
            logger.warning(
                f"Skipping dataset index {i}: Missing key '{e}' in dataset sample."
            )
        except Exception as e:
            logger.warning(
                f"Skipping dataset index {i} due to unexpected error during processing: {e}",
                exc_info=False,
            )

    if not prompts_data:
        logger.warning("No valid prompts found in the batch.")
        # Return empty lists/arrays consistent with the signature
        return [], mx.array([], dtype=mx.int32), 0

    # --- Pad the prompts ---
    padded_prompts_list = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        error_msg = "Tokenizer must have a pad_token_id for padding."
        logger.critical(error_msg)
        raise ValueError(error_msg)

    for p_data in prompts_data:
        p_tokens = p_data["tokens"]
        pad_len = max_len_in_batch - len(p_tokens)
        padded_tokens = ([pad_id] * pad_len) + p_tokens  # Pad on the left
        padded_prompts_list.append(padded_tokens)

    # Convert to MLX array
    try:
        prompts_mx = mx.array(padded_prompts_list, dtype=mx.int32)
    except Exception as e_pad:
        logger.error(
            f"Failed to create MLX array from padded prompts: {e_pad}", exc_info=True
        )
        # Return empty lists/arrays on error
        return [], mx.array([], dtype=mx.int32), 0

    # Return prompts_data (includes ref_answer_str), the padded token array, and the max length.
    return prompts_data, prompts_mx, max_len_in_batch


def generate_rollouts_for_batch(
    model: nn.Module,
    ref_model: nn.Module,
    tokenizer: TokenizerWrapper,
    prompts_data: List[
        Dict
    ],  # <<< Added: Use prompts_data returned by build_rollout_batch
    prompts_mx: mx.array,  # Shape: (num_prompts, max_prompt_len_batch)
    max_prompt_len_batch: int,  # <<< Keep for clarity
    num_samples_per_prompt: int,
    max_gen_len: int,
    args: TrainingArgs,  # Pass full args to access sampling, reward config, etc.
) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:
    """
    Generates rollouts, calculates rewards, advantages, and reference log probs.
    Uses explicit indexing for correct reward alignment.
    """
    logger = logging.getLogger(__name__)
    num_prompts = prompts_mx.shape[0]
    if num_prompts == 0:
        logger.warning(
            "Received empty prompts_mx batch in generate_rollouts_for_batch."
        )
        # Return empty batch data and zero rewards
        return (
            {
                "tokens": mx.array([], dtype=mx.int32),
                "response_mask": mx.array([], dtype=mx.float32),
                "advantages": mx.array([], dtype=mx.float32),
                "ref_log_probs": mx.array([], dtype=mx.float32),
            },
            0.0,
            {"raw_format": 0.0, "raw_content": 0.0},
        )

    total_samples = num_prompts * num_samples_per_prompt
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must have a pad_token_id.")

    eos_token_str = (
        tokenizer.decode(
            [eos_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if eos_id is not None
        else ""
    )
    pad_token_str = (
        tokenizer.decode(
            [pad_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if pad_id is not None
        else ""
    )

    # Create RewardConfig instance from args
    reward_config = RewardConfig(
        think_start_tag=args.think_start_tag,
        think_end_tag=args.think_end_tag,
        answer_start_tag=args.answer_start_tag,
        answer_end_tag=args.answer_end_tag,
    )

    # --- Determine Model Dtypes ---
    model_dtype = mx.bfloat16  # Default
    try:
        model_dtype = next(tree_flatten(model.parameters()))[1].dtype
    except Exception:
        logger.debug(
            f"Generate: Could not detect actor model dtype, using {model_dtype}."
        )

    ref_model_dtype = mx.bfloat16  # Default
    try:
        ref_model_dtype = next(tree_flatten(ref_model.parameters()))[1].dtype
    except Exception:
        logger.debug(
            f"Generate: Could not detect ref model dtype, using {ref_model_dtype}."
        )

    # --- Process Prompts and Initialize Cache ---
    # Repeat prompts for each sample
    prompts_mx_repeated = mx.repeat(prompts_mx, repeats=num_samples_per_prompt, axis=0)
    prompt_attn_mask_4d = _create_4d_attention_mask(
        prompts_mx_repeated, pad_id, dtype=model_dtype
    )
    cache = None
    try:
        # Initial pass for prompt processing
        logger.debug(
            f"Calling actor model for prompt processing. Input shape: {prompts_mx_repeated.shape}"
        )
        model.eval()  # Ensure actor is in eval mode for generation (no dropout etc.)
        model_output = model(prompts_mx_repeated, mask=prompt_attn_mask_4d, cache=cache)
        model.train()  # Switch back to train mode immediately after first forward pass
        if isinstance(model_output, tuple):
            logits_prompt, cache = model_output[:2]
        else:  # Assume only logits returned
            logits_prompt = model_output
            cache = None
        next_token_logits = logits_prompt[:, -1, :]  # Logits for next token
        mx.eval(next_token_logits, cache)
    except Exception as e_prompt:
        logger.error(
            f"Failed during initial prompt processing pass: {e_prompt}", exc_info=True
        )
        # Return empty results if prompt processing fails
        return (
            {
                "tokens": mx.array([], dtype=mx.int32),
                "response_mask": mx.array([], dtype=mx.float32),
                "advantages": mx.array([], dtype=mx.float32),
                "ref_log_probs": mx.array([], dtype=mx.float32),
            },
            0.0,
            {"raw_format": 0.0, "raw_content": 0.0},
        )

    # --- Generation Loop ---
    response_tokens_list = []
    sampler = make_sampler(temp=args.sampling_temperature, top_p=args.sampling_top_p)

    try:  # Sample first token
        current_tokens = sampler(next_token_logits)
    except Exception as e_sample:
        logger.error(f"Failed during initial token sampling: {e_sample}", exc_info=True)
        # Pad the rest if sampling fails
        current_tokens = mx.full((total_samples,), pad_id, dtype=mx.int32)
        # Ensure 'ended' is True for all samples
        ended = mx.ones_like(current_tokens)
        response_tokens_list.append(current_tokens[:, None])  # Append the failure token

    ended = mx.zeros_like(current_tokens)
    if eos_id is not None:
        ended = mx.equal(current_tokens, eos_id)

    # Need to handle the first token *before* the loop if cache is updated
    # current_tokens holds the first sampled tokens for the batch
    if (
        not response_tokens_list
    ):  # Only append if not already padded due to sampling error
        response_tokens_list.append(
            current_tokens[:, None]
        )  # Append the first token column

    gen_start_time = time.monotonic()
    # Start the generation loop from step 1 (since step 0 was the first token)
    # The loop runs max_gen_len - 1 times to generate the remaining tokens
    for step in range(max_gen_len - 1):
        if mx.all(ended).item():
            break

        try:
            # Pass the previously sampled token (current_tokens) to the model
            model_step_output = model(current_tokens[:, None], mask=None, cache=cache)
            if isinstance(model_step_output, tuple):
                logits_step, cache = model_step_output[:2]
            else:
                logits_step = model_step_output
                cache = None

            next_token_logits = logits_step[:, -1, :]
            current_tokens = sampler(next_token_logits)

            just_ended = mx.zeros_like(ended)
            if eos_id is not None:
                just_ended = mx.equal(current_tokens, eos_id)

            # Check which sequences *were not* ended before this step
            ended_before_step = ended  # Use the 'ended' state *before* updating it

            # Update 'ended' state: a sequence ends if it just generated EOS *and was not already ended*
            # Or more simply, a sequence is ended if it was ended OR it just generated EOS
            mx.eval(ended_before_step)  # Ensure computation before using in where
            ended = mx.logical_or(ended_before_step, just_ended)

            # For sequences that ended *before* this step, add padding instead of the sampled token
            pad_val = mx.array(pad_id, dtype=current_tokens.dtype)
            tokens_to_add = mx.where(ended_before_step, pad_val, current_tokens)
            mx.eval(tokens_to_add, cache, ended)  # Evaluate all relevant outputs

            response_tokens_list.append(tokens_to_add[:, None])
            current_tokens = (
                tokens_to_add  # Use the (potentially padded) token for next input
            )

        except Exception as e_gen:
            logger.error(f"Generation failed at step {step+1}: {e_gen}", exc_info=True)
            # Pad remaining steps if generation fails
            current_batch_size = (
                response_tokens_list[0].shape[0]
                if response_tokens_list
                else total_samples
            )
            pad_dtype = (
                response_tokens_list[0].dtype if response_tokens_list else mx.int32
            )
            remaining_len = max_gen_len - len(response_tokens_list)
            if remaining_len > 0:
                logger.warning(
                    f"Padding remaining {remaining_len} steps due to generation error."
                )
                # Create a pad array for the remaining steps
                pad_array_remaining = mx.full(
                    (current_batch_size, remaining_len), pad_id, dtype=pad_dtype
                )
                # Append the remaining pad steps
                # Need to append column by column or concatenate?
                # Appending column by column:
                for k in range(remaining_len):
                    response_tokens_list.append(
                        pad_array_remaining[:, k : k + 1]
                    )  # Append (B, 1) slice

            break  # Exit generation loop

    gen_duration = time.monotonic() - gen_start_time
    logger.debug(
        f"Generation loop finished ({len(response_tokens_list)} steps) in {gen_duration:.2f}s."
    )

    # --- Combine and Process Responses ---
    if not response_tokens_list:
        logger.warning("No response tokens were generated.")
        generated_seq_len = 0
        responses = mx.zeros((total_samples, 0), dtype=mx.int32)  # Empty array
        decoded_responses = [""] * total_samples
    else:
        try:
            # response_tokens_list contains tensors of shape (total_samples, 1)
            responses = mx.concatenate(
                response_tokens_list, axis=1
            )  # Shape (total_samples, generated_seq_len)
            generated_seq_len = responses.shape[1]
            mx.eval(responses)
            responses_list = responses.tolist()
            # Decode without skipping special tokens initially to preserve tags
            decoded_responses = tokenizer.batch_decode(
                responses_list,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
        except Exception as e_concat_decode:
            logger.error(
                f"Failed to concatenate or decode response tokens: {e_concat_decode}. Using empty/error responses.",
                exc_info=True,
            )
            generated_seq_len = 0
            responses = mx.zeros((total_samples, 0), dtype=mx.int32)
            decoded_responses = ["[concat/decode error]"] * total_samples

    # --- Calculate Rewards ---
    rewards_list_total, rewards_list_format, rewards_list_content = [], [], []
    # Use explicit indexing to get the correct reference answer for each generated sample
    # The generated samples are ordered: Prompt0_Sample0, Prompt0_Sample1, ..., Prompt1_Sample0, ...
    # So sample i corresponds to prompt index i // num_samples_per_prompt
    num_successful_rewards = 0

    for i in range(total_samples):
        resp_text = decoded_responses[i]
        prompt_index = (
            i // num_samples_per_prompt
        )  # Index in the original prompts_data batch
        ref_ans = prompts_data[prompt_index][
            "ref_answer_str"
        ]  # Get ref answer from original prompt data

        # Clean response: remove padding, truncate at first EOS
        cleaned_resp = resp_text
        if pad_token_str and cleaned_resp.endswith(pad_token_str):
            # Use regex to remove trailing pad tokens and surrounding whitespace robustly
            cleaned_resp = re.sub(
                rf"(?:\s*{re.escape(pad_token_str)})+$", "", cleaned_resp
            )
            cleaned_resp = (
                cleaned_resp.rstrip()
            )  # Clean any remaining trailing whitespace
        eos_pos = cleaned_resp.find(eos_token_str) if eos_token_str else -1
        if eos_pos != -1:
            cleaned_resp = cleaned_resp[
                :eos_pos
            ].strip()  # Strip after truncating at EOS

        if cleaned_resp.startswith("[") and "error]" in cleaned_resp:
            total_reward, fmt_rew, content_rew = (
                0.0,
                0.0,
                0.0,
            )  # Penalize errors heavily implicitly
        else:
            # Use the reward calculation function with configured weights and type
            total_reward, fmt_rew, content_rew = calculate_total_reward(
                cleaned_resp,
                ref_ans,
                reward_config,
                args.reward_format_weight,
                args.reward_content_weight,
                args.reward_content_type,
            )
            num_successful_rewards += 1

        rewards_list_total.append(total_reward)
        rewards_list_format.append(fmt_rew)
        rewards_list_content.append(content_rew)

    rewards = mx.array(
        rewards_list_total, dtype=mx.float32
    )  # Use float32 for rewards/advantages
    rewards_raw_format = mx.array(rewards_list_format, dtype=mx.float32)
    rewards_raw_content = mx.array(rewards_list_content, dtype=mx.float32)
    mx.eval(rewards, rewards_raw_format, rewards_raw_content)

    # --- Calculate Advantages (GRPO style: normalize per prompt group) ---
    if (
        num_prompts > 0 and num_samples_per_prompt > 1
    ):  # Need multiple samples per prompt for baseline
        rewards_per_prompt = rewards.reshape(num_prompts, num_samples_per_prompt)
        baseline = mx.mean(
            rewards_per_prompt, axis=1, keepdims=True
        )  # Baseline = mean reward in group
        variance = mx.var(rewards_per_prompt, axis=1, keepdims=True)
        std_dev = mx.sqrt(
            variance + args.advantage_epsilon
        )  # Add epsilon inside sqrt for stability

        # Normalize advantages: (reward - baseline) / (std_dev + epsilon)
        # Add epsilon *outside* std_dev for normalization robustness as per common implementations
        advantages = (rewards_per_prompt - baseline) / (
            std_dev + args.advantage_epsilon
        )
        advantages = advantages.reshape(
            total_samples, 1
        )  # Reshape back to (total_samples, 1)
    else:
        # If num_samples_per_prompt is 1, baseline is just the reward itself, advantage is 0
        # Or if num_prompts is 0, there are no advantages
        logger.debug(
            "Cannot calculate GRPO advantage baseline (num_samples_per_prompt <= 1 or num_prompts == 0). Setting advantages to rewards (equivalent to REINFORCE)."
        )
        advantages = rewards[
            :, None
        ]  # Use raw reward as advantage (REINFORCE behavior)

    mx.eval(advantages)

    # --- Calculate Reference Log Probabilities ---
    # Concatenate prompts (repeated) and generated responses
    full_sequence = mx.concatenate([prompts_mx_repeated, responses], axis=1)
    full_seq_len = full_sequence.shape[1]

    # Initialize outputs (use float32 for log probs)
    # Ensure these arrays are created even if generated_seq_len is 0
    ref_log_probs_response = mx.zeros(
        (total_samples, max(0, generated_seq_len)), dtype=mx.float32
    )
    response_mask_loss = mx.zeros(
        (total_samples, max(0, generated_seq_len)), dtype=mx.float32
    )

    if (
        generated_seq_len > 0 and full_seq_len > 1
    ):  # Need at least one generated token and sequence length > 1 for shifting
        try:
            # Use reference model (frozen)
            ref_model.eval()  # Ensure ref model is in eval mode
            ref_attention_mask_4d = _create_4d_attention_mask(
                full_sequence, pad_id, dtype=ref_model_dtype
            )
            logger.debug(f"Calling reference model. Input shape: {full_sequence.shape}")
            # Ensure input dtype matches ref model expectations if necessary
            ref_output = ref_model(
                full_sequence.astype(mx.int32)
                if ref_model_dtype != full_sequence.dtype
                else full_sequence,
                mask=ref_attention_mask_4d,
            )
            ref_logits = ref_output[0] if isinstance(ref_output, tuple) else ref_output
            # Ensure logits are float32 for selective_softmax stability
            ref_logits = ref_logits.astype(mx.float32)
            mx.eval(ref_logits)

            # Log probs of actual generated tokens (shifted view: logits[:, :-1, :] predicting tokens[:, 1:])
            log_probs_all_sequence = selective_softmax(
                ref_logits[:, :-1, :], full_sequence[:, 1:]
            )
            # Shape: (total_samples, full_seq_len - 1), dtype float32

            # --- Create Mask for Response Tokens ---
            # The generated sequence starts right after the padded prompt.
            # The logits are predicting the *next* token.
            # Logit at position K predicts token at position K+1.
            # The first generated token is at index max_prompt_len_batch in full_sequence.
            # Its log probability is predicted by the logit at index max_prompt_len_batch - 1.
            # So the relevant log probs start from index max_prompt_len_batch - 1 in log_probs_all_sequence.
            response_log_probs_start_idx_in_shifted = max_prompt_len_batch - 1

            if response_log_probs_start_idx_in_shifted < 0:
                response_log_probs_start_idx_in_shifted = (
                    0  # Should not happen with padding
                )
            if (
                response_log_probs_start_idx_in_shifted
                >= log_probs_all_sequence.shape[1]
            ):
                logger.warning(
                    f"Calculated response_log_probs_start_idx_in_shifted ({response_log_probs_start_idx_in_shifted}) is out of bounds for log_probs_all_sequence (shape {log_probs_all_sequence.shape}). Generated length issues? Setting ref_log_probs to zero."
                )
                # Keep zeroed tensors initialized earlier
            else:
                # Extract the part of log_probs_all_sequence that corresponds to the generated tokens
                # The size should match the generated sequence length
                try:
                    ref_log_probs_response_sliced = log_probs_all_sequence[
                        :,
                        response_log_probs_start_idx_in_shifted : response_log_probs_start_idx_in_shifted
                        + generated_seq_len,
                    ]
                    if ref_log_probs_response_sliced.shape[1] != generated_seq_len:
                        logger.error(
                            f"Sliced ref_log_probs_response has incorrect length. Expected {generated_seq_len}, got {ref_log_probs_response_sliced.shape[1]}. Skipping KL penalty calculation."
                        )
                        # Keep zeroed tensors
                    else:
                        ref_log_probs_response = (
                            ref_log_probs_response_sliced  # Update the tensor
                        )

                        # Create mask for the loss function: only for actual response tokens, excluding padding
                        # The tokens corresponding to the response are at indices max_prompt_len_batch onwards in full_sequence
                        # The log probs predicting these tokens are at indices max_prompt_len_batch - 1 onwards in full_sequence[:, 1:]
                        # Mask should cover indices [max_prompt_len_batch, max_prompt_len_batch + generated_seq_len - 1] in tokens[:, 1:]
                        # and exclude padding (pad_id)
                        response_tokens_in_full_seq = full_sequence[
                            :, max_prompt_len_batch:
                        ]  # Shape (B, GenLen)
                        # The mask needs to align with the *shifted* sequence index
                        # The mask should be True for generated, non-pad tokens
                        response_mask_loss = (
                            response_tokens_in_full_seq != pad_id
                        ).astype(
                            mx.float32
                        )  # Shape (B, GenLen)

                except Exception as e_slicing:
                    logger.error(
                        f"Error during ref log prob slicing or masking: {e_slicing}",
                        exc_info=True,
                    )
                    # Keep zeroed tensors

            mx.eval(ref_log_probs_response, response_mask_loss)

        except Exception as e_ref:
            logger.error(
                f"Reference model forward pass or log prob calculation failed: {e_ref}",
                exc_info=True,
            )
            # Keep zeroed tensors

    # --- Assemble Batch Data ---
    batch_data = {
        "tokens": full_sequence,  # Shape: (total_samples, full_seq_len)
        "response_mask": response_mask_loss,  # Shape: (total_samples, generated_seq_len), float32
        "advantages": advantages,  # Shape: (total_samples, 1), float32
        "ref_log_probs": ref_log_probs_response,  # Shape: (total_samples, generated_seq_len), float32
    }

    # Calculate overall average reward metrics for logging
    avg_reward_scalar = np.mean(rewards_list_total) if rewards_list_total else 0.0
    avg_format_reward_raw = np.mean(rewards_list_format) if rewards_list_format else 0.0
    avg_content_reward_raw = (
        np.mean(rewards_list_content) if rewards_list_content else 0.0
    )

    raw_reward_components = {
        "raw_format": avg_format_reward_raw,
        "raw_content": avg_content_reward_raw,
    }

    return batch_data, avg_reward_scalar, raw_reward_components


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


# --- Metrics Logger (Same as provided) ---
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
            needs_header = (
                not self.file_path.exists() or self.file_path.stat().st_size == 0
            )
            # Use buffering=1 for line buffering or buffering=-1 for default system buffer (often better performance)
            # Using 0 for unbuffered (binary mode) is generally not recommended for text.
            # Let's use default text buffering for simplicity unless performance is an issue.
            self._file = open(
                self.file_path, "a", newline="", encoding="utf-8"
            )  # Default buffering
            self.logger.info(
                f"Metrics CSV {'created' if needs_header else 'opened (append mode)'}: {self.file_path}"
            )
        except OSError as e:
            self.logger.error(
                f"Failed to open metrics file {self.file_path} for appending: {e}. CSV logging will be disabled.",
                exc_info=True,
            )
            self._file = None

    def log(self, metrics: Dict[str, Any]):
        if self._file is None or self._file.closed:
            if not self._logged_file_closed_warning:
                self.logger.warning(
                    f"Metrics file '{self.file_path.name}' is not open or has been closed. Cannot log metrics."
                )
                self._logged_file_closed_warning = True
            return

        loggable: Dict[str, Union[str, int, float, bool]] = {}
        for key, value in metrics.items():
            if isinstance(value, mx.array):
                try:
                    # Convert single-element MLX arrays to Python scalar types
                    if value.size == 1:
                        item = value.item()
                        loggable[key] = (
                            item if isinstance(item, (int, float, bool)) else str(item)
                        )
                    else:  # Convert multi-element arrays to string representation
                        loggable[key] = str(value.tolist())
                except Exception as e:
                    loggable[key] = f"[MLX conv error: {e}]"
            elif isinstance(value, np.ndarray):
                try:
                    # Convert single-element NumPy arrays to Python scalar types
                    if value.size == 1:
                        item = value.item()
                        loggable[key] = (
                            item if isinstance(item, (int, float, bool)) else str(item)
                        )
                    else:  # Convert multi-element arrays to string representation
                        loggable[key] = str(value.tolist())
                except Exception as e:
                    loggable[key] = f"[NumPy conv error: {e}]"
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
                    # If writer exists but headers changed, close and reopen file to rewrite header
                    if self._writer is not None:
                        self.logger.warning(
                            f"CSV headers changed. Old: {self._headers}, New: {current_headers}. Rewriting header."
                        )
                        self._file.flush()
                        self._file.close()
                        # Re-open in write mode to truncate, then append mode
                        self._file = open(
                            self.file_path, "w", newline="", encoding="utf-8"
                        )  # TRUNCATE
                        self._file.close()
                        self._file = open(
                            self.file_path, "a", newline="", encoding="utf-8"
                        )  # APPEND

                    self._headers = current_headers
                    self._writer = csv.DictWriter(
                        self._file, fieldnames=self._headers, extrasaction="ignore"
                    )
                    # Check if the file is empty *after* potentially truncating and re-opening
                    if self._file.tell() == 0:
                        self.logger.debug(f"Writing CSV header: {self._headers}")
                        self._writer.writeheader()

                self._writer.writerow(loggable)
                self._file.flush()  # Ensure data is written to disk immediately
            except Exception as e:
                self.logger.error(
                    f"Error writing metrics to CSV file '{self.file_path.name}': {e}",
                    exc_info=True,
                )

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
                    self.logger.error(
                        f"Error closing metrics file '{self.file_path.name}': {e}"
                    )
            elif self._file is None:
                self.logger.debug("Metrics file was already closed or never opened.")


# --- Checkpoint Loading/Saving (Revised for MLX state/RNG and Symlink) ---


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
            _save_directory_and_commit(
                "tokenizer_",  # 1st arg: temp_prefix (str)
                str(save_path),  # 2nd arg: target_directory_path (str)
                lambda tmp_dir_p: tokenizer.save_pretrained(
                    tmp_dir_p
                ),  # 3rd arg: save_fn (Callable)
            )
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


# --- GRPO Loss Function (Revised for clarity, uses float32 internally) ---
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
        logger.debug(
            f"Loss Fn: Full sequence length is {full_seq_len}, requires > 1 for shifted loss calculation. Returning zero loss."
        )
        return mx.zeros((), dtype=mx.float32)
    if batch_size == 0:
        logger.debug("Loss Fn: Received empty batch. Returning zero loss.")
        return mx.zeros((), dtype=mx.float32)

    generated_seq_len = response_mask.shape[1]
    if generated_seq_len == 0:
        logger.debug(
            "Loss Fn: Generated sequence length is 0. No response tokens to calculate loss on. Returning zero loss."
        )
        return mx.zeros((), dtype=mx.float32)

    # Determine model dtype for mask creation
    model_dtype = mx.bfloat16  # Default
    try:
        model_dtype = next(tree_flatten(model.parameters()))[1].dtype
    except (StopIteration, Exception):
        logger.debug(
            f"Loss Fn: Could not detect model dtype from parameters, using {model_dtype}."
        )

    # Model forward pass
    try:
        # Ensure input dtype matches model expectations
        model_input_tokens = (
            tokens.astype(mx.int32) if model_dtype != tokens.dtype else tokens
        )
        attn_mask_4d = _create_4d_attention_mask(
            tokens, pad_token_id, dtype=model_dtype
        )  # Mask creation still uses int tokens

        model_output = model(model_input_tokens, mask=attn_mask_4d)
        logits = model_output[0] if isinstance(model_output, tuple) else model_output
        # Use float32 for loss calculations for stability
        logits = logits.astype(mx.float32)

    except Exception as e:
        logger.error(f"Model forward pass failed in loss function: {e}", exc_info=True)
        return mx.zeros((), dtype=mx.float32)  # Return scalar zero loss

    # Calculate current log probs (pi_theta(a|s))
    # Logits are (batch, seq_len, vocab_size)
    # We need logits[:, :-1, :] to predict tokens[:, 1:]
    logits_for_next_token = logits[
        :, :-1, :
    ]  # Shape: (batch, full_seq_len - 1, vocab_size)

    # selective_softmax calculates log_probs for the actual tokens[:, 1:]
    current_log_probs_all_sequence = selective_softmax(
        logits_for_next_token, tokens[:, 1:]
    )
    # Shape: (batch, full_seq_len - 1), dtype float32

    # Align with response mask and reference log probs
    # The response mask and ref_log_probs cover only the generated tokens
    # The generated tokens start in the full sequence *after* the prompt padding.
    # The log probs for these generated tokens are located in the full_sequence[:, 1:] tensor.
    # The first token of the *response* is at index `max_prompt_len_batch` in the `tokens` tensor.
    # The log probability of this token is calculated using the logit at index `max_prompt_len_batch - 1`.
    # This corresponds to index `max_prompt_len_batch - 1` in the `current_log_probs_all_sequence` tensor.
    # Need to get the max_prompt_len_batch used when building the rollout batch.
    # This value isn't explicitly passed, but it is the length of the prompt part of `tokens`.
    prompt_len_actual = (
        tokens.shape[1] - generated_seq_len
    )  # Assuming no padding at the end of generated response
    # This assumption might be fragile if generation is stopped early or includes trailing padding.
    # A safer way is to determine the start index based on where the prompt ends.
    # The padding happens on the left, so prompt ends at index max_prompt_len_batch-1 in the *padded* prompt array.
    # The response starts at index max_prompt_len_batch in the full sequence.
    # The logits at index K predict token K+1.
    # Logits at `max_prompt_len_batch - 1` predict token at `max_prompt_len_batch` (the first response token).
    # So the relevant log probs start index in `current_log_probs_all_sequence` is `max_prompt_len_batch - 1`.
    # We need the original max_prompt_len_batch from `build_rollout_batch`.
    # Let's assume `tokens` is structured as [padding ... prompt | response | ... padding]
    # The split point should be consistent. The mask logic in generate_rollouts relies on this.

    # Find the index where the response starts in the *full sequence*
    # This is the length of the prompt part which includes left padding
    response_start_idx_in_full_sequence = full_seq_len - generated_seq_len
    if response_start_idx_in_full_sequence < 0:
        # This implies generated_seq_len > full_seq_len, which is an error
        logger.error(
            f"Generated sequence length ({generated_seq_len}) is greater than full sequence length ({full_seq_len}). Returning zero loss."
        )
        return mx.zeros((), dtype=mx.float32)

    # The log probs predicting the response start at index `response_start_idx_in_full_sequence - 1`
    # in the `current_log_probs_all_sequence` tensor (which is `full_sequence[:, 1:]`).
    response_log_probs_start_idx_in_shifted = response_start_idx_in_full_sequence - 1

    if response_log_probs_start_idx_in_shifted < 0:
        # This can happen if full_seq_len is 1, meaning the prompt was 1 token and generated 0.
        # If full_seq_len == 1, current_log_probs_all_sequence is empty.
        # If full_seq_len == 2, prompt is 1 token, generated 1 token. Logits[:, 0, :] predicts token[:, 1].
        # response_start_idx = 1. response_log_probs_start_idx_in_shifted = 0. Correct.
        if full_seq_len > 1:  # Should not be negative if full_seq_len > 1
            logger.error(
                f"Calculated response_log_probs_start_idx_in_shifted ({response_log_probs_start_idx_in_shifted}) is negative but full_seq_len is > 1 ({full_seq_len}). Indexing error. Returning zero loss."
            )
            return mx.zeros((), dtype=mx.float32)
        else:  # If full_seq_len <= 1, current_log_probs_all_sequence is empty or has len 0. Handled by initial checks.
            pass  # Safe to continue if generated_seq_len is also 0

    # Slice current log probs to match response length, aligning with response_mask and ref_log_probs
    # Only slice if there are generated tokens
    if generated_seq_len > 0:
        # The slice needs to go from `response_log_probs_start_idx_in_shifted`
        # for a length of `generated_seq_len`.
        current_log_probs_response_sliced = current_log_probs_all_sequence[
            :,
            response_log_probs_start_idx_in_shifted : response_log_probs_start_idx_in_shifted
            + generated_seq_len,
        ]

        # Ensure shapes match (all should be float32 now)
        if (
            current_log_probs_response_sliced.shape != response_mask.shape
            or ref_log_probs.shape != response_mask.shape
        ):
            logger.error(
                f"Shape mismatch in loss fn. Curr: {current_log_probs_response_sliced.shape}, Ref: {ref_log_probs.shape}, Mask: {response_mask.shape}. Returning zero loss."
            )
            return mx.zeros((), dtype=mx.float32)

        current_log_probs_response = (
            current_log_probs_response_sliced  # Use the sliced tensor
        )
    else:
        # If generated_seq_len is 0, these should be empty tensors
        current_log_probs_response = mx.zeros((batch_size, 0), dtype=mx.float32)
        # response_mask and ref_log_probs were initialized as zeros with shape (batch_size, 0)
        pass

    # Calculate log ratio: log(pi_theta / pi_ref) -> use response_mask later
    # All inputs are float32 here
    # log(A/B) = log(A) - log(B)
    # This should only be calculated for the response tokens.
    # The mask handles zeroing out contributions from padding.
    # Need to ensure ref_log_probs has the correct shape (batch, generated_seq_len) which it does.
    if generated_seq_len > 0:
        log_ratio = (
            current_log_probs_response - ref_log_probs
        )  # Shape: (batch, generated_seq_len)
    else:
        log_ratio = mx.zeros(
            (batch_size, 0), dtype=mx.float32
        )  # Empty tensor if no generation

    # Policy gradient term: - advantage * log_prob(action)
    # advantages shape: (batch, 1) -> broadcast
    # Use masked current_log_probs for PG term
    # Mask is (batch, generated_seq_len)
    if generated_seq_len > 0:
        pg_term = -advantages * (
            current_log_probs_response * response_mask
        )  # Shape: (batch, generated_seq_len)
    else:
        pg_term = mx.zeros((batch_size, 0), dtype=mx.float32)

    # KL divergence term: beta * (ratio - 1 - log_ratio) approx beta * KL(pi || pi_ref)
    # ratio = mx.exp(log_ratio)
    # Use mask here: only include KL penalty for actual response tokens
    # Note: If log_ratio is 0 (due to masking elsewhere), exp(0)=1, kl term is beta*(1-1-0)=0. Correct.
    # Apply mask at the end
    if generated_seq_len > 0:
        ratio_unmasked = mx.exp(log_ratio)
        kl_div_approx_unmasked = (
            ratio_unmasked - 1 - log_ratio
        )  # Shape: (batch, generated_seq_len)
        kl_term = (
            beta * kl_div_approx_unmasked * response_mask
        )  # Shape: (batch, generated_seq_len)
    else:
        kl_term = mx.zeros((batch_size, 0), dtype=mx.float32)

    # Combine terms: Loss = - [ PG - KL ] = -PG + KL
    # per_token_loss = pg_term - kl_term # PPO style: Loss = -min(ratio*adv, clip(ratio)*adv) + beta*KL
    # GRPO style loss is simpler, directly related to log_ratio and advantage
    # Loss = -E[log(pi_theta/pi_ref) * Advantage] + beta * E[KL(pi_theta || pi_ref)]
    # Where expectation is over sampled actions (tokens) and states (sequence positions).
    # This should be sum over tokens, mean over batch.
    # The per-token loss is: - log_ratio * Advantage + beta * KL_approx
    # Per-token loss needs to be masked.
    if generated_seq_len > 0:
        per_token_loss = (
            -log_ratio * advantages + beta * kl_div_approx_unmasked
        ) * response_mask  # Shape: (batch, generated_seq_len)
    else:
        per_token_loss = mx.zeros((batch_size, 0), dtype=mx.float32)

    # Sum over sequence dimension, normalize by number of valid response tokens
    token_counts = mx.sum(response_mask, axis=1)  # (batch,)
    # Ensure sum happens over correct axis, handle potential division by zero
    # Sum per_token_loss over the sequence dimension
    sequence_loss_sum = mx.sum(per_token_loss, axis=1)  # (batch,)

    # Avoid division by zero if a sample had 0 generated tokens (mask sum is 0)
    # If token_counts is 0, the loss for that sample should be 0.
    # sequence_loss = sequence_loss_sum / mx.maximum(token_counts, 1e-8) # Old way, might give large values
    # Better: use where or mask *after* division (if token_counts > 0) or before sum.
    # Let's mask the sum:
    # Use a mask where token_counts > 0
    non_zero_count_mask = (token_counts > 0).astype(sequence_loss_sum.dtype)
    # Divide the sum by the count for samples where count > 0
    sequence_loss = sequence_loss_sum / (
        token_counts + (1 - non_zero_count_mask)
    )  # Add 1 where count is 0 to avoid DivByZero, then mask it out
    sequence_loss = (
        sequence_loss * non_zero_count_mask
    )  # Zero out loss for samples with 0 generated tokens

    # Average loss over the batch
    loss = mx.mean(sequence_loss)

    # Check for NaN/Inf
    if mx.isnan(loss).any() or mx.isinf(loss).any():
        logger.error(f"NaN or Inf detected in GRPO loss! Loss: {loss.item()}")
        # Add component logging if needed
        # Need to calculate masked sums/means for logging
        if generated_seq_len > 0:
            masked_pg_sum_per_sample = mx.sum(pg_term, axis=1)  # Sum over sequence
            masked_kl_sum_per_sample = mx.sum(kl_term, axis=1)
            # Use non_zero_count_mask to average only over samples with generated tokens
            num_samples_with_tokens = mx.sum(
                (token_counts > 0).astype(mx.float32)
            ).item()
            num_samples_with_tokens = max(
                num_samples_with_tokens, 1.0
            )  # Avoid div by zero if no samples have tokens

            avg_pg_loss_approx = (
                mx.sum(masked_pg_sum_per_sample * non_zero_count_mask).item()
                / num_samples_with_tokens
            )
            avg_kl_loss_approx = (
                mx.sum(masked_kl_sum_per_sample * non_zero_count_mask).item()
                / num_samples_with_tokens
            )
            logger.error(
                f"Approx Components (Avg per sample w/ tokens): PG={avg_pg_loss_approx:.4f}, KL={avg_kl_loss_approx:.4f}"
            )
            logger.error(
                f"Advantage Stats: mean={mx.mean(advantages).item():.4f}, std={mx.std(advantages).item():.4f}"
            )
            # Need to mask log_ratio before stats
            masked_log_ratio = log_ratio * response_mask
            # Compute mean/std only over non-zero masked values
            num_masked_tokens = mx.sum(response_mask).item()
            num_masked_tokens = max(num_masked_tokens, 1.0)  # Avoid div by zero
            mean_log_ratio = mx.sum(masked_log_ratio).item() / num_masked_tokens
            # Calculate std dev only over the relevant values
            # Find non-zero elements and calculate std dev
            flat_masked_log_ratio = masked_log_ratio.flatten()
            relevant_log_ratios = flat_masked_log_ratio[flat_masked_log_ratio != 0]
            std_log_ratio = (
                mx.std(relevant_log_ratios).item()
                if relevant_log_ratios.size > 1
                else 0.0
            )

            logger.error(
                f"LogRatio Stats (masked tokens): mean={mean_log_ratio:.4f}, std={std_log_ratio:.4f}"
            )

        raise ValueError("NaN/Inf loss detected")

    return loss


# --- REVISED Evaluation Function ---
def evaluate_grpo(
    model: nn.Module,
    dataset: Dataset,
    tokenizer: TokenizerWrapper,
    args: TrainingArgs,
    model_config: Dict,  # Pass model config for potential use
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

    # --- Setup ---
    total_rewards, total_format_rewards, total_content_rewards = [], [], []
    samples_processed = 0
    rollout_examples = []  # To store examples for logging
    reward_config = RewardConfig(  # Use tags from args
        think_start_tag=args.think_start_tag,
        think_end_tag=args.think_end_tag,
        answer_start_tag=args.answer_start_tag,
        answer_end_tag=args.answer_end_tag,
    )
    try:
        get_content_reward_fn(args.reward_content_type)  # Validate content type exists
    except ValueError:
        eval_logger.error(
            f"Invalid reward_content_type '{args.reward_content_type}'. Evaluation aborted."
        )
        return {}

    eval_batch_size = args.ppo_batch_size  # Use same micro-batch size for consistency
    indices = list(range(max_iters))
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        error_msg = "Tokenizer needs pad_token_id for evaluation padding."
        eval_logger.critical(error_msg)
        # Decide if this should halt evaluation or training - better to halt training setup.
        # For evaluation function itself, return empty metrics.
        return {}

    eos_token_str = (
        tokenizer.decode(
            [eos_id], skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        if eos_id is not None
        else ""
    )
    pad_token_str = (
        tokenizer.decode(
            [pad_id], skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        if pad_id is not None
        else ""
    )

    # --- Determine Model Dtype ---
    model_dtype = mx.bfloat16  # Default
    try:
        model_dtype = next(tree_flatten(model.parameters()))[1].dtype
    except (StopIteration, Exception):
        eval_logger.warning(
            f"Eval: Could not detect model dtype from parameters, using {model_dtype}."
        )

    # --- Evaluation Loop ---
    model.eval()  # Set model to eval mode
    if progress and task_id:
        progress.reset(task_id, total=max_iters, description="Evaluating...")

    # Greedy sampler (temp=0.0, top_p=0.0)
    eval_sampler = make_sampler(temp=0.0, top_p=0.0)

    for i in range(0, max_iters, eval_batch_size):
        if shutdown_requested:
            eval_logger.info("Shutdown requested. Stopping evaluation.")
            break
        batch_indices = indices[i : i + eval_batch_size]
        if not batch_indices:
            continue

        # --- Prepare Batch (using chat template) ---
        # num_samples_per_prompt is 1 for greedy evaluation
        prompts_data_batch, prompts_mx, max_prompt_len_batch = build_rollout_batch(
            tokenizer,
            dataset,
            batch_indices,
            1,
            args.max_prompt_len,
            args.system_prompt,
            args.dataset_prompt_key,
            args.dataset_answer_key,
        )
        if not prompts_data_batch:
            eval_logger.warning(
                f"Skipping evaluation batch starting at index {i}: No valid prompts found."
            )
            if progress and task_id:
                progress.update(
                    task_id, advance=len(batch_indices)
                )  # Still advance bar for skipped items
            continue

        batch_size_actual = prompts_mx.shape[0]
        if batch_size_actual == 0:
            eval_logger.warning(
                f"Skipping evaluation batch starting at index {i}: prompts_mx is empty after build_rollout_batch."
            )
            if progress and task_id:
                progress.update(
                    task_id, advance=len(batch_indices)
                )  # Still advance bar
            continue

        batch_responses_tokens = []
        cache = None

        # --- Generate Responses (Greedy, Batch) ---
        try:
            prompt_attn_mask_4d = _create_4d_attention_mask(
                prompts_mx, pad_id, dtype=model_dtype
            )
            # Ensure input dtype matches model expectations
            model_input_prompts = (
                prompts_mx.astype(model_dtype)
                if model_dtype != prompts_mx.dtype
                else prompts_mx
            )
            model_output = model(
                model_input_prompts, mask=prompt_attn_mask_4d, cache=cache
            )
            if isinstance(model_output, tuple):
                logits_prompt, cache = model_output[:2]
            else:
                logits_prompt, cache = model_output, None
            next_token_logits = logits_prompt[:, -1, :]
            mx.eval(next_token_logits, cache)

            current_tokens = eval_sampler(next_token_logits)
            batch_responses_tokens.append(current_tokens[:, None])  # Append first token

            ended = mx.zeros_like(current_tokens)
            if eos_id is not None:
                ended = mx.equal(current_tokens, eos_id)

            for step in range(
                args.max_gen_len - 1
            ):  # Generate remaining max_gen_len - 1 tokens
                if mx.all(ended).item():
                    eval_logger.debug(
                        f"All sequences in eval batch ended generation at step {step}."
                    )
                    break  # Exit generation loop if all sequences are done

                # Pass the previously sampled token (current_tokens) to the model
                model_input_step = (
                    current_tokens[:, None].astype(model_dtype)
                    if model_dtype != current_tokens.dtype
                    else current_tokens[:, None]
                )
                model_step_output = model(model_input_step, mask=None, cache=cache)
                if isinstance(model_step_output, tuple):
                    logits_step, cache = model_step_output[:2]
                else:
                    logits_step, cache = model_step_output, None
                next_token_logits = logits_step[:, -1, :]
                sampled_tokens = eval_sampler(
                    next_token_logits
                )  # Sample for the full batch

                just_ended = mx.zeros_like(ended)
                if eos_id is not None:
                    just_ended = mx.equal(sampled_tokens, eos_id)

                ended = mx.logical_or(ended, just_ended)
                # Use the ended state *before* the update to decide on padding
                ended_at_start_of_step = ended.copy()  # Capture state

                # For sequences that *were already ended* before this step, use pad_id instead of the sampled token
                pad_val = mx.array(pad_id, dtype=sampled_tokens.dtype)
                tokens_to_add = mx.where(
                    ended_at_start_of_step, pad_val, sampled_tokens
                )

                mx.eval(
                    tokens_to_add, cache, ended
                )  # Evaluate tokens_to_add, cache, and the new ended state

                batch_responses_tokens.append(tokens_to_add[:, None])
                current_tokens = tokens_to_add

            # Ensure evaluation of collected tokens list after loop
            if batch_responses_tokens:
                mx.eval(batch_responses_tokens)

        except Exception as e_eval_gen:
            eval_logger.error(
                f"Error during evaluation generation for batch starting at index {i}: {e_eval_gen}",
                exc_info=True,
            )
            batch_responses_tokens = []  # Signal error by making list empty

        # --- Decode and Calculate Rewards ---
        if batch_responses_tokens:  # Check list is not empty
            try:
                responses_mx = mx.concatenate(batch_responses_tokens, axis=1)
                mx.eval(responses_mx)
                # Decode without skipping special tokens initially to preserve tags
                decoded_responses = tokenizer.batch_decode(
                    responses_mx.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            except Exception as e_decode:
                eval_logger.error(f"Evaluation batch decoding failed: {e_decode}")
                decoded_responses = ["[decoding error]"] * batch_size_actual
        else:  # Generation failed completely or no tokens were generated
            eval_logger.warning(
                f"No tokens generated for evaluation batch starting at index {i}. Using empty responses."
            )
            decoded_responses = [""] * batch_size_actual

        for j in range(batch_size_actual):
            # Get the corresponding prompt data for this generated response (index j in the batch)
            # Since num_samples_per_prompt is 1, the prompt index is just j relative to the batch start
            resp_text = decoded_responses[j]
            # Get ref answer string from the prompts_data_batch (aligned with prompts_mx batch)
            ref_ans = prompts_data_batch[j]["ref_answer_str"]
            # Get the original formatted prompt text for logging
            prompt_text_for_log = prompts_data_batch[j]["text"]

            # Clean response: remove padding, truncate at first EOS
            cleaned_resp = resp_text
            if pad_token_str and cleaned_resp.endswith(pad_token_str):
                # Use regex to remove trailing pad tokens and surrounding whitespace robustly
                cleaned_resp = re.sub(
                    rf"(?:\s*{re.escape(pad_token_str)})+$", "", cleaned_resp
                )
                cleaned_resp = (
                    cleaned_resp.rstrip()
                )  # Clean any remaining trailing whitespace
            eos_pos = cleaned_resp.find(eos_token_str) if eos_token_str else -1
            if eos_pos != -1:
                cleaned_resp = cleaned_resp[
                    :eos_pos
                ].strip()  # Strip after truncating at EOS

            if cleaned_resp.startswith("[") and "error]" in cleaned_resp:
                total_rew, fmt_rew, content_rew_combined = 0.0, 0.0, 0.0
            else:
                # calculate_total_reward returns (total_weighted_reward, raw_format_reward, raw_content_reward_combined)
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
            total_content_rewards.append(
                content_rew_combined
            )  # Store the combined content reward
            samples_processed += 1

            # Store examples for logging (take from first batch)
            if (
                len(rollout_examples) < min(5, batch_size_actual) and i == 0
            ):  # Log first 5 examples from the very first eval batch
                rollout_examples.append(
                    {
                        # Use the formatted prompt text stored in prompts_data_batch
                        "prompt": prompt_text_for_log[-250:] + "..."
                        if prompt_text_for_log
                        else "[empty]",
                        "generated": cleaned_resp[:350] + "..."
                        if cleaned_resp
                        else "[empty]",
                        "reference": ref_ans if ref_ans is not None else "N/A",
                        "reward": f"{total_rew:.3f} (Fmt:{fmt_rew:.2f}, Cont:{content_rew_combined:.2f})",
                    }
                )

        if progress and task_id:
            progress.update(task_id, advance=batch_size_actual)

    model.train()  # Set model back to training mode

    # --- Calculate and Log Final Metrics ---
    mean_total_reward = np.mean(total_rewards) if total_rewards else 0.0
    mean_format_reward_raw = (
        np.mean(total_format_rewards) if total_format_rewards else 0.0
    )
    # Average of the *combined* content reward
    mean_content_reward_raw_combined = (
        np.mean(total_content_rewards) if total_content_rewards else 0.0
    )

    eval_logger.info(
        f"Evaluation finished. Processed: {samples_processed}/{max_iters} samples."
    )
    eval_logger.info(
        f"  [bold]Mean Total Reward (Weighted): {mean_total_reward:.4f}[/]"
    )
    eval_logger.info(f"  Mean Format Reward (Raw): {mean_format_reward_raw:.4f}")
    eval_logger.info(
        f"  Mean Content Reward (Raw/Combined): {mean_content_reward_raw_combined:.4f} (Type: {args.reward_content_type})"
    )

    if rollout_examples:
        # Recreate table to ensure correct width calculation after potentially missing samples
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
        table.add_column(
            "Reward Details", style="bold yellow", justify="right", min_width=25
        )
        for ex in rollout_examples:
            table.add_row(ex["prompt"], ex["generated"], ex["reference"], ex["reward"])
        console.print(table)

    final_metrics = {
        "eval/mean_reward_weighted": mean_total_reward,
        "eval/mean_format_reward_raw": mean_format_reward_raw,
        "eval/mean_content_reward_raw_combined": mean_content_reward_raw_combined,  # Log the combined value
        "eval/samples_processed": float(samples_processed),
    }
    return final_metrics


# --- Training Orchestration (Minor adjustments for new tokens/init) ---
def _get_learning_rate(
    update_step: int,
    decay_steps: Optional[int],
    learning_rate: float,
    decay_rate: float,
    min_lr: float,
) -> float:
    """Calculates the potentially decayed learning rate using a simple step decay."""
    # (Same as provided, noting MLX limitations)
    if decay_steps is None or decay_steps <= 0 or update_step < decay_steps:
        return learning_rate
    else:
        # Simple step decay: drop LR every 'decay_steps' updates
        num_decay_periods = (
            update_step - 1
        ) // decay_steps  # Use update_step - 1 so decay happens *after* the first step of the new period
        lr = learning_rate * (decay_rate**num_decay_periods)
        return max(lr, min_lr)


def train(
    args: TrainingArgs,
    actor_model: nn.Module,
    ref_model: nn.Module,
    model_config: dict,
    tokenizer: TokenizerWrapper,
    train_dset: Dataset,
    val_dset: Optional[Dataset],
    optimizer: optim.Optimizer,
    start_step: int,
    start_updates: int,
):
    """Main training loop for GRPO."""
    global shutdown_requested, mlx_rng_key  # Allow modification
    train_logger = logging.getLogger(__name__)
    wandb_run = None

    # --- WandB Initialization ---
    if args.use_wandb:
        if WANDB_AVAILABLE:
            try:
                wandb_config_dict = asdict(args)
                for k, v in wandb_config_dict.items():
                    if isinstance(v, Path):
                        wandb_config_dict[k] = str(v)
                # Use a fixed ID for resuming if starting from a checkpoint
                # This assumes checkpoint saving logic includes the wandb run ID if resuming was enabled.
                # For simplicity, let's generate a new ID if not resuming from a checkpoint
                run_id = None  # Or load from checkpoint state if saved there?
                # If resuming, maybe read the run ID from the training_state.json args snapshot?
                # This is complex, let's stick to 'allow' resume strategy for now.
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name,
                    config=wandb_config_dict,
                    resume="allow",  # wandb handles resuming based on process info/config
                )
                train_logger.info(
                    f"WandB logging enabled. Run: {wandb.run.name} ({wandb.run.url})"
                )
            except Exception as e:
                train_logger.error(
                    f"WandB initialization failed: {e}. Disabling WandB.", exc_info=True
                )
                args.use_wandb = False
        else:
            train_logger.warning(
                "WandB requested but not available. Disabling WandB logging."
            )
            args.use_wandb = False

    # --- Gradient Checkpointing Setup (Optional, keep if needed) ---
    if args.use_grad_checkpointing:
        train_logger.info(
            "Gradient checkpointing enabled (ensure layer wrapping works for your model)."
        )
        # Need to identify the TransformerDecoderLayer or equivalent in your specific model architecture.
        # Common names: `model.layers` (Llama), `transformer.h` (GPT).
        # This part needs to be adapted based on the model loaded.
        try:
            # Attempt to find the layers attribute
            layers_module = None
            if hasattr(actor_model, "model") and hasattr(actor_model.model, "layers"):
                layers_module = actor_model.model.layers
            elif hasattr(actor_model, "transformer") and hasattr(
                actor_model.transformer, "h"
            ):
                layers_module = actor_model.transformer.h
            # Add checks for other model types if needed

            if layers_module and isinstance(layers_module, (list, nn.Sequential)):
                num_layers_total = len(layers_module)
                num_ckpt = (
                    args.grad_checkpoint_layers
                    if args.grad_checkpoint_layers is not None
                    and args.grad_checkpoint_layers > 0
                    else num_layers_total
                )
                num_ckpt = min(num_ckpt, num_layers_total)  # Cap at total layers
                # Apply to the top `num_ckpt` layers
                indices_to_checkpoint = list(
                    range(num_layers_total - num_ckpt, num_layers_total)
                )
                # Ensure indices are valid
                indices_to_checkpoint = [
                    i for i in indices_to_checkpoint if 0 <= i < num_layers_total
                ]

                if indices_to_checkpoint:
                    train_logger.info(
                        f"Applying grad checkpoint to {len(indices_to_checkpoint)}/{num_layers_total} layers (top {num_ckpt}). Indices: {indices_to_checkpoint}"
                    )
                    applied_count = 0
                    for i in indices_to_checkpoint:
                        try:
                            # grad_checkpoint wraps the module in-place
                            # Check if it's already checkpointed to avoid double wrapping?
                            # grad_checkpoint function might handle this internally.
                            if not hasattr(
                                layers_module[i], "_checkpointed_wrapped"
                            ):  # Basic check if already wrapped
                                grad_checkpoint(layers_module[i])  # Apply wrapping
                                applied_count += 1
                            # else: logger.debug(f"Layer {i} already appears to be checkpointed.") # Avoid noisy log

                        except Exception as e_gc_apply:
                            train_logger.error(
                                f"Failed applying grad checkpoint to layer {i}: {e_gc_apply}"
                            )
                    train_logger.info(
                        f"Successfully applied grad checkpointing to {applied_count}/{len(indices_to_checkpoint)} specified layers."
                    )
                else:
                    train_logger.warning(
                        "No layers selected or found for gradient checkpointing based on args."
                    )
            else:
                train_logger.warning(
                    "Could not find suitable layers for gradient checkpointing in the model structure."
                )

        except Exception as e_gc_setup:
            train_logger.error(
                f"Error during gradient checkpointing setup: {e_gc_setup}"
            )

    # --- Training State Initialization ---
    global_step = start_step  # Loaded from checkpoint or 0
    num_updates = start_updates  # Loaded from checkpoint or 0

    # Ensure model/optimizer evaluated after loading/init
    mx.eval(actor_model.parameters())
    mx.eval(optimizer.state)

    # --- Setup Training ---
    start_time = time.monotonic()
    last_save_update = num_updates
    last_eval_update = num_updates
    peak_mem_gb = 0.0

    output_dir = Path(args.output_dir)
    metrics_logger = MetricsLogger(output_dir / "training_metrics.csv")
    reward_config = RewardConfig(  # Use args tags
        think_start_tag=args.think_start_tag,
        think_end_tag=args.think_end_tag,
        answer_start_tag=args.answer_start_tag,
        answer_end_tag=args.answer_end_tag,
    )

    train_logger.info(
        f"Training outputs will be saved to: [link=file://{output_dir.resolve()}]{output_dir}[/]"
    )

    # Prepare loss function and gradient calculation
    # The lambda function captures `args.grpo_beta` and `tokenizer.pad_token_id`
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
    # Compile the value_and_grad_fn for performance
    # Requires defining the input shapes/dtypes. Can be complex with variable sequence lengths.
    # Let's skip compilation for simplicity unless profiling shows it's a bottleneck.
    # If compiling, need to handle variable batch/seq dims:
    # @mx.compile
    # def compiled_loss_fn(model, tokens, response_mask, advantages, ref_log_probs):
    #     return grpo_loss(model, tokens, response_mask, advantages, ref_log_probs, ...)

    train_indices = list(range(len(train_dset)))
    total_updates_target = args.num_training_steps
    train_logger.info(
        f"Starting GRPO training from update {num_updates}. Target updates: {total_updates_target}. Effective batch size (samples per update): {args.effective_batch_size}"
    )

    # --- Setup Rich Progress Bar ---
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

    # --- Main Training Loop ---
    try:
        with Progress(
            *progress_cols, console=console, transient=False, refresh_per_second=2
        ) as progress:
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

            current_indices_pos = 0  # Track position in shuffled indices

            while num_updates < total_updates_target and not shutdown_requested:
                # Shuffle data at start of each 'epoch' (pass through data) if shuffle is enabled
                # An 'epoch' here is defined as processing `len(train_dset)` samples in total across all rollouts.
                # Since effective batch size is `ppo_batch_size * grad_accum_steps * num_rollout_samples`,
                # the number of dataset samples consumed per update is `ppo_batch_size * grad_accum_steps`.
                # Number of updates per epoch is roughly len(train_dset) / (ppo_batch_size * grad_accum_steps)
                # For simplicity, let's just shuffle indices at the start of the training loop, or every N updates.
                # The current logic shuffles when `current_indices_pos == 0`. This means it shuffles
                # after consuming all indices once.

                if args.shuffle_data and current_indices_pos == 0:
                    random.shuffle(train_indices)
                    train_logger.debug("Shuffled training indices.")
                elif not args.shuffle_data and current_indices_pos == 0:
                    # If not shuffling, reset position to 0 after one pass
                    pass  # Already happens implicitly by modulo arithmetic below

                # --- Accumulation Phase ---
                accumulated_grads = None
                accum_count = 0
                effective_batch_losses, effective_batch_rewards_weighted = [], []
                effective_batch_rewards_fmt, effective_batch_rewards_content = [], []

                # Get indices for the prompts in this effective batch
                # Need args.ppo_batch_size * args.grad_accum_steps unique prompts per effective batch
                num_prompts_for_effective_batch = (
                    args.ppo_batch_size * args.grad_accum_steps
                )
                effective_batch_prompt_indices = []
                start_pos = current_indices_pos
                for k in range(num_prompts_for_effective_batch):
                    # Wrap around the training indices if needed
                    idx_pos = (start_pos + k) % len(train_indices)
                    effective_batch_prompt_indices.append(train_indices[idx_pos])
                current_indices_pos = (
                    start_pos + num_prompts_for_effective_batch
                ) % len(
                    train_indices
                )  # Update position for next effective batch

                # Split into micro-batches of prompt indices
                micro_batch_indices_list = [
                    effective_batch_prompt_indices[k : k + args.ppo_batch_size]
                    for k in range(
                        0, num_prompts_for_effective_batch, args.ppo_batch_size
                    )
                ]

                for micro_batch_indices in micro_batch_indices_list:
                    if shutdown_requested:
                        break  # Check signal before processing micro-batch
                    if not micro_batch_indices:
                        continue

                    # --- Build Rollout Batch for Micro-batch ---
                    # This returns prompts_data (includes ref_ans), prompts_mx (padded tokens), max_prompt_len_batch
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
                            args.num_rollout_samples,
                            args.max_prompt_len,
                            args.system_prompt,
                            args.dataset_prompt_key,
                            args.dataset_answer_key,
                        )
                        if not prompts_data_mb:
                            train_logger.warning(
                                f"Skipping micro-batch (Update {num_updates+1}, accum {accum_count+1}): No valid prompts from indices {micro_batch_indices}."
                            )
                            continue  # Skip to next micro-batch

                        # --- Generate Rollouts for Micro-batch ---
                        # Pass the prompts_data list to the generation function for correct alignment
                        #
                        # rollout_data, avg_rollout_rew_w, raw_rew_comp = generate_rollouts_for_batch(
                        #                             actor_model, ref_model, tokenizer, prompts_mx_mb, max_prompt_len_b,
                        #                             ref_answers_mb, args.num_rollout_samples, args.max_gen_len, args
                        #                         )
                        (
                            rollout_data,
                            avg_rollout_rew_w,
                            raw_rew_comp,
                        ) = generate_rollouts_for_batch(
                            actor_model,
                            ref_model,
                            tokenizer,
                            prompts_data_mb,
                            prompts_mx_mb,
                            max_prompt_len_mb,  # <<< Pass prompts_data_mb and other required args
                            args.num_rollout_samples,
                            args.max_gen_len,
                            args,
                        )

                        # Check if rollout_data is valid (e.g., not empty due to generation errors)
                        if (
                            rollout_data["tokens"].shape[0] == 0
                            or rollout_data["response_mask"].shape[1] == 0
                        ):
                            train_logger.warning(
                                f"Skipping micro-batch (Update {num_updates+1}, accum {accum_count+1}): Rollout generation resulted in empty sequences."
                            )
                            continue  # Skip to next micro-batch

                        rollout_dur = time.monotonic() - rollout_start
                        effective_batch_rewards_weighted.append(avg_rollout_rew_w)
                        effective_batch_rewards_fmt.append(
                            raw_rew_comp.get("raw_format", math.nan)
                        )  # Use .get for robustness
                        effective_batch_rewards_content.append(
                            raw_rew_comp.get("raw_content_combined", math.nan)
                        )  # Use .get for robustness

                        # Update progress bar reward fields (average over micro-batches seen so far)
                        progress.update(
                            main_task,
                            roll_rew=np.nanmean(effective_batch_rewards_weighted)
                            if effective_batch_rewards_weighted
                            else math.nan,
                            fmt_rew=np.nanmean(effective_batch_rewards_fmt)
                            if effective_batch_rewards_fmt
                            else math.nan,
                            cont_rew=np.nanmean(effective_batch_rewards_content)
                            if effective_batch_rewards_content
                            else math.nan,
                        )

                    except Exception as e_rollout:
                        train_logger.error(
                            f"Rollout failed for micro-batch (Update {num_updates+1}, accum {accum_count+1}): {e_rollout}",
                            exc_info=True,
                        )
                        # Discard results from this micro-batch and continue to the next micro-batch
                        continue  # Skip to next micro-batch within the accumulation loop

                    # --- Loss Calculation & Gradient Accumulation ---
                    loss_calc_start = time.monotonic()
                    try:
                        # Use the rollout_data dictionary directly as keyword arguments
                        loss, grads_tree = value_and_grad_fn(
                            actor_model, **rollout_data
                        )
                        mx.eval(loss, grads_tree)  # Evaluate loss & grads
                        loss_item = loss.item()
                        if math.isnan(loss_item) or math.isinf(loss_item):
                            train_logger.error(
                                f"NaN/Inf loss ({loss_item:.4f}) detected in micro-batch {accum_count+1}. Skipping grad accumulation for this micro-batch."
                            )
                            continue  # Skip accumulation for this micro-batch, try next micro-batch

                        effective_batch_losses.append(loss_item)

                        # Scale gradients by 1 / grad_accum_steps before accumulating
                        # This ensures the final accumulated grad represents the mean gradient
                        scaled_grads_tree = tree_map(
                            lambda g: g / args.grad_accum_steps, grads_tree
                        )

                        if accumulated_grads is None:
                            accumulated_grads = scaled_grads_tree
                        else:
                            # Use tree_map to handle potential None values if a gradient was missing
                            # This requires a lambda that handles None or ensure tree_map does.
                            # Default tree_map should handle structures correctly.
                            accumulated_grads = tree_map(
                                lambda acc, new: acc + new
                                if acc is not None and new is not None
                                else acc or new,
                                accumulated_grads,
                                scaled_grads_tree,
                            )

                        accum_count += 1
                        loss_calc_dur = time.monotonic() - loss_calc_start
                        train_logger.debug(
                            f"Loss/Grad (Upd {num_updates}, MB {accum_count}/{args.grad_accum_steps}): Loss: {loss_item:.4f}. Time: {loss_calc_dur:.2f}s."
                        )

                    except (
                        ValueError
                    ) as e_loss_val:  # Catch NaN/Inf from loss function raise
                        train_logger.error(
                            f"Loss/Grad calculation failed (NaN/Inf) for micro-batch {accum_count+1} (Update {num_updates+1}): {e_loss_val}. Discarding effective batch gradients.",
                            exc_info=True,
                        )
                        accumulated_grads = (
                            None  # Discard potentially partial accumulated gradients
                        )
                        effective_batch_losses = (
                            []
                        )  # Clear losses for this effective batch
                        break  # Exit micro-batch loop for this effective batch
                    except Exception as e_loss:
                        train_logger.error(
                            f"Loss/Grad calculation failed for micro-batch (Update {num_updates+1}, accum {accum_count+1}): {e_loss}",
                            exc_info=True,
                        )
                        accumulated_grads = (
                            None  # Discard potentially partial accumulated gradients
                        )
                        effective_batch_losses = (
                            []
                        )  # Clear losses for this effective batch
                        break  # Exit micro-batch loop for this effective batch

                # --- End of Micro-batch Loop for Accumulation ---

                if shutdown_requested:
                    break  # Check signal before optimizer step

                # --- Optimizer Step (if accumulation complete and grads valid) ---
                # An optimizer step happens only if we successfully accumulated gradients
                # for the required number of accumulation steps.
                if (
                    accum_count == args.grad_accum_steps
                    and accumulated_grads is not None
                ):
                    update_start_time = time.monotonic()
                    num_updates += 1  # Increment update counter *before* optimizer step

                    try:
                        # Evaluate accumulated grads
                        mx.eval(accumulated_grads)

                        # Calculate Target LR for logging and potentially application
                        current_lr_target = _get_learning_rate(
                            num_updates,
                            args.lr_decay_steps,
                            args.learning_rate,
                            args.lr_decay_rate,
                            args.lr_decay_min_lr,
                        )
                        # TODO: Manually update optimizer's learning rate if LR decay is desired and implemented

                        # Apply Gradient Clipping
                        final_grads = accumulated_grads
                        grad_norm_mx = mx.array(math.nan)  # Initialize
                        grad_norm = math.nan
                        if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                            final_grads, grad_norm_mx = optim.clip_grad_norm(
                                final_grads, args.grad_clip_norm
                            )
                            mx.eval(
                                final_grads, grad_norm_mx
                            )  # Eval clipped grads and norm
                            grad_norm = grad_norm_mx.item()
                        else:  # Compute norm anyway for logging if clipping is off
                            try:
                                grad_norm_mx = optim.compute_grad_norm(
                                    accumulated_grads
                                )
                                mx.eval(grad_norm_mx)
                                grad_norm = grad_norm_mx.item()
                            except Exception:
                                pass  # Keep NaN if norm computation fails

                        # Apply Gradients
                        # Optimizer applies the mean gradient calculated over the effective batch
                        # Need to get the trainable parameters from the model
                        trainable_params = actor_model.trainable_parameters()
                        # Check that the structure of grads and trainable_params match
                        # This is implicitly handled by tree_map in apply_gradients
                        optimizer.apply_gradients(final_grads, trainable_params)
                        # Note: MLX optimizers modify parameters in-place

                        mx.eval(
                            trainable_params, optimizer.state
                        )  # Evaluate updated parameters and optimizer state

                        # Get average metrics for this effective batch
                        avg_loss_step = (
                            np.mean(effective_batch_losses)
                            if effective_batch_losses
                            else math.nan
                        )
                        # Using np.nanmean to handle potential NaNs if some micro-batches were skipped
                        avg_reward_step_weighted = (
                            np.nanmean(effective_batch_rewards_weighted)
                            if effective_batch_rewards_weighted
                            else math.nan
                        )
                        avg_reward_step_fmt = (
                            np.nanmean(effective_batch_rewards_fmt)
                            if effective_batch_rewards_fmt
                            else math.nan
                        )
                        avg_reward_step_content = (
                            np.nanmean(effective_batch_rewards_content)
                            if effective_batch_rewards_content
                            else math.nan
                        )

                        # --- Logging & Progress Update ---
                        update_duration = time.monotonic() - update_start_time
                        progress.update(
                            main_task,
                            advance=1,
                            completed=num_updates,
                            loss=avg_loss_step,
                            roll_rew=avg_reward_step_weighted,
                            fmt_rew=avg_reward_step_fmt,
                            cont_rew=avg_reward_step_content,
                            lr=current_lr_target,
                            grad_norm=grad_norm,
                        )

                        # --- Log Metrics ---
                        mem_metric = {}
                        if PSUTIL_AVAILABLE:
                            try:
                                process = psutil.Process(os.getpid())
                                mem_info = process.memory_info()
                                current_mem_gb = mem_info.rss / (1024**3)
                                peak_mem_gb = max(peak_mem_gb, current_mem_gb)
                                mlx_mem_peak_gb = (
                                    mx.metal.get_peak_memory() / (1024**3)
                                    if hasattr(mx, "metal") and mx.metal.is_available()
                                    else 0.0
                                )
                                mlx_mem_current_gb = (
                                    mx.metal.get_active_memory() / (1024**3)
                                    if hasattr(mx, "metal") and mx.metal.is_available()
                                    else 0.0
                                )
                                mem_metric = {
                                    "memory/process_peak_gb": peak_mem_gb,
                                    "memory/process_current_gb": current_mem_gb,
                                    "memory/mlx_peak_gb": mlx_mem_peak_gb,
                                    "memory/mlx_current_gb": mlx_mem_current_gb,
                                }
                            except Exception:
                                pass  # Silently fail memory logging if psutil or mlx.metal have issues

                        metrics_to_log = {
                            "train/loss": avg_loss_step,
                            "train/mean_rollout_reward_weighted": avg_reward_step_weighted,
                            "train/mean_format_reward_raw": avg_reward_step_fmt,
                            "train/mean_content_reward_raw_combined": avg_reward_step_content,
                            "train/learning_rate_target": current_lr_target,
                            "train/grad_norm": grad_norm
                            if not math.isnan(grad_norm)
                            else 0.0,
                            "timers/update_duration_sec": update_duration,
                            "timers/rollout_duration_sec": rollout_dur,  # From last successful micro-batch in accumulation
                            "train/accum_count": accum_count,  # Log how many micro-batches were successful
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

                        # Log summary periodically to console
                        if (
                            num_updates % args.eval_every == 0
                            or num_updates == 1
                            or num_updates == total_updates_target
                            or num_updates % 10 == 0
                        ):  # Log more often
                            train_logger.info(
                                f"Update {num_updates}/{total_updates_target} | "
                                f"Loss: {avg_loss_step:.4f} | RollRew(Wt): {avg_reward_step_weighted:.3f} | "
                                f"RawFmt: {avg_reward_step_fmt:.2f} | RawCont(Comb): {avg_reward_step_content:.2f} ({args.reward_content_type}) | "
                                f"GradNorm: {grad_norm:.2f} | LR: {current_lr_target:.1e} | "
                                f"Time: {update_duration:.2f}s"
                            )
                        # Reset accumulated_grads and counts for the next update step
                        accumulated_grads = None
                        accum_count = 0
                        effective_batch_losses, effective_batch_rewards_weighted = (
                            [],
                            [],
                        )
                        effective_batch_rewards_fmt, effective_batch_rewards_content = (
                            [],
                            [],
                        )

                    except Exception as e_update:
                        train_logger.error(
                            f"Optimizer step failed for update {num_updates}: {e_update}",
                            exc_info=True,
                        )
                        # Continue training loop, but this update failed.
                        # Grads and accum count are already reset, so next loop iteration starts fresh accumulation.

                # --- Periodic Actions (Eval, Save) after completing an Optimizer Step ---
                # These should ideally happen *after* a successful optimizer step (update_step incremented)
                if (
                    num_updates > last_save_update or num_updates > last_eval_update
                ):  # Only check if update happened
                    # --- Evaluation Step ---
                    perform_eval = (val_dset is not None) and (
                        (num_updates % args.eval_every == 0)
                        or (num_updates >= total_updates_target)
                    )
                    if perform_eval and num_updates > last_eval_update:
                        eval_start = time.monotonic()
                        train_logger.info(
                            f"--- Starting Evaluation @ Update {num_updates} ---"
                        )
                        try:
                            with Progress(
                                SpinnerColumn(),
                                TextColumn("[cyan]{task.description}"),
                                BarColumn(),
                                MofNCompleteColumn(),
                                TimeElapsedColumn(),
                                console=console,
                                transient=True,
                            ) as eval_prog:
                                eval_task = eval_prog.add_task(
                                    f"Evaluating", total=len(val_dset)
                                )
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
                            train_logger.info(
                                f"--- Evaluation Finished ({eval_dur:.2f}s) ---"
                            )
                            if eval_metrics:
                                fm = {k: f"{v:.4f}" for k, v in eval_metrics.items()}
                                train_logger.info(
                                    f"Eval Metrics @ Update {num_updates}: {fm}"
                                )
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
                                        train_logger.error(
                                            f"WandB eval logging failed: {e_wandb_eval}"
                                        )
                            last_eval_update = num_updates
                        except Exception as e_eval:
                            train_logger.error(
                                f"Evaluation failed at update {num_updates}: {e_eval}",
                                exc_info=True,
                            )

                    # --- Checkpoint Saving ---
                    should_save, save_reason = False, ""
                    # Prioritize exit requests and shutdown signals
                    if SAVE_ON_EXIT_FLAG_PATH.exists():
                        should_save, save_reason = True, "exit_request"
                        logger.warning("Save-on-exit flag detected, initiating save...")
                        try:
                            SAVE_ON_EXIT_FLAG_PATH.unlink()
                        except OSError:
                            pass  # Ignore error if unlink fails
                    elif shutdown_requested:
                        should_save, save_reason = True, "shutdown_signal"
                    # Periodic save happens only if not already covered by exit/shutdown or end of training
                    elif num_updates % args.save_every == 0:
                        should_save, save_reason = True, "periodic"
                    # Final save happens only at the very end if not already covered
                    elif num_updates >= total_updates_target:
                        should_save, save_reason = True, "final_step"

                    # Perform save only if prompted and we haven't saved at this update step already
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

                # --- Check for Termination Conditions ---
                # Check after potential save/eval
                if num_updates >= total_updates_target:
                    train_logger.info(
                        f"Target number of updates ({total_updates_target}) reached."
                    )
                    break  # Exit outer (while) loop
                if shutdown_requested:
                    train_logger.info("Shutdown requested, exiting training loop.")
                    break  # Exit outer loop

            # End of while loop (training updates)
            final_loss = (
                progress.tasks[main_task].fields.get("loss", math.nan)
                if progress.tasks
                else math.nan
            )  # Safely get loss
            progress.update(
                main_task,
                description=f"Training Finished (Loss: {final_loss:.3f})",
                refresh=True,
            )  # Final refresh

    except KeyboardInterrupt:  # Explicitly catch Ctrl+C
        train_logger.warning("\nTraining interrupted by user (KeyboardInterrupt).")
        shutdown_requested = True  # Ensure shutdown flag is set

    except Exception as train_err:
        train_logger.critical("Critical error during training loop!", exc_info=True)
        console.print_exception(show_locals=args.verbose)
        exit_code = 1  # Set error exit code
        # Attempt emergency save
        if num_updates > 0:  # Only save if at least one update occurred
            try:
                train_logger.warning(
                    "Attempting emergency checkpoint save due to critical error..."
                )
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
                train_logger.error(
                    f"Emergency checkpoint save FAILED: {final_save_err}"
                )
    finally:
        # --- Cleanup and Final Reporting ---
        total_training_duration = time.monotonic() - start_time
        final_status = (
            "Completed"
            if num_updates >= total_updates_target and not shutdown_requested
            else "Interrupted"
        )
        train_logger.info(f"--- Training {final_status} ---")
        train_logger.info(
            f" Total Updates Performed: {num_updates}/{total_updates_target}"
        )
        train_logger.info(
            f" Total Duration: {total_training_duration / 3600:.2f} hrs ({total_training_duration:.1f} sec)"
        )
        if PSUTIL_AVAILABLE:
            train_logger.info(f" Peak Process Memory Usage: {peak_mem_gb:.2f} GB")
        if hasattr(mx, "metal") and mx.metal.is_available():
            # Capture final MLX memory usage before exit
            final_mlx_peak_gb = mx.metal.get_peak_memory() / (1024**3)
            final_mlx_active_gb = mx.metal.get_active_memory() / (1024**3)
            train_logger.info(f" Peak MLX Metal Memory: {final_mlx_peak_gb:.2f} GB")
            train_logger.info(
                f" Final MLX Metal Active Memory: {final_mlx_active_gb:.2f} GB"
            )

        save_reason_final = locals().get("save_reason", "N/A")
        train_logger.info(f" Final Checkpoint Reason: {save_reason_final}")
        train_logger.info(
            f" Log files and checkpoints saved in: [link=file://{output_dir.resolve()}]{output_dir}[/]"
        )

        # Final save if needed (ensure we don't double-save if a periodic/final save just occurred)
        # Check if the last save occurred at the current num_updates
        saved_at_final_update = last_save_update == num_updates

        if shutdown_requested and not saved_at_final_update:
            train_logger.info("Performing final save due to interruption...")
            # Use current step and num_updates for the interrupted save
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
            # Use current step and num_updates for the final save
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

        # Close resources
        metrics_logger.close()
        if wandb_run:
            train_logger.info("Finishing WandB run...")
            try:
                wandb.finish(exit_code=exit_code)
            except Exception as e_wb_finish:
                logger.error(f"WandB finish failed: {e_wb_finish}")

        # Clean up save flag file
        if SAVE_ON_EXIT_FLAG_PATH.exists():
            train_logger.info(f"Cleaning up final save flag: {SAVE_ON_EXIT_FLAG_PATH}")
            try:
                SAVE_ON_EXIT_FLAG_PATH.unlink()
            except OSError:
                pass  # Ignore if already gone


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
        elif hasattr(model, "embed_tokens"):  # Root level?
            embeddings = model.embed_tokens
        else:
            logger.error(
                "Could not find embedding layer ('embed_tokens' or 'transformer.wte') in the model structure."
            )
            raise ValueError("Embedding layer not found.")

        if not isinstance(embeddings, nn.Embedding):
            logger.error(
                f"Found embedding attribute is not an nn.Embedding layer: {type(embeddings)}"
            )
            raise ValueError("Attribute is not an Embedding layer.")

        embedding_matrix = embeddings.weight
        num_existing_tokens = embedding_matrix.shape[0] - len(new_token_ids)
        embedding_dim = embedding_matrix.shape[1]
        logger.info(
            f"Initializing embeddings for {len(new_token_ids)} new tokens (IDs: {new_token_ids}). Existing vocab: {num_existing_tokens}, Dim: {embedding_dim}."
        )

        if num_existing_tokens <= 0:
            logger.warning(
                "No existing tokens to compute mean embedding from. Using default random initialization."
            )
            init_with_mean = False  # Fallback to default

        if init_with_mean:
            logger.info("Calculating mean of existing embeddings...")
            # Calculate mean on existing embeddings only
            mean_embedding = mx.mean(
                embedding_matrix[:num_existing_tokens], axis=0, keepdims=False
            )  # keepdims=False for 1D vector
            # Ensure mean is computed and has the right shape/dtype
            mx.eval(mean_embedding)
            if mean_embedding.shape != (embedding_dim,):
                raise ValueError(
                    f"Mean embedding shape incorrect: Expected ({embedding_dim},), got {mean_embedding.shape}"
                )

            logger.info("Assigning mean embedding to new tokens...")
            for token_id in new_token_ids:
                if token_id < 0 or token_id >= embedding_matrix.shape[0]:
                    logger.warning(
                        f"Token ID {token_id} out of bounds for embedding matrix shape {embedding_matrix.shape}. Skipping initialization for this ID."
                    )
                    continue
                # Directly update the weight matrix row
                # Ensure the mean_embedding is broadcastable or has matching shape (embedding_dim,)
                embeddings.weight[
                    token_id
                ] = mean_embedding  # Assign the 1D mean vector

            mx.eval(
                embeddings.weight[new_token_ids]
            )  # Evaluate the changes for the specific rows
            logger.info("Mean embedding initialization complete.")

        else:
            # Default initialization (often random small values) is usually handled by nn.Embedding resize.
            # Log that default init is used.
            logger.info(
                "Using default random initialization for new token embeddings (usually done by resize)."
            )

    except Exception as e:
        logger.error(f"Failed to initialize new token embeddings: {e}", exc_info=True)
        # Decide whether to raise or continue with potentially uninitialized embeddings
        # Raising is safer as training might fail with uninitialized embeddings
        raise  # Raising is safer to ensure user knows init failed


def check_and_set_resume_path(args):
    """
    Checks if resuming is requested and potentially uses the 'latest' symlink
    in the output directory if no specific checkpoint is given.
    Modifies args.resume_from_checkpoint in-place.
    """
    logger = logging.getLogger(__name__)

    # Convert output_dir to Path object for easier handling
    if args.output_dir:
        output_dir_path = Path(args.output_dir)

        # Check if resume is NOT explicitly specified
        if not args.resume_from_checkpoint:
            logger.info("No specific --resume-from-checkpoint provided.")
            # Check if the output directory exists and contains a 'latest' symlink
            if output_dir_path.is_dir():
                latest_symlink_path = output_dir_path / "latest"
                if latest_symlink_path.is_symlink():
                    try:
                        # Resolve the symlink to get the actual target directory path
                        resolved_checkpoint_path = latest_symlink_path.resolve()

                        if resolved_checkpoint_path.is_dir():
                            # Update args.resume_from_checkpoint with the resolved path
                            args.resume_from_checkpoint = str(resolved_checkpoint_path)
                            logger.info(
                                f"Found 'latest' symlink pointing to existing directory."
                            )
                            logger.info(
                                f"Setting resume_from_checkpoint to: [cyan]{args.resume_from_checkpoint}[/cyan]"
                            )
                        else:
                            logger.warning(
                                f"'latest' symlink exists at {latest_symlink_path} but points to a non-directory: {resolved_checkpoint_path}. Skipping automatic resume."
                            )

                    except OSError as e:
                        logger.warning(
                            f"Could not resolve 'latest' symlink at {latest_symlink_path}: {e}. Skipping automatic resume."
                        )
                    except Exception as e:
                        logger.error(
                            f"An unexpected error occurred while resolving 'latest' symlink at {latest_symlink_path}: {e}",
                            exc_info=True,
                        )
                else:
                    logger.info(
                        f"'latest' symlink not found in existing output directory {args.output_dir}. Starting new run."
                    )
            else:
                logger.info(
                    f"Output directory {args.output_dir} does not exist. Starting new run."
                )
        else:  # resume_from_checkpoint was explicitly provided
            logger.info(
                f"Resuming from explicitly provided checkpoint: [cyan]{args.resume_from_checkpoint}[/cyan]"
            )

    else:
        # This case should ideally be caught by argparse validation (output_dir is mandatory)
        logger.error(
            "args.output_dir is not set. Cannot check for 'latest' symlink. This should not happen with current args validation."
        )
        # If it did happen, args.resume_from_checkpoint would remain None.

    # The function modifies args in-place, no need to return it unless the calling
    # code needs the object itself (though args is already passed by reference implicitly).


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
                cliname = f"--no-{f.name.replace('', '-')}"
                kwargs["action"] = "store_false"
                kwargs["help"] += " (disables this option)"
            else:
                kwargs["action"] = "store_true"
        elif is_optional_type:
            actual_type = str
            for arg_t in type_args:
                if arg_t is not type(None) and arg_t in [int, float, str]:
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
                kwargs["help"] = (
                    f"(Required) {kwargs['help']}" if kwargs["help"] else "(Required)"
                )

        if (
            kwargs.get("required", False)
            and f.default is MISSING
            and f.default_factory is MISSING
        ):
            kwargs.pop("default", None)

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


# --- Main Execution (Revised for Special Tokens/Init) ---
def main():
    """Main function to parse arguments, set up, and run training."""
    # Memory limit attempt (keep as is) - This attempts to set a hard memory limit (in MB)
    # on the *process*. Be cautious with this; it might kill the process abruptly
    # if the limit is reached. 60GB might be too large for typical systems.
    # Consider commenting this out or adjusting the limit based on your hardware.
    # try: limit_memory(60 * 1024) # limit_memory expects MB
    # except Exception as e: print(f"Warning: Failed to set memory limit: {e}", file=sys.stderr)

    args = parse_arguments()  # Parse validated args
    check_and_set_resume_path(
        args
    )  # Check for 'latest' symlink if --resume-from-checkpoint wasn't set

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
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        log_fmt = "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        file_handler.setFormatter(logging.Formatter(log_fmt, date_fmt))
        handlers.append(file_handler)
        log_file_msg = (
            f"Debug Log: [link=file://{log_file.resolve()}]{log_file.name}[/]"
        )
    except Exception as e:
        print(f"Warning: Could not set up file logging in '{args.output_dir}': {e}")
    # Configure the root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )
    # Set levels for specific loggers to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    # Set mlx logger level based on verbosity, but keep it below DEBUG unless verbose
    mlx_log_level = logging.INFO if not args.verbose else logging.DEBUG
    # Prevent mlx debug logs if overall level is INFO
    if log_level > logging.DEBUG:
        mlx_log_level = logging.INFO
    logging.getLogger("mlx").setLevel(mlx_log_level)  # Control mlx verbosity
    logging.getLogger("transformers").setLevel(
        logging.WARNING
    )  # Suppress transformers warnings

    logger.info(
        f"Logging Level: Console={logging.getLevelName(log_level)}, File=DEBUG. {log_file_msg}"
    )

    # --- Check Dependency Availability ---
    if args.use_wandb and not WANDB_AVAILABLE:
        logger.warning(
            "WandB requested but not installed (`pip install wandb`). Disabling."
        )
        args.use_wandb = False
    # Check if at least one dataset source is provided
    if args.train_dataset_path is None and args.dataset_name is None:
        logger.critical(
            "No training dataset source specified. Please provide --train-dataset-path or --dataset-name. Exiting."
        )
        sys.exit(1)
    # Check if HF dataset is specified but datasets library is missing
    if args.dataset_name is not None and not DATASETS_AVAILABLE:
        logger.critical(
            "Hugging Face dataset specified (--dataset-name) but `datasets` library is not installed (`pip install datasets`). Exiting."
        )
        sys.exit(1)

    # --- Setup Signal Handling ---
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    logger.info("Registered signal handlers for SIGINT and SIGTERM.")
    # Clean up stale save flag on startup
    if SAVE_ON_EXIT_FLAG_PATH.exists():
        logger.warning(
            f"Removing stale save flag file from previous run: {SAVE_ON_EXIT_FLAG_PATH}"
        )
        try:
            SAVE_ON_EXIT_FLAG_PATH.unlink()
        except OSError as e:
            logger.error(f"Failed to remove stale save flag: {e}", exc_info=False)

    # --- Display Configuration ---
    logger.info("[bold green]--- Effective Training Configuration ---[/]")
    config_table = Table(show_header=False, box=None, padding=(0, 1), title="Arguments")
    config_table.add_column("Parameter", style="dim cyan", justify="right")
    config_table.add_column("Value", style="white")
    args_dict_display = asdict(args)
    for key, value in sorted(args_dict_display.items()):
        # Mask sensitive values like paths that might contain usernames or keys (basic masking)
        display_value = str(value)
        if (
            "path" in key.lower() or "token" in key.lower() and isinstance(value, str)
        ):  # More general path/token check
            display_value = "***"  # Mask potential paths/tokens
        # Also mask Hugging Face dataset/config details if not necessary to display
        # Or just display all args as they are config.
        config_table.add_row(key, display_value)
    console.print(config_table)
    logger.info("[bold green]------------------------------------[/]")

    # --- Load Dataset ---
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

    # --- Initialize Models, Tokenizer, Optimizer ---
    actor_model: Optional[nn.Module] = None
    tokenizer: Optional[TokenizerWrapper] = None
    ref_model: Optional[nn.Module] = None
    model_config: Optional[Dict] = None
    optimizer: Optional[optim.Optimizer] = None
    start_step = 0
    start_updates = 0
    rng_restored_from_checkpoint = False
    newly_added_token_ids = []  # Track IDs of newly added special tokens

    # --- Load from Checkpoint OR Initialize New ---
    if args.resume_from_checkpoint:
        # --- Resuming Logic ---
        logger.info(
            f"Attempting to resume training from checkpoint: [cyan]{args.resume_from_checkpoint}[/]"
        )
        ckpt_path = Path(args.resume_from_checkpoint)
        config_path = ckpt_path / "config.json"
        tokenizer_path = ckpt_path  # Load tokenizer from checkpoint dir

        if not config_path.exists():
            logger.critical(f"Resume failed: config.json not found in {ckpt_path}")
            sys.exit(1)
        # Check for tokenizer files (e.g., tokenizer.json, vocab.json, tokenizer_config.json)
        if not any(
            list(tokenizer_path.glob("tokenizer*.*"))
            + list(tokenizer_path.glob("vocab*.*"))
            + list(tokenizer_path.glob("special_tokens_map.json"))
        ):
            logger.critical(
                f"Resume failed: Tokenizer file(s) not found in {ckpt_path}. Cannot load tokenizer."
            )
            sys.exit(1)

        try:
            # 1. Load config from checkpoint
            with open(config_path, "r") as f:
                model_config = json.load(f)
            logger.info(
                f"Loaded model config from checkpoint (Type: {model_config.get('model_type', 'Unknown')})."
            )

            # 2. Load Tokenizer from checkpoint dir FIRST (needed for model structure and padding)
            tokenizer = load_tokenizer(str(tokenizer_path))
            logger.info(
                f"Tokenizer loaded from checkpoint: {tokenizer_path.name}, Vocab size: {tokenizer.vocab_size}"
            )
            # Add special tokens defined in args to tokenizer loaded from checkpoint
            # This is important if args has new tokens that weren't in the checkpointed tokenizer
            special_tokens_to_add_args = {
                "additional_special_tokens": sorted(
                    list(
                        set(
                            [  # Use set to avoid duplicates
                                args.think_start_tag.strip(),
                                args.think_end_tag.strip(),
                                args.answer_start_tag.strip(),
                                args.answer_end_tag.strip(),
                            ]
                        )
                    )
                )
            }
            num_tokens_before_ckpt = tokenizer.vocab_size
            num_added_ckpt = tokenizer.add_special_tokens(special_tokens_to_add_args)
            if num_added_ckpt > 0:
                logger.info(
                    f"Added {num_added_ckpt} new special tokens from args to checkpoint tokenizer. New vocab size: {tokenizer.vocab_size}."
                )
                # Get IDs of tokens newly added *in this step* (not from checkpoint)
                newly_added_token_ids = list(
                    range(num_tokens_before_ckpt, tokenizer.vocab_size)
                )

            # 3. Rebuild model structure (using mlx_lm utilities is robust)
            # The `load` function in mlx_lm can often rebuild the structure and load weights.
            # If LoRA was used, it *must* apply LoRA layers *before* attempting to load weights.
            logger.info(
                "Rebuilding actor model structure from config and loading checkpoint weights..."
            )
            # Pass checkpoint path to load function. It should handle full or adapter loading.
            actor_model, _ = load(
                ckpt_path
            )  # `load` can often handle rebuilding structure + loading weights later
            logger.info(
                f"Rebuilt actor model structure and loaded weights: {type(actor_model).__name__}"
            )

            # If new tokens were added to the tokenizer, resize model embeddings
            if num_added_ckpt > 0:
                logger.info(
                    f"Resizing actor model embeddings for {num_added_ckpt} newly added tokens..."
                )
                actor_model.resize_token_embeddings(tokenizer.vocab_size)
                logger.info(
                    f"Actor model embeddings resized to {tokenizer.vocab_size}."
                )
                mx.eval(actor_model.parameters())  # Evaluate after resize

                # Initialize embeddings for the tokens added *in this step* if requested
                if args.init_new_embeddings_with_mean and newly_added_token_ids:
                    initialize_new_token_embeddings(
                        actor_model,
                        tokenizer,
                        newly_added_token_ids,
                        args.init_new_embeddings_with_mean,
                    )
                    mx.eval(actor_model.parameters())  # Evaluate after init

            # 4. Load reference model (always fresh based on ref_model_path)
            # The ref model also needs the updated tokenizer size if tokens were added
            logger.info(
                f"Loading reference model: [cyan]{args.ref_model_path}[/cyan]..."
            )
            ref_model, ref_tokenizer = load(
                Path(args.ref_model_path)
            )  # Load ref model and its tokenizer
            if ref_tokenizer.vocab_size != tokenizer.vocab_size:
                logger.warning(
                    f"Reference model tokenizer vocab size ({ref_tokenizer.vocab_size}) does not match actor tokenizer vocab size ({tokenizer.vocab_size}). Resizing ref model embeddings."
                )
                ref_model.resize_token_embeddings(tokenizer.vocab_size)
                mx.eval(ref_model.parameters())  # Evaluate after resize
                # Initialize embeddings for new tokens in ref model if different from actor model path
                # Only if they were added *to the actor tokenizer* and need to be synced to ref.
                if (
                    num_added_ckpt > 0
                    and args.model_path != args.ref_model_path
                    and newly_added_token_ids
                ):
                    logger.info("Initializing new embeddings in reference model...")
                    initialize_new_token_embeddings(
                        ref_model,
                        ref_tokenizer,
                        newly_added_token_ids,
                        args.init_new_embeddings_with_mean,
                    )  # Use ref_tokenizer here for info, but ids are from actor tokenizer add
                    mx.eval(ref_model.parameters())  # Evaluate after init

            ref_model.freeze()
            ref_model.eval()
            mx.eval(ref_model.parameters())
            logger.info(
                f"Reference model loaded ({type(ref_model).__name__}) and frozen."
            )

            # 5. Initialize optimizer shell (weights are loaded by `load_checkpoint` below)
            # The optimizer object needs to be created *before* loading its state
            optimizer = optim.AdamW(
                learning_rate=args.learning_rate,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                weight_decay=args.optimizer_weight_decay,
            )
            logger.info(f"Initialized new AdamW optimizer for resuming.")

            # 6. Load optimizer state and training progress (step, updates, RNG)
            # This modifies the optimizer object and sets RNG
            start_step, start_updates, rng_restored_from_checkpoint = load_checkpoint(
                ckpt_path, args, actor_model, optimizer
            )

        except Exception as e:
            logger.critical(
                f"Failed during checkpoint loading/rebuilding: {e}", exc_info=True
            )
            sys.exit(1)

    else:
        # --- Initialize New Training Run ---
        logger.info(
            f"Loading base model/tokenizer from: [cyan]{args.model_path}[/cyan]..."
        )
        try:
            # Load base actor model and tokenizer
            actor_model, tokenizer = load(Path(args.model_path))
            if not isinstance(actor_model, nn.Module):
                raise TypeError(
                    f"Loaded actor model is not an nn.Module: {type(actor_model)}"
                )
            logger.info(f"Base actor model loaded: {type(actor_model).__name__}")

            # Load config from the base model path
            model_path_obj = get_model_path(args.model_path)
            model_config = load_config(model_path_obj)
            logger.info(
                f"Model config loaded for {model_config.get('model_type', 'Unknown Type')}"
            )

            # --- Add Special Tokens & Resize Embeddings ---
            special_tokens_to_add = {
                "additional_special_tokens": sorted(
                    list(
                        set(
                            [  # Use set to avoid duplicates
                                args.think_start_tag.strip(),
                                args.think_end_tag.strip(),
                                args.answer_start_tag.strip(),
                                args.answer_end_tag.strip(),
                            ]
                        )
                    )
                )
            }
            logger.info(
                f"Adding special tokens to tokenizer: {special_tokens_to_add['additional_special_tokens']}"
            )
            num_tokens_before = tokenizer.vocab_size
            num_added = tokenizer.add_special_tokens(special_tokens_to_add)
            num_tokens_after = tokenizer.vocab_size
            num_new_tokens = num_tokens_after - num_tokens_before

            if num_new_tokens > 0:
                logger.info(
                    f"Added {num_new_tokens} new special tokens. Resizing model embeddings..."
                )
                # Store the IDs of the newly added tokens for initialization
                newly_added_token_ids = list(range(num_tokens_before, num_tokens_after))
                # Resize actor model embeddings
                actor_model.resize_token_embeddings(tokenizer.vocab_size)
                logger.info(
                    f"Actor model embeddings resized to {tokenizer.vocab_size}."
                )
                mx.eval(actor_model.parameters())  # Evaluate after resize

                # --- Initialize New Embeddings ---
                if args.init_new_embeddings_with_mean:
                    initialize_new_token_embeddings(
                        actor_model,
                        tokenizer,
                        newly_added_token_ids,
                        args.init_new_embeddings_with_mean,
                    )
                    mx.eval(actor_model.parameters())  # Evaluate after init

            else:
                logger.info(
                    "No new special tokens were added (they might exist already in the base tokenizer)."
                )
                # Still collect existing IDs for format/answer tags if they exist, for potential init check?
                # No, initialize_new_token_embeddings is only for tokens *newly* added to this tokenizer instance.
                # If they were already there, their embeddings are loaded with the base model.

            # --- Apply LoRA (if needed, AFTER resizing) ---
            if args.use_lora:
                if args.adapter_path:
                    logger.info(f"Loading adapters from: {args.adapter_path}")
                    # Load adapters. This modifies the model structure (adds LoRA layers)
                    # and loads weights into them.
                    actor_model = load_adapters(
                        actor_model, args.adapter_path
                    )  # Modifies model in-place
                    logger.info("Loaded and applied LoRA adapters.")
                else:
                    logger.info("Applying NEW LoRA layers...")
                    lora_config_args = {
                        "rank": args.lora_rank,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                        "scale": args.lora_scale,
                    }
                    # This adds LoRA layers with random initialization
                    linear_to_lora_layers(
                        actor_model, args.lora_layers, lora_config_args
                    )
                    logger.info("Applied new LoRA layers with random initialization.")
                mx.eval(
                    actor_model.parameters()
                )  # Evaluate after applying/loading adapters

            # Load reference model
            logger.info(
                f"Loading reference model from: [cyan]{args.ref_model_path}[/cyan]..."
            )
            ref_model, ref_tokenizer = load(Path(args.ref_model_path))
            if ref_tokenizer.vocab_size != tokenizer.vocab_size:
                logger.warning(
                    f"Reference model tokenizer vocab size ({ref_tokenizer.vocab_size}) does not match actor tokenizer vocab size ({tokenizer.vocab_size}). Resizing ref model embeddings."
                )
                ref_model.resize_token_embeddings(tokenizer.vocab_size)
                mx.eval(ref_model.parameters())  # Evaluate after resize
                # Initialize embeddings for new tokens in ref model if it's different from actor model
                # This ensures the KL divergence is calculated with correctly initialized embeddings for the ref model too
                if (
                    num_new_tokens > 0
                    and args.model_path != args.ref_model_path
                    and newly_added_token_ids
                ):
                    logger.info("Initializing new embeddings in reference model...")
                    initialize_new_token_embeddings(
                        ref_model,
                        ref_tokenizer,
                        newly_added_token_ids,
                        args.init_new_embeddings_with_mean,
                    )  # Use ref_tokenizer here for info, but ids are from actor tokenizer add
                    mx.eval(ref_model.parameters())  # Evaluate after init

            ref_model.freeze()
            ref_model.eval()
            mx.eval(ref_model.parameters())
            logger.info(
                f"Reference model loaded ({type(ref_model).__name__}) and frozen."
            )

            # --- Initialize Optimizer for New Run ---
            # Get trainable parameters *after* LoRA layers have been added
            trainable_params_dict = actor_model.trainable_parameters()
            trainable_params_list = tree_flatten(trainable_params_dict)
            trainable_param_arrays = [p for _, p in trainable_params_list]

            if not trainable_param_arrays:
                mode = "LoRA" if args.use_lora else "Full parameter"
                logger.warning(
                    f"No trainable parameters found ({mode} mode)! Check LoRA/adapter setup or model freezing. Training will not proceed."
                )
                # Consider exiting here if no trainable params are found
                # sys.exit(1) # Maybe raise error instead?
                raise ValueError("No trainable parameters found.")

            else:
                num_trainable = sum(p.size for p in trainable_param_arrays)
                logger.info(
                    f"Creating optimizer for {len(trainable_param_arrays)} trainable tensors ({num_trainable:,} parameters)."
                )

            optimizer = optim.AdamW(
                learning_rate=args.learning_rate,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                weight_decay=args.optimizer_weight_decay,
            )
            logger.info(
                f"Initialized new AdamW optimizer (LR={args.learning_rate:.2e})"
            )

        except Exception as e:
            logger.critical(
                f"Failed to load base/ref model, tokenizer, or apply LoRA/init tokens: {e}",
                exc_info=True,
            )
            sys.exit(1)

    # --- Final Checks & Seeding ---
    if (
        actor_model is None
        or ref_model is None
        or tokenizer is None
        or model_config is None
        or optimizer is None
    ):
        logger.critical("Initialization failed: Core components missing. Exiting.")
        sys.exit(1)
    if not actor_model.trainable_parameters() and not args.resume_from_checkpoint:
        # If not resuming and no trainable parameters were found during setup, something is wrong
        logger.critical(
            "No trainable parameters found and not resuming from checkpoint. Cannot train. Exiting."
        )
        sys.exit(1)

    print_trainable_parameters(actor_model)  # Print summary

    # --- Check PAD Token ---
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            logger.warning(
                f"Tokenizer missing PAD token. Using EOS token '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id}) as PAD."
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
            # Update underlying tokenizer if possible (best effort)
            if hasattr(tokenizer, "tokenizer") and hasattr(
                tokenizer.tokenizer, "pad_token_id"
            ):
                tokenizer.tokenizer.pad_token_id = tokenizer.eos_token_id
            if hasattr(tokenizer, "tokenizer") and hasattr(
                tokenizer.tokenizer, "pad_token"
            ):
                tokenizer.tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.critical(
                "Tokenizer has no PAD token and no EOS token! Cannot proceed."
            )
            sys.exit(1)
    else:
        logger.info(
            f"Using Tokenizer PAD token ID: {tokenizer.pad_token_id} ('{tokenizer.pad_token}')"
        )

    # Add PAD token to special_tokens_map if it's not there after setting it
    if tokenizer.pad_token and (
        tokenizer.pad_token not in tokenizer.special_tokens_map
        or tokenizer.special_tokens_map.get("pad_token") != tokenizer.pad_token
    ):
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
    mx.eval(actor_model.parameters(), ref_model.parameters(), optimizer.state)
    logger.info("Initial state evaluation complete.")

    # --- Start Training Process ---
    logger.info("[bold blue]>>> Starting GRPO Training Process <<<[/]")
    training_start_time = time.monotonic()
    exit_code = 0
    try:
        # Pass starting steps/updates loaded from checkpoint
        train(
            args,
            actor_model,
            ref_model,
            model_config,
            tokenizer,
            train_dset,
            val_dset,
            optimizer,
            start_step,
            start_updates,
        )
    except Exception as train_err:
        logger.critical(
            "Training process terminated due to an unhandled exception.", exc_info=True
        )
        console.print_exception(show_locals=args.verbose)
        exit_code = 1
        # Exception handling within train() already attempts emergency save
    finally:
        # --- Final Cleanup ---
        total_duration_mins = (time.monotonic() - training_start_time) / 60
        logger.info(f"Total script execution time: {total_duration_mins:.2f} minutes.")
        logger.info("[bold blue]>>> Training Script Finished <<<[/]")
        logging.shutdown()  # Attempt to flush logs
        sys.exit(exit_code)  # Exit with the determined code


# --- Entry Point ---
if __name__ == "__main__":
    rprint("\n[bold underline]GRPO MLX Trainer (Aligned)[/]")
    rprint("[dim] - Uses `tokenizer.apply_chat_template`.")
    rprint("[dim] - Adds special tokens, resizes embeddings, and initializes them.")
    rprint(
        f"[red bold] - WARNING: 'math_eval' reward uses `eval()`. Use --reward-content-type jaccard unless input is fully trusted.[/]"
    )
    rprint(
        "[dim] - [green]Using GRPO loss with group-based advantage normalization.[/]"
    )
    rprint(
        "[dim] - Checkpoints include training state, optimizer state, tokenizer, and weights.[/]"
    )
    rprint("[dim] - Supports resuming from last checkpoint via 'latest' symlink.[/]")
    rprint("-" * 30 + "\n")
    main()
