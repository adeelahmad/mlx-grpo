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

try:
    from llm_templates import (
        Formatter,
        Conversation,
        Content,
    )  # Import llm_templates components
except ImportError as e:
    print(
        f"Error importing llm_templates: {e}. Ensure llm_templates is installed and accessible.",
        file=sys.stderr,
    )
    sys.exit(1)


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
    # Import the specific model class if needed for type hinting or instantiation
    # from mlx_lm.models.llama import Model as LlamaModel # Example

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
    warnings = []
    if not isinstance(generated_text, str) or not generated_text.strip():
        return False, ["Input text is empty or not a string."]

    logging.debug(f"validating: generated_text:{generated_text}")
    tags = config.special_tokens
    think_start, think_end = tags["think_start"], tags["think_end"]
    answer_start, answer_end = tags["answer_start"], tags["answer_end"]
    required_tags = [ think_start, think_end, answer_start, answer_end]
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
                tag_positions[tag] = indices[0] # Use first occurrence if multiple
            else:
                tag_positions[tag] = indices[0]
        except Exception as e:
            warnings.append(f"Regex error finding {tag}: {e}")
            return False, warnings # Critical error finding tags

    if missing:
        warnings.append(f"Missing tags: {', '.join(missing)}")
    if multiple:
        # This might be acceptable depending on the use case, but strict format requires single occurrence
        warnings.append(f"Multiple occurrences of tags: {', '.join(multiple)}")

    # Proceed with order check even if multiple tags were found (using first occurrence)
    # If any tag was missing, return False
    if missing:
        return False, warnings

    # Check order of the first occurrences
    p0, p1, p2, p3 = (
        tag_positions[think_start],
        tag_positions[think_end],
        tag_positions[answer_start],
        tag_positions[answer_end],
    )
    if not (p0 < p1 < p2 < p3):
        warnings.append(f"Tag order incorrect: think_start={p0}, think_end={p1}, answer_start={p2}, answer_end={p3}")
        return False, warnings

    # Check for non-whitespace content between </thinking> and <answer>
    try:
        separator = generated_text[p1 + len(think_end) : p2]
        if separator.strip(): # Check if the stripped separator is non-empty
            warnings.append(f"Non-whitespace content found between thinking and answer blocks: '{separator[:50]}...'")
            # Depending on strictness, this could be a failure or just a warning
            # return False, warnings # Make it a failure
    except IndexError:
        # This shouldn't happen if tag order is correct, but handle defensively
        warnings.append("Error checking separator between blocks (IndexError).")
        return False, warnings

    # If all checks passed (or warnings were acceptable)
    return True, warnings # Return True even if there were 'multiple tag' warnings


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
                transient=True, # Progress bar disappears after completion
            )
            if RICH_AVAILABLE
            else None
        )
        try:
            # Estimate length for progress bar
            try:
                # Count lines efficiently
                with open(self.file_path, "rb") as f: # Open in binary for faster counting
                     total_lines = sum(1 for _ in f)
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
                        # Indeterminate progress bar if length couldn't be estimated
                        task_id = progress.add_task(
                            f"Loading {self.file_path.name}", total=None
                        )
                    iterable_f = progress.track(f, task_id=task_id) # Use track for auto-update

                for i, line in enumerate(iterable_f):
                    line_num = i + 1
                    line = line.strip()
                    if not line: # Skip empty lines
                        continue
                    try:
                        data = json.loads(line)
                        prompt = data.get("prompt")
                        completion = data.get("completion") # Check completion too

                        # Validate prompt and completion presence and type
                        if isinstance(prompt, str) and prompt and isinstance(completion, str) and completion:
                            self.prompts.append(prompt)
                            self.raw_data.append(data)  # Store original data
                            loaded_count += 1
                        else:
                            # Log which field was missing or invalid
                            if not isinstance(prompt, str) or not prompt:
                                 self.logger.debug(f"Line {line_num}: Skipping, missing/invalid 'prompt'.")
                            if not isinstance(completion, str) or not completion:
                                 self.logger.debug(f"Line {line_num}: Skipping, missing/invalid 'completion'.")
                            skipped += 1
                    except json.JSONDecodeError:
                        self.logger.warning(f"Line {line_num}: Skipping invalid JSON.")
                        skipped += 1
                    except Exception as e: # Catch other potential errors during processing
                        self.logger.warning(f"Line {line_num}: Error processing line: {e}")
                        skipped += 1
                    # No finally needed if using progress.track

                # Final update for progress bar if it was determinate
                # if progress and task_id is not None and total_lines is not None:
                #     progress.update(task_id, completed=total_lines) # Ensure it shows 100%

            self.logger.info(
                f"Loaded {loaded_count} valid samples (skipped {skipped}) from {self.file_path}"
            )
            if not self.prompts:
                # Changed from error to warning, as training might still proceed with 0 samples (though unlikely useful)
                self.logger.warning("No valid samples (prompt+completion) loaded! Training might fail.")

        except FileNotFoundError:
            self.logger.error(f"Dataset file not found during load: {self.file_path}")
            raise # Re-raise critical error
        except Exception as e:
            self.logger.error(
                f"Failed to load prompts from {self.file_path}: {e}", exc_info=True
            )
            # Depending on severity, might want to raise here too


    def __len__(self):
        return len(self.prompts)  # Length based on valid loaded prompts

    def get_raw_sample(self, idx):
        """Gets the raw data dictionary for a given index, handling wrap-around."""
        if not self.raw_data:
             raise IndexError("Dataset is empty.")
        return self.raw_data[idx % len(self.raw_data)]

    def __getitem__(self, idx):
        """Gets the prompt string for a given index, handling wrap-around."""
        if not self.prompts:
            raise IndexError("Dataset is empty.")
        return self.prompts[idx % len(self.prompts)]


def create_chat_text_format(
    prompt: str, final_completion: str, model_name: str = "llama3"
) -> str:
    """Creates the chat formatted string using llm_templates."""
    try:
        # Construct messages list in standard format expected by llm_templates
        # print(final_completion)
        # sys.exit(0)
        messages = [
            Content(role="user", content=prompt),
            Content(role="assistant", content=final_completion),
        ]
        # conversation = Conversation(model=model_name, messages=messages)
        # # Assuming default formatter works; adjust if specific template needed
        # formatter = Formatter()
        # # Render without adding the assistant prompt turn marker at the end
        # formatted_str = formatter.render(conversation, add_assistant_prompt=False)

        conversation = Conversation(model="llama3", messages=messages)
        conversation_str = Formatter().render(conversation, add_assistant_prompt=True)
        return "<|begin_of_text|>"+conversation_str + random.choice(["","\n<thinking>\n"])
    except Exception as e:
        logging.error(f"Error creating chat text format: {e}")
        # Return a basic fallback representation
        return f"User: {prompt.strip()}\nAssistant: {final_completion.strip()}"


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

        # Ensure dataset is not empty before proceeding
        if len(self.dataset) == 0:
            raise ValueError(f"Environment '{self.env_id}' cannot be initialized with an empty dataset.")

        self.data_indices = list(range(len(self.dataset)))
        random.shuffle(self.data_indices)
        self.data_iterator_idx = 0

        # State variables for the current step
        self.current_prompt_text: Optional[str] = None
        self.current_reference_completion: Optional[str] = None
        self.current_sample_data: Optional[Dict] = None

        self.logger.info(
            f"LLMEnv '{self.env_id}' initialized with {len(self.data_indices)} samples."
        )

    def _get_next_sample_data(self) -> Optional[Dict]:
        """Gets the next full sample data dictionary using the shuffled index iterator."""
        if not self.data_indices: # Check if indices list is empty
             self.logger.warning(f"Dataset '{self.env_id}' has no valid indices.")
             return None

        # Reset iterator and reshuffle if exhausted
        if self.data_iterator_idx >= len(self.data_indices):
            self.logger.debug(
                f"Dataset '{self.env_id}' iterator exhausted. Resetting & shuffling."
            )
            random.shuffle(self.data_indices)
            self.data_iterator_idx = 0

        # Get the next index from the shuffled list
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
                # This case should ideally not happen if dataset loading is correct
                self.logger.warning(
                    f"Invalid data content retrieved at internal index {current_internal_idx}. Attempting next sample."
                )
                # Recursively call to get the *next* valid sample
                return self._get_next_sample_data()
        except IndexError:
            # This indicates a mismatch between data_indices and dataset length
            self.logger.error(
                f"Index {current_internal_idx} out of bounds for raw_data (size {len(self.dataset.raw_data)}). Reshuffling."
            )
            # Attempt to recover by reshuffling and resetting (might infinite loop if dataset is truly broken)
            random.shuffle(self.data_indices)
            self.data_iterator_idx = 0
            return self._get_next_sample_data() # Retry after reshuffle
        except Exception as e:
            self.logger.error(
                f"Error fetching sample at internal index {current_internal_idx}: {e}", exc_info=True
            )
            # Attempt to recover by getting the next sample
            return self._get_next_sample_data()

    def reset(self) -> Optional[mx.array]:
        """Resets environment by loading a new prompt, returns tokenized prompt (state)."""
        attempts = 0
        # Set a reasonable max attempts based on dataset size to avoid infinite loops
        max_attempts = max(10, len(self.dataset) // 5 if len(self.dataset) > 0 else 10)

        while attempts < max_attempts:
            attempts += 1
            self.current_sample_data = self._get_next_sample_data()

            # If iterator fails to return data after reshuffling
            if self.current_sample_data is None:
                 self.logger.error(f"Failed to get next sample data after {attempts} attempts in reset.")
                 return None # Cannot proceed if no data can be fetched

            self.current_prompt_text = self.current_sample_data.get("prompt")
            self.current_reference_completion = self.current_sample_data.get("completion")

            # Check if required fields are present and valid
            if not self.current_prompt_text or not self.current_reference_completion:
                self.logger.debug(f"Reset attempt {attempts}: Skipping sample due to missing prompt/completion.")
                continue # Try next sample

            # Tokenize the prompt
            try:
                # Ensure prompt is string
                prompt_str = create_chat_text_format(self.current_prompt_text,"")
                tokenized_ids = self.tokenizer.encode(prompt_str)

                if not tokenized_ids: # Handle empty tokenization result
                     self.logger.warning(f"Reset attempt {attempts}: Skipping sample due to empty tokenization for prompt: '{prompt_str[:100]}...'")
                     continue

                # Truncate if necessary
                if len(tokenized_ids) > self.max_prompt_len:
                    tokenized_ids = tokenized_ids[: self.max_prompt_len]
                    self.logger.debug(f"Reset attempt {attempts}: Truncated prompt to {self.max_prompt_len} tokens.")

                # Convert to MLX array
                prompt_mx = mx.array(tokenized_ids, dtype=mx.int32).reshape(1, -1)
                mx.eval(prompt_mx) # Ensure array is materialized

                # Successfully reset, return the state
                self.logger.debug(f"Reset successful (attempt {attempts}). Prompt length: {prompt_mx.shape[1]}")
                return prompt_mx

            except Exception as e:
                self.logger.warning(f"Reset attempt {attempts}: Tokenization/processing failed: {e}", exc_info=True)
                # Clear state for the failed attempt before retrying
                self.current_prompt_text = None
                self.current_reference_completion = None
                self.current_sample_data = None
                # Continue to next attempt

        # If max attempts reached
        self.logger.error(f"Failed to reset environment '{self.env_id}' after {max_attempts} attempts.")
        return None

    def step(self, generated_text: str) -> Tuple[float, Dict[str, Any]]:
        """Calculates reward for the generated text based on the current reference completion."""
        if self.current_reference_completion is None:
            self.logger.error("step() called before successful reset or reference completion is missing.")
            min_reward = getattr(self.reward_config, "min_reward_clip", 0.0) # Use configured min reward
            return min_reward, {"error": "Environment not reset or reference missing"}

        try:
            # Validate the format of the generated text first
            is_valid, fmt_warnings = validate_text_format(
                generated_text, self.reward_config
            )
            reward_metrics = {"format_warnings": fmt_warnings} # Include warnings in metrics

            if not is_valid:
                self.logger.debug(
                    f"Generated text failed format validation: {fmt_warnings}"
                )
                # Assign minimum reward for invalid format
                reward_val = getattr(self.reward_config, "min_reward_clip", -1.0)
                reward_metrics["format_score"] = 0.0 # Explicitly set format score to 0
            else:
                # Calculate reward using the reward function if format is valid
                reward_val, calculated_metrics = self.reward_fn.calculate_reward(
                    generated_text=str(generated_text),
                    reference_text=str(self.current_reference_completion),
                )
                reward_metrics.update(calculated_metrics) # Merge metrics from reward function
                reward_metrics["format_score"] = 1.0 # Valid format

            # Ensure reward is float
            reward_val = float(reward_val)

            # Add generation format validity flag
            reward_metrics["generation_valid_format"] = 1.0 if is_valid else 0.0

            # Add per-sample bounds from dataset to info if they exist
            if self.current_sample_data:
                if "min_reward" in self.current_sample_data:
                    reward_metrics["sample_min_reward"] = self.current_sample_data["min_reward"]
                if "max_reward" in self.current_sample_data:
                    reward_metrics["sample_max_reward"] = self.current_sample_data["max_reward"]

            return reward_val, reward_metrics

        except Exception as e:
            self.logger.error(f"Reward calculation error during step: {e}", exc_info=True)
            min_reward = getattr(self.reward_config, "min_reward_clip", 0.0)
            # Return min reward and error info
            return min_reward, {"reward_error": str(e), "generation_valid_format": 0.0}

    def __len__(self):
        """Return the number of samples in the underlying dataset."""
        return len(self.dataset)


# === MLX Critic Network ===
class CriticNetwork(nn.Module):
    """Simple MLP Critic Network."""
    def __init__(self, input_dim: int, hidden_dim_factor: int = 2):
        super().__init__()
        hidden_dim = input_dim * hidden_dim_factor
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"CriticNetwork initialized: input={input_dim}, hidden={hidden_dim}")

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Forward pass for the critic network.
        Args:
            hidden_states: Input hidden states, shape (batch, seq_len, hidden_dim) or (batch, hidden_dim).
        Returns:
            Predicted value, shape (batch, 1).
        """
        try:
            # Pool if sequence dimension exists (handles different input shapes)
            if hidden_states.ndim == 3:
                # Mean pooling over the sequence length dimension (axis 1)
                pooled_state = mx.mean(hidden_states, axis=1)
            elif hidden_states.ndim == 2:
                # Assume already pooled or single state representation (batch, hidden_dim)
                pooled_state = hidden_states
            else:
                raise ValueError(
                    f"Unsupported input ndim for CriticNetwork: {hidden_states.ndim}, shape: {hidden_states.shape}"
                )

            # Check if pooling resulted in expected shape
            if pooled_state.ndim != 2:
                 raise ValueError(f"Pooled state has unexpected ndim: {pooled_state.ndim}")

            # Pass through linear layers with GELU activation
            x = nn.gelu(self.fc1(pooled_state))
            value = self.fc2(x) # Output shape: (batch, 1)
            return value

        except Exception as e:
            self.logger.error(f"Critic forward pass failed: {e}", exc_info=True)
            # Return zeros with the expected batch dimension and output dimension 1
            batch_dim = hidden_states.shape[0]
            return mx.zeros((batch_dim, 1))


# === MLX Rollout Buffer ===
@dataclass
class RolloutBuffer:
    """Stores rollout data for PPO updates."""
    prompts: List[mx.array] = field(default_factory=list)
    generations: List[mx.array] = field(default_factory=list)
    actions_text: List[str] = field(default_factory=list)
    log_probs: List[mx.array] = field(default_factory=list) # Log prob of the generated sequence
    rewards: List[float] = field(default_factory=list)
    values: List[mx.array] = field(default_factory=list) # Value estimate of the prompt state
    dones: List[bool] = field(default_factory=list) # Typically True for each generation
    advantages: Optional[mx.array] = None
    returns: Optional[mx.array] = None
    logger = logging.getLogger(__name__)

    def add(
        self,
        prompt: mx.array,
        gen_ids: mx.array,
        action_text: str,
        log_prob: mx.array, # Log prob of the full generated sequence
        reward: float,
        done: bool, # Should be True if each generation is one step
        value: mx.array, # Value estimate of the prompt state
    ):
        """Adds one step of experience to the buffer."""
        # Validate inputs
        if not all(isinstance(x, mx.array) for x in [prompt, gen_ids, log_prob, value]):
            self.logger.error(f"Invalid type passed to RolloutBuffer.add. Expected mx.array, got types: "
                              f"prompt={type(prompt)}, gen_ids={type(gen_ids)}, log_prob={type(log_prob)}, value={type(value)}")
            return
        if not isinstance(reward, (float, int)):
             self.logger.error(f"Invalid reward type: {type(reward)}")
             return
        if not isinstance(done, bool):
             self.logger.error(f"Invalid done type: {type(done)}")
             return

        # Squeeze batch dimension (usually 1) before appending
        self.prompts.append(prompt.squeeze(0))
        self.generations.append(gen_ids.squeeze(0))
        self.actions_text.append(action_text)
        self.log_probs.append(log_prob.squeeze()) # Should be scalar log prob for sequence
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.squeeze()) # Should be scalar value estimate

    def _pad_sequences(
        self, sequences: List[mx.array], pad_value: int
    ) -> Tuple[mx.array, mx.array]:
        """Pads a list of sequences to the maximum length in the list."""
        if not sequences: # Handle empty list
            # Return empty arrays with appropriate dimensions and types
            return mx.array([[]], dtype=mx.int32), mx.array([[]], dtype=mx.bool_)

        # Ensure all elements are mx.arrays
        if not all(isinstance(s, mx.array) for s in sequences):
             raise TypeError("All elements in sequences must be mx.array")

        lengths = [s.size for s in sequences]
        if not lengths: # Should be caught by `if not sequences` but double-check
             return mx.array([[]], dtype=mx.int32), mx.array([[]], dtype=mx.bool_)

        max_len = max(lengths)
        padded_seqs, masks = [], []

        for i, s in enumerate(sequences):
            pad_len = max_len - lengths[i]
            if pad_len < 0: # Should not happen if max_len is calculated correctly
                 self.logger.error(f"Negative padding length encountered: {pad_len}")
                 pad_len = 0

            # Create padding array
            padding = mx.full((pad_len,), pad_value, dtype=s.dtype)
            # Create mask (1s for real tokens, 0s for padding)
            mask = mx.concatenate(
                [mx.ones(lengths[i], dtype=mx.bool_), mx.zeros(pad_len, dtype=mx.bool_)]
            )
            # Concatenate sequence and padding
            padded_seq = mx.concatenate([s, padding])

            padded_seqs.append(padded_seq)
            masks.append(mask)

        # Stack padded sequences and masks into batch tensors
        return mx.stack(padded_seqs, axis=0), mx.stack(masks, axis=0)

    def compute_advantages_and_returns(
        self, last_value: mx.array, gamma: float, gae_lambda: float
    ):
        """Computes Generalized Advantage Estimation (GAE) and returns."""
        num_steps = len(self.rewards)
        if num_steps == 0:
            self.logger.warning("Cannot compute GAE: Rollout buffer is empty.")
            self.advantages = mx.zeros((0,)) # Empty array with correct shape
            self.returns = mx.zeros((0,))
            return

        # Ensure last_value is a scalar float
        try:
            last_value_scalar = last_value.item()
        except Exception as e:
            self.logger.error(f"Could not convert last_value ({last_value}, shape {last_value.shape}) to scalar: {e}. Using 0.0.")
            last_value_scalar = 0.0

        # Convert buffer lists to numpy arrays for calculation ease
        # Add the last_value to the end of values for GAE calculation
        values_np = np.array(
            [v.item() for v in self.values] + [last_value_scalar], dtype=np.float32
        )
        rewards_np = np.array(self.rewards, dtype=np.float32)
        # Assuming dones are always True for this setup, otherwise need dones_np
        # dones_np = np.array(self.dones + [True], dtype=np.bool_) # Add final done state

        advantages_np = np.zeros_like(rewards_np)
        last_gae_lam = 0.0

        # Iterate backwards through the rollout steps
        for t in reversed(range(num_steps)):
            # Calculate delta: TD error
            # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            # Assumes done=True, so next state value contribution is just gamma * V(s_{t+1})
            # If done=False, it would be gamma * (1 - dones_np[t+1]) * values_np[t+1]
            delta = rewards_np[t] + gamma * values_np[t + 1] - values_np[t]

            # Calculate GAE advantage for step t
            # advantage_t = delta_t + gamma * lambda * advantage_{t+1}
            # Assumes done=True, so no (1 - dones_np[t+1]) term multiplying last_gae_lam
            last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
            advantages_np[t] = last_gae_lam

        # Calculate returns: Returns = Advantages + Values
        returns_np = advantages_np + values_np[:-1] # Exclude the last_value added earlier

        # Convert back to MLX arrays
        self.advantages = mx.array(advantages_np)
        self.returns = mx.array(returns_np)

        # Evaluate to ensure computation happens
        mx.eval(self.advantages, self.returns)
        self.logger.debug(
            f"Computed GAE: Adv shape {self.advantages.shape}, Ret shape {self.returns.shape}. "
            f"Avg Adv: {mx.mean(self.advantages).item():.3f}, Avg Ret: {mx.mean(self.returns).item():.3f}"
        )

    def get_batch_generator(
        self, batch_size: int, tokenizer: Any
    ) -> Generator[Optional[Dict[str, mx.array]], None, None]:
        """Yields batches of experiences, padding sequences as needed."""
        num_samples = len(self.prompts)
        if num_samples == 0 or self.advantages is None or self.returns is None:
            self.logger.warning(
                "Cannot generate batches: buffer empty or advantages/returns not computed."
            )
            # yield None # Don't yield None, just return to stop iteration
            return

        indices = np.random.permutation(num_samples)

        # Determine pad token ID
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
             pad_token_id = getattr(tokenizer, "eos_token_id", None)
             if pad_token_id is None:
                  self.logger.error("Cannot pad sequences: Tokenizer lacks pad_token_id and eos_token_id.")
                  return # Stop iteration if padding isn't possible

        # Iterate through shuffled indices in batches
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            if len(batch_indices) == 0: # Should not happen with range step, but check
                continue

            try:
                # Collect data for the batch
                batch_prompts_list = [self.prompts[i] for i in batch_indices]
                batch_gens_list = [self.generations[i] for i in batch_indices]

                # Pad sequences and create masks
                prompts_padded, prompt_mask = self._pad_sequences(
                    batch_prompts_list, pad_token_id
                )
                generations_padded, gen_mask = self._pad_sequences(
                    batch_gens_list, pad_token_id
                )

                # Stack other data for the batch
                batch_log_probs_old = mx.stack([self.log_probs[i] for i in batch_indices])
                batch_advantages = self.advantages[batch_indices]
                batch_returns = self.returns[batch_indices]
                batch_values_old = mx.stack([self.values[i] for i in batch_indices])

                # Yield the complete batch dictionary
                yield {
                    "prompts_padded": prompts_padded,
                    "prompt_mask": prompt_mask, # Mask for prompts (optional use)
                    "generations_padded": generations_padded,
                    "generation_mask": gen_mask, # Crucial mask for generated part
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
                # Optionally yield None or just continue to next batch attempt
                # yield None
                continue # Skip this batch on error

    def clear(self):
        """Clears all data from the buffer."""
        self.prompts.clear()
        self.generations.clear()
        self.actions_text.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None
        gc.collect() # Suggest garbage collection after clearing large lists
        self.logger.debug("Rollout buffer cleared.")

    def __len__(self) -> int:
        """Return the number of steps collected in the buffer."""
        # Use rewards list length as the indicator of collected steps
        return len(self.rewards)


# === Metrics Logger ===
class MetricsLogger:
    """Logs metrics to a JSONL file."""
    def __init__(self, log_file: Union[str, Path]):
        self.log_path = Path(log_file)
        self.logger = logging.getLogger(__name__) # Use module logger
        # Ensure parent directory exists
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Metrics will be logged to: {self.log_path.resolve()}")
        except OSError as e:
            self.logger.error(f"Failed to create directory for metrics log file {self.log_path}: {e}")
            self.log_path = None # Disable logging if directory fails

    def log(self, data: Dict[str, Any]):
        """Logs a dictionary of metrics to the file."""
        if self.log_path is None:
            self.logger.warning("Metrics logging disabled due to initialization error.")
            return

        try:
            # Ensure data is JSON serializable
            serializable_data = {}
            for k, v in data.items():
                if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                    serializable_data[k] = v
                elif isinstance(v, (mx.array, np.number, np.bool_)):
                    # Convert numpy/mlx scalars to Python types
                    try:
                         serializable_data[k] = v.item()
                    except Exception as item_err:
                         self.logger.debug(f"Could not convert metric '{k}' value {v} to item: {item_err}")
                         serializable_data[k] = str(v) # Fallback to string
                elif isinstance(v, np.ndarray):
                    # Convert numpy arrays to lists
                    serializable_data[k] = v.tolist()
                elif isinstance(v, Path):
                     serializable_data[k] = str(v) # Convert Path objects to strings
                else:
                    # Fallback: convert other types to string
                    serializable_data[k] = str(v)
                    self.logger.debug(f"Converted metric '{k}' of type {type(v)} to string for logging.")

            # Append JSON line to the log file
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(serializable_data) + "\n")

        except Exception as e:
            # Log error without logging the potentially large data dict directly unless debugging
            self.logger.error(f"Failed to write metrics to {self.log_path}: {e}", exc_info=True)
            # Optionally log keys for debugging: self.logger.error(f"Failed data keys: {list(data.keys())}")


# === PPO Agent (Functional Updates, Enhanced & Fixed) ===
class PPOAgent:
    """PPO Agent implementation using MLX functional API."""
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
        ppo_epochs: int,
        ppo_batch_size: int,
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
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.grad_clip_norm = grad_clip_norm
        self.logger = logger

        # Determine hidden size robustly
        try:
            if hasattr(model, 'args') and hasattr(model.args, 'hidden_size'):
                 self.hidden_size = model.args.hidden_size
            elif hasattr(model, 'config') and isinstance(model.config, dict) and 'hidden_size' in model.config:
                 self.hidden_size = model.config['hidden_size']
            elif hasattr(model, 'model') and hasattr(model.model, 'config') and isinstance(model.model.config, dict) and 'hidden_size' in model.model.config:
                 self.hidden_size = model.model.config['hidden_size']
            else:
                 params = model.parameters()
                 found_dim = None
                 for p in tree_flatten(params)[0]:
                     if isinstance(p, mx.array) and p.ndim > 1: # Need at least 2 dims
                         found_dim = p.shape[-1]
                         break
                 if found_dim:
                     self.hidden_size = found_dim
                     self.logger.warning(f"Guessed hidden_size={self.hidden_size} from parameter shapes.")
                 else:
                     raise ValueError("Cannot determine actor hidden size from args, config, or parameters.")
        except Exception as e: # Catch broader exceptions during init
            raise ValueError(f"Failed to determine actor hidden size during PPOAgent initialization: {e}")

        # Store parameters and optimizer states
        # These will be updated during training
        self.actor_params = tree_map(lambda p: p, self.actor.parameters())
        self.critic_params = tree_map(lambda p: p, self.critic.parameters())
        # Initialize optimizer states based on the parameters they will optimize
        self.actor_opt_state = self.actor_optimizer.state # Should be initialized by caller or load_checkpoint
        self.critic_opt_state = self.critic_optimizer.state # Should be initialized by caller or load_checkpoint

        # Evaluate initial state
        mx.eval(
            self.actor_params,
            self.critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
        )

        # Log parameter counts
        param_count = sum(p.size for p in tree_flatten(self.actor_params)[0] if isinstance(p, mx.array))
        trainable_param_count = sum(p.size for p in tree_flatten(self.actor.trainable_parameters())[0] if isinstance(p, mx.array))
        self.logger.info(
            f"Agent Initialized. Actor Total Params: {param_count / 1e6:.2f}M, Trainable: {trainable_param_count / 1e6:.3f}M"
        )
        critic_param_count = sum(p.size for p in tree_flatten(self.critic_params)[0] if isinstance(p, mx.array))
        self.logger.info(f"Critic Total Params: {critic_param_count / 1e3:.1f}K")


    def _get_state_embedding(
        self, actor_model: nn.Module, prompt_ids: mx.array
    ) -> mx.array:
        """Gets the embedding from the actor model for the critic."""
        try:
            # Determine if the actor model has a separate 'model' attribute
            model_to_call = actor_model.model if hasattr(actor_model, "model") and isinstance(actor_model.model, nn.Module) else actor_model

            # Call the appropriate model part
            outputs = model_to_call(prompt_ids)

            # Extract hidden states (handle tuple output)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

            # --- Value Head Input Strategy ---
            # Option 1: Mean pooling (current implementation)
            # pooled_state = mx.mean(hidden_states, axis=1)

            # Option 2: Last token hidden state
            # Ensure hidden_states has sequence length dimension
            if hidden_states.ndim == 3 and hidden_states.shape[1] > 0:
                 pooled_state = hidden_states[:, -1, :] # Shape: (batch, hidden_dim)
            elif hidden_states.ndim == 2: # If input was already 2D
                 pooled_state = hidden_states
            else:
                 raise ValueError(f"Unexpected hidden_states shape for embedding: {hidden_states.shape}")

            return pooled_state

        except Exception as e:
            self.logger.error(f"Failed state embedding calculation: {e}", exc_info=True)
            batch_dim = prompt_ids.shape[0]
            return mx.zeros((batch_dim, self.hidden_size)) # Return zeros on error

    def get_value(self, prompt_ids: mx.array) -> mx.array:
        """Gets the value prediction from the critic for given prompt_ids."""
        self.actor.eval()
        self.critic.eval()
        value = mx.zeros((prompt_ids.shape[0],)) # Default value shape (batch,)
        try:
            if prompt_ids.ndim == 1:
                prompt_ids = prompt_ids[None, :] # Add batch dimension

            # Get embedding using the current actor instance
            state_embedding = self._get_state_embedding(self.actor, prompt_ids)

            # Get value using the current critic instance
            value_output = self.critic(state_embedding) # Shape (batch, 1)
            value = value_output.squeeze(-1) # Shape (batch,)

            # Stop gradients as this is for rollouts/GAE, not critic training directly
            value = mx.stop_gradient(value)
            mx.eval(value) # Ensure computation

        except Exception as e:
            self.logger.error(f"Error getting value: {e}", exc_info=True)
            value = mx.zeros((prompt_ids.shape[0],)) # Fallback

        return value

    def _sample_token(self, logits: mx.array, temp: float) -> Tuple[mx.array, mx.array]:
        """Samples a token from logits and calculates its log probability."""
        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            # Ensure logits are float32 for categorical sampling stability
            token = mx.random.categorical(logits.astype(mx.float32) * (1 / temp))

        log_probs_all = nn.log_softmax(logits.astype(mx.float32), axis=-1) # Use float32 for log_softmax
        log_prob_sampled = mx.take_along_axis(
            log_probs_all, token[:, None], axis=-1
        ).squeeze(-1)

        return token, log_prob_sampled

    def generate_action(
        self, prompt_ids: mx.array, temp: float = 0.7
    ) -> Tuple[str, mx.array, mx.array, mx.array, mx.array]:
        """Generates an action (sequence of tokens) from the actor, using KV caching."""
        self.actor.eval()
        self.critic.eval()

        generated_tokens_list = []
        log_probs_list = []
        current_ids = prompt_ids
        batch_size = prompt_ids.shape[0]

        if batch_size != 1:
            self.logger.warning(f"generate_action expects batch size 1, got {batch_size}. Using first element.")
            current_ids = prompt_ids[0:1]

        initial_value = self.get_value(current_ids)

        # --- KV Caching Implementation ---
        kv_cache = None # Initialize cache
        input_ids = current_ids # Start with the prompt

        try:
            for i in range(self.max_gen_len):
                # Model forward pass with cache
                # Pass only the last token if cache is available
                if kv_cache is not None:
                    model_input = input_ids[:, -1:] # Only the new token
                else:
                    model_input = input_ids # Full prompt initially

                # Call the actor model (assumes it handles cache argument)
                outputs = self.actor(model_input, cache=kv_cache)

                # Extract logits and update cache
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    logits = outputs[0]
                    kv_cache = outputs[1] # Update cache for next step
                else:
                    # Model might not return cache if not implemented or at first step
                    logits = outputs
                    # kv_cache remains None or its previous state

                # Get logits for the *next* token prediction (usually the last time step)
                next_token_logits = logits[:, -1, :]

                # Sample the next token
                token, log_prob = self._sample_token(next_token_logits, temp)

                generated_tokens_list.append(token)
                log_probs_list.append(log_prob)

                # Update input_ids for the *next* iteration's input
                input_ids = token[:, None] # Next input is just the sampled token

                # Check for EOS token
                if self.tokenizer.eos_token_id is not None and mx.all(token == self.tokenizer.eos_token_id):
                    break

            # --- Post-generation processing ---
            if not generated_tokens_list:
                generation_ids = mx.array([[]], dtype=mx.int32)
                sequence_log_prob = mx.array(0.0)
            else:
                generation_ids = mx.concatenate(generated_tokens_list, axis=0)[None, :]
                sequence_log_prob = mx.sum(mx.stack(log_probs_list))

            full_sequence_ids = mx.concatenate([current_ids, generation_ids], axis=1)
            generated_text = self.tokenizer.decode(generation_ids[0].tolist())

            mx.eval(sequence_log_prob, initial_value, generation_ids, full_sequence_ids)

        except Exception as e:
            self.logger.error(f"Error during generate_action with KV caching: {e}", exc_info=True)
            generated_text = "[Generation Error]"
            sequence_log_prob = mx.array(0.0)
            initial_value = mx.zeros_like(initial_value) # Match shape
            generation_ids = mx.array([[]], dtype=mx.int32)
            full_sequence_ids = current_ids

        # Restore train mode
        self.actor.train()
        self.critic.train()

        generated_text = "<thinking>\n"+ generated_text

        return (
            generated_text,
            sequence_log_prob,
            initial_value,
            generation_ids,
            full_sequence_ids,
        )


    def evaluate_actions(
        self,
        actor_model: nn.Module, # Pass the model instance
        critic_model: nn.Module, # Pass the model instance
        prompts_padded: mx.array,
        generations_padded: mx.array,
        generation_mask: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Evaluate actions (generations) using specific model instances.
        Called within the loss function computation where gradients are tracked.
        """
        # Concatenate prompt and generation for full sequence input
        full_sequence = mx.concatenate([prompts_padded, generations_padded], axis=1)
        prompt_len = prompts_padded.shape[1]
        gen_len = generations_padded.shape[1]

        # --- Actor Evaluation (Logits, LogProbs, Entropy) ---
        # Get logits from the actor model for the full sequence
        # No KV caching needed here as we need gradients through the full sequence
        outputs = actor_model(full_sequence)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs # Shape: (batch, seq_len, vocab_size)

        # Select logits corresponding to the *actions* (generated tokens)
        # Logits at index t-1 predict token at index t.
        # We need logits from prompt_len-1 up to seq_len-2 to predict tokens from prompt_len to seq_len-1.
        action_logits = logits[:, prompt_len - 1 : -1, :]

        # Validate shape consistency (should be batch, gen_len, vocab_size)
        if action_logits.shape[1] != gen_len:
             # This check might be overly strict if generation stops early via EOS.
             # The generation_mask should handle this correctly later.
             self.logger.warning(
                f"Logit/Generation length mismatch in evaluate_actions: "
                f"Action Logits shape[1] {action_logits.shape[1]} vs Gen length {gen_len}. "
                f"Using mask for calculations."
             )
             # Adjust slicing if necessary, though mask should dominate.
             # Example: If action_logits is shorter due to EOS, padding handles the rest.

        # Ensure calculations use float32 for stability
        action_logits = action_logits.astype(mx.float32)

        # Calculate log probabilities and probabilities of the distribution
        log_probs_dist = nn.log_softmax(action_logits, axis=-1)
        probs_dist = mx.softmax(action_logits, axis=-1)

        # --- Entropy Calculation ---
        # entropy = -sum(p * log(p)) over vocab dimension
        # Apply generation mask to zero out entropy for padded tokens
        entropy_per_token = (
            -mx.sum(probs_dist * log_probs_dist, axis=-1) # Shape: (batch, gen_len)
        ) * generation_mask # Apply mask element-wise
        # Calculate mean entropy per sequence (average over non-masked tokens)
        sum_entropy = mx.sum(entropy_per_token, axis=1) # Sum entropy over sequence length
        num_valid_tokens = mx.sum(generation_mask, axis=1) # Count non-padded tokens
        mean_sequence_entropy = sum_entropy / mx.maximum(num_valid_tokens, 1) # Avoid division by zero

        # --- Action Log Probability Calculation ---
        # Get log probability of the *actual* generated tokens under the current policy
        action_log_probs = mx.take_along_axis(
            log_probs_dist, generations_padded[..., None], axis=-1 # generations_padded needs extra dim
        ).squeeze(-1) # Shape: (batch, gen_len)

        # Apply mask and sum to get total log probability for each sequence
        masked_action_log_probs = action_log_probs * generation_mask
        sequence_log_probs = mx.sum(masked_action_log_probs, axis=1) # Shape: (batch,)

        # --- Critic Evaluation ---
        # Get state embedding using the *same actor model instance*
        # Use the prompt as input to the value function
        state_embeddings = self._get_state_embedding(actor_model, prompts_padded)

        # Get value prediction from the critic model
        values = critic_model(state_embeddings).squeeze(-1) # Shape: (batch,)

        # Evaluate results (optional, mainly for debugging or ensuring computation)
        # mx.eval(sequence_log_probs, values, mean_sequence_entropy)

        return sequence_log_probs, values, mean_sequence_entropy


    # ================================================================
    # === Core PPO Update Logic Implementation (MLX - Functional) ===
    # ================================================================
    def _compute_ppo_losses_and_grads_functional(
        self, batch: Dict[str, mx.array], actor_trainable_params: Dict, critic_params: Dict
    ) -> Tuple[Dict, Dict, Dict[str, float]]:
        """
        Computes PPO losses and gradients for one batch using functional calls.
        Takes trainable actor parameters and full critic parameters.
        Returns actor_grads (for trainable params), critic_grads, loss_metrics.
        """

        # --- Actor Loss Function (operates on trainable parameters) ---
        def actor_loss_fn(current_actor_trainable_params):
            # Update a temporary *copy* or the main actor instance with current trainable params
            # Using the main instance is okay if state is restored later, but copying is safer conceptually.
            # However, copying large models is inefficient. Let's update the main instance temporarily.
            # NOTE: This assumes self.actor holds the non-trainable params correctly.
            original_actor_params = self.actor.parameters() # Store original state if needed
            self.actor.update(tree_unflatten(list(current_actor_trainable_params.items())))

            # Update critic temporarily (though its params aren't changing in this grad calc)
            original_critic_params = self.critic.parameters()
            self.critic.update(tree_unflatten(list(critic_params.items())))

            # Evaluate actions using the temporarily updated models
            new_log_probs, _, entropy = self.evaluate_actions(
                self.actor, # Pass the updated model instance
                self.critic, # Pass the updated model instance
                batch["prompts_padded"],
                batch["generations_padded"],
                batch["generation_mask"],
            )

            # Restore original parameters after evaluation if necessary (optional, depends on structure)
            # self.actor.update(original_actor_params)
            # self.critic.update(original_critic_params)

            log_probs_old = batch["log_probs_old"]
            advantages = batch["advantages"]
            new_log_probs = new_log_probs.reshape(log_probs_old.shape)
            entropy = entropy.reshape(advantages.shape) # Mean entropy per sequence

            # Normalize advantages per-batch
            # Stop gradient flow through mean/std calculation
            adv_mean = mx.stop_gradient(mx.mean(advantages))
            adv_std = mx.stop_gradient(mx.std(advantages) + 1e-8)
            norm_advantages = (advantages - adv_mean) / adv_std

            # PPO Clipped Surrogate Objective
            log_ratio = new_log_probs - log_probs_old # Log prob difference
            ratio = mx.exp(log_ratio)
            surr1 = ratio * norm_advantages
            surr2 = mx.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * norm_advantages
            actor_clip_loss = -mx.mean(mx.minimum(surr1, surr2)) # Maximize objective -> Minimize negative

            # Entropy Bonus (maximize entropy -> minimize negative entropy)
            entropy_loss = -mx.mean(entropy)

            # Total Actor Loss
            total_actor_loss = actor_clip_loss + self.entropy_coef * entropy_loss

            # Metrics for logging (no gradients needed)
            kl_approx = mx.mean(log_ratio).item()
            ratio_mean = mx.mean(ratio).item()
            clip_fraction = mx.mean((mx.abs(ratio - 1.0) > self.clip_epsilon).astype(mx.float32)).item()

            return total_actor_loss, (
                actor_clip_loss,
                entropy_loss,
                kl_approx,
                ratio_mean,
                clip_fraction, # Add clip fraction metric
            )

        # --- Critic Loss Function (operates on critic parameters) ---
        def critic_loss_fn(current_critic_params):
             # Update temporary models
             temp_critic = self.critic
             temp_critic.update(tree_unflatten(list(current_critic_params.items())))

             # Actor parameters are fixed (use the non-trainable base + trainable from outer scope)
             temp_actor = self.actor
             # Combine non-trainable base with current trainable params for evaluation
             # This assumes actor_trainable_params is from the outer scope and fixed here.
             full_actor_params_for_eval = tree_unflatten(list(actor_trainable_params.items()))
             temp_actor.update(full_actor_params_for_eval)

             # Evaluate actions to get new value predictions
             _, new_values, _ = self.evaluate_actions(
                temp_actor,
                temp_critic,
                batch["prompts_padded"],
                batch["generations_padded"],
                batch["generation_mask"],
             )
             returns = batch["returns"]
             new_values = new_values.reshape(returns.shape)

             # Value Loss (Mean Squared Error) - L2 Loss
             value_loss = mx.mean(mx.square(new_values - returns))
             # Optional: Value Clipping (compare new_values with clipped old_values)
             # values_old = batch["values_old"].reshape(returns.shape)
             # values_clipped = values_old + mx.clip(new_values - values_old, -self.clip_epsilon, self.clip_epsilon)
             # vf_loss_clipped = mx.mean(mx.square(values_clipped - returns))
             # value_loss = mx.maximum(value_loss, vf_loss_clipped)

             total_critic_loss = self.value_loss_coef * value_loss
             return total_critic_loss, value_loss

        # --- Compute Gradients ---
        grad_actor_fn = mx.value_and_grad(actor_loss_fn, argnums=0)
        grad_critic_fn = mx.value_and_grad(critic_loss_fn, argnums=0)

        # Calculate actor loss, metrics, and gradients (only for trainable params)
        (
            actor_total_loss_val,
            (actor_clip_loss_val, entropy_loss_val, kl_approx, ratio_mean, clip_fraction),
        ), actor_grads = grad_actor_fn(actor_trainable_params) # Pass only trainable params

        # Calculate critic loss and gradients
        (critic_total_loss_val, value_loss_val), critic_grads = grad_critic_fn(critic_params)

        # --- Gradient Clipping ---
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            # Clip gradients by norm for stability
            actor_grads = optim.clip_grad_norm(actor_grads, self.grad_clip_norm)
            critic_grads = optim.clip_grad_norm(critic_grads, self.grad_clip_norm)

        # --- Prepare Loss Metrics Dictionary ---
        losses = {
            "actor_clip_loss": actor_clip_loss_val.item(),
            "entropy_loss": entropy_loss_val.item(),
            "actor_total_loss": actor_total_loss_val.item(),
            "value_loss": value_loss_val.item(),
            "critic_loss": critic_total_loss_val.item(), # Scaled loss
            "total_ppo_loss": actor_total_loss_val.item() + critic_total_loss_val.item(),
            "kl_approx": kl_approx, # Estimated KL divergence
            "ratio_mean": ratio_mean, # Average importance sampling ratio
            "clip_fraction": clip_fraction, # Fraction of samples where clipping occurred
        }

        # Return gradients for trainable actor params and all critic params
        return actor_grads, critic_grads, losses


    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Performs PPO update epochs using data in the rollout buffer."""
        if len(buffer) == 0 or buffer.advantages is None or buffer.returns is None:
            self.logger.error("Cannot update: Rollout buffer empty or advantages/returns not computed.")
            return {} # Return empty dict if no update performed

        # Accumulate losses over all batches and epochs
        total_losses_accum = {
            "actor_clip_loss": 0.0, "entropy_loss": 0.0, "actor_total_loss": 0.0,
            "value_loss": 0.0, "critic_loss": 0.0, "total_ppo_loss": 0.0,
            "kl_approx": 0.0, "ratio_mean": 0.0, "clip_fraction": 0.0,
        }
        total_batches_processed = 0

        self.actor.train() # Set models to training mode
        self.critic.train()
        self.logger.info(
            f"Starting PPO update phase ({self.ppo_epochs} epochs, batch size {self.ppo_batch_size}, buffer size {len(buffer)})..."
        )

        # PPO Update Loop (multiple epochs over the same rollout data)
        for ppo_epoch in range(self.ppo_epochs):
            batch_generator = buffer.get_batch_generator(self.ppo_batch_size, self.tokenizer)
            epoch_batches = 0
            for batch in batch_generator:
                if not batch: # Skip if batch generation failed
                    self.logger.warning(f"Skipping empty batch in PPO epoch {ppo_epoch+1}")
                    continue

                # Get current parameters and optimizer states
                # Pass only *trainable* actor parameters to the loss function gradient calculation
                current_actor_trainable_params = self.actor.trainable_parameters()
                current_critic_params = self.critic_params # Full critic params
                current_actor_opt_state = self.actor_opt_state
                current_critic_opt_state = self.critic_opt_state

                try:
                    # Compute Gradients and Losses
                    actor_grads, critic_grads, losses = self._compute_ppo_losses_and_grads_functional(
                        batch, current_actor_trainable_params, current_critic_params
                    )

                    # Apply Gradients using optimizers
                    # Actor optimizer updates only the trainable parameters
                    new_actor_opt_state, updated_actor_trainable_params = self.actor_optimizer.apply_gradients(
                        actor_grads, current_actor_trainable_params, current_actor_opt_state
                    )
                    new_critic_opt_state, updated_critic_params = self.critic_optimizer.apply_gradients(
                        critic_grads, current_critic_params, current_critic_opt_state
                    )

                    # Update the stateful models and agent's stored state
                    self.actor.update(tree_unflatten(list(updated_actor_trainable_params.items())))
                    self.critic.update(tree_unflatten(list(updated_critic_params.items())))

                    # Update stored parameters and optimizer states in the agent
                    self.actor_params = self.actor.parameters() # Update full params potentially
                    self.critic_params = updated_critic_params
                    self.actor_opt_state = new_actor_opt_state
                    self.critic_opt_state = new_critic_opt_state
                    # Ensure optimizer instances also hold the latest state
                    self.actor_optimizer.state = new_actor_opt_state
                    self.critic_optimizer.state = new_critic_opt_state

                    # Evaluate changes to ensure computation and synchronization
                    mx.eval(
                        updated_actor_trainable_params, # Only need to eval updated parts
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
                        f"Error during PPO batch update {epoch_batches+1}/{len(buffer)//self.ppo_batch_size + 1} "
                        f"in epoch {ppo_epoch+1}: {e}",
                        exc_info=True,
                    )
                    # Decide whether to break epoch or continue
                    # break # Example: break epoch on error

            self.logger.debug(
                f"  PPO Epoch {ppo_epoch+1}/{self.ppo_epochs} finished ({epoch_batches} batches processed)."
            )
            # Check for shutdown request between epochs
            if shutdown_requested:
                 self.logger.warning(f"Shutdown requested during PPO epoch {ppo_epoch+1}. Stopping update phase.")
                 break

        # Calculate average losses over all processed batches across all epochs
        avg_losses = (
            {k: v / total_batches_processed for k, v in total_losses_accum.items()}
            if total_batches_processed > 0
            else total_losses_accum # Avoid division by zero if no batches processed
        )
        if total_batches_processed > 0:
             self.logger.info(
                f"PPO Update finished. Avg Losses over {total_batches_processed} batches: "
                f"{ {k: f'{v:.4f}' for k,v in avg_losses.items()} }"
             )
        else:
             self.logger.warning("PPO Update finished, but no batches were processed.")

        return avg_losses


# === Checkpoint and Resume Logic ===
def save_checkpoint(
    agent: PPOAgent, output_dir: Path, global_step: int, num_updates: int
):
    """Saves model, critic, optimizers, and training state."""
    save_path = output_dir / f"checkpoint_update_{num_updates}_step_{global_step}"
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}...")
        save_time = time.time()

        # Save actor weights using mlx_lm utility (handles sharding, adapters)
        # Saves all parameters (trainable and non-trainable) by default
        save_weights(str(save_path), agent.actor.parameters())

        # Save critic weights (simple safetensors save)
        mx.save_safetensors(str(save_path / "critic.safetensors"), agent.critic_params)

        # Save optimizer states (convert tree to dict for safetensors)
        actor_opt_state_save = dict(tree_flatten(agent.actor_opt_state))
        critic_opt_state_save = dict(tree_flatten(agent.critic_opt_state))
        if actor_opt_state_save:
             mx.save_safetensors(str(save_path / "actor_optimizer.safetensors"), actor_opt_state_save)
        if critic_opt_state_save:
             mx.save_safetensors(str(save_path / "critic_optimizer.safetensors"), critic_opt_state_save)

        # Save training state (step, update, random states)
        state = {
            "global_step": global_step,
            "num_updates": num_updates,
            "np_random_state": np.random.get_state(),
            "random_state": random.getstate(),
            "mx_random_state": mx.random.get_state().tolist(),
        }
        with open(save_path / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4, default=lambda o: "<not serializable>")

        # Save tokenizer configuration using its save_pretrained method
        tok_save_path = save_path / "tokenizer"
        tok_save_path.mkdir(exist_ok=True)
        tokenizer_to_save = agent.tokenizer.processor if hasattr(agent.tokenizer, "processor") and hasattr(agent.tokenizer.processor, "save_pretrained") else agent.tokenizer
        if hasattr(tokenizer_to_save, "save_pretrained"):
            try:
                 tokenizer_to_save.save_pretrained(str(tok_save_path))
            except Exception as e:
                 logger.error(f"Failed to save tokenizer using save_pretrained: {e}")
        else:
            logger.warning("Could not save tokenizer automatically (no save_pretrained method).")

        logger.info(f"Checkpoint saved successfully ({time.time()-save_time:.2f}s).")

        # Update the latest checkpoint pointer file atomically (write to temp then rename)
        latest_path_tmp = output_dir / "checkpoint_latest.txt.tmp"
        latest_path = output_dir / "checkpoint_latest.txt"
        with open(latest_path_tmp, "w", encoding="utf-8") as f:
            f.write(str(save_path.resolve())) # Store absolute path
        os.replace(latest_path_tmp, latest_path) # Atomic rename

    except Exception as e:
        logger.error(f"Failed to save checkpoint at {save_path}: {e}", exc_info=True)
        # Clean up potentially incomplete checkpoint directory?
        # if save_path.exists(): shutil.rmtree(save_path) # Example cleanup


def load_checkpoint(agent: PPOAgent, checkpoint_dir: Path) -> Tuple[int, int]:
    """Loads state from a checkpoint directory into the agent object."""
    logger.info(f"Attempting to load checkpoint from: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    try:
        # --- Load Actor Weights ---
        # Use agent.actor.load_weights which should handle adapters correctly if saved by save_weights
        logger.info(f"Loading actor weights from {checkpoint_dir}...")
        agent.actor.load_weights(str(checkpoint_dir))
        agent.actor_params = tree_map(lambda p: p, agent.actor.parameters()) # Recapture potentially updated params
        logger.info("Loaded actor weights.")
        # Note: Assumes the agent's actor model structure is already compatible (e.g., LoRA applied if needed)

        # --- Load Critic Weights ---
        critic_weights_path = checkpoint_dir / "critic.safetensors"
        if not critic_weights_path.exists():
             raise FileNotFoundError(f"Critic weights not found: {critic_weights_path}")
        loaded_critic_weights = mx.load(str(critic_weights_path))
        agent.critic_params = tree_unflatten(list(loaded_critic_weights.items()))
        agent.critic.update(agent.critic_params) # Apply weights to the stateful critic model
        logger.info("Loaded critic parameters.")

        # --- Load Optimizer States ---
        actor_opt_path = checkpoint_dir / "actor_optimizer.safetensors"
        critic_opt_path = checkpoint_dir / "critic_optimizer.safetensors"
        # Actor optimizer state (load if exists, otherwise reset)
        if actor_opt_path.exists():
            actor_opt_state_flat = list(mx.load(str(actor_opt_path)).items())
            # Ensure loaded state structure matches current trainable params structure
            expected_actor_state_tree = tree_map(lambda _: None, agent.actor.trainable_parameters())
            loaded_actor_state = tree_unflatten(actor_opt_state_flat)
            # TODO: Add structure validation if necessary before assigning
            agent.actor_opt_state = loaded_actor_state
            agent.actor_optimizer.state = agent.actor_opt_state
            logger.info("Loaded actor optimizer state.")
        else:
            logger.warning("Actor optimizer state not found. Resetting state.")
            agent.actor_opt_state = tree_map(lambda _: None, agent.actor.trainable_parameters())
            agent.actor_optimizer.state = agent.actor_opt_state
        # Critic optimizer state (load if exists, otherwise reset)
        if critic_opt_path.exists():
            critic_opt_state_flat = list(mx.load(str(critic_opt_path)).items())
            agent.critic_opt_state = tree_unflatten(critic_opt_state_flat)
            agent.critic_optimizer.state = agent.critic_opt_state
            logger.info("Loaded critic optimizer state.")
        else:
            logger.warning("Critic optimizer state not found. Resetting state.")
            agent.critic_opt_state = tree_map(lambda _: None, agent.critic.parameters())
            agent.critic_optimizer.state = agent.critic_opt_state

        # --- Load Training State ---
        state_path = checkpoint_dir / "training_state.json"
        if not state_path.exists():
             raise FileNotFoundError(f"Training state file not found: {state_path}")
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        global_step = state["global_step"]
        num_updates = state["num_updates"]

        # Restore random states
        try:
            np_state = state["np_random_state"]
            np.random.set_state(
                (np_state[0], np.array(np_state[1], dtype=np.uint32), int(np_state[2]), int(np_state[3]), float(np_state[4]))
            )
            random.setstate(tuple(state["random_state"]))
            mx.random.set_state(mx.array(state["mx_random_state"], dtype=mx.uint64))
            logger.info("Restored random states.")
        except KeyError as e:
            logger.warning(f"Could not restore random state (missing key: {e}). Using current seed.")
        except Exception as e:
            logger.error(f"Error restoring random states: {e}", exc_info=True)

        logger.info(f"Resuming from global_step={global_step}, num_updates={num_updates}.")

        # Evaluate loaded parameters and states to ensure they are loaded correctly
        mx.eval(
            agent.actor_params, agent.critic_params,
            agent.actor_opt_state, agent.critic_opt_state,
        )
        return global_step, num_updates

    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_dir}: {e}", exc_info=True)
        raise # Re-raise the exception


# === Evaluation Function ===
def evaluate(
    agent: PPOAgent, eval_env: LLMEnv, eval_steps: int, generation_temp: float = 0.0
) -> Dict[str, float]:
    """Performs evaluation on the validation set."""
    logger.info(f"Starting evaluation for {eval_steps} steps (temp={generation_temp})...")
    agent.actor.eval() # Set models to evaluation mode
    agent.critic.eval()

    total_reward = 0.0
    total_valid_format = 0.0
    actual_eval_steps = 0
    all_metrics = [] # Collect metrics from each step

    # Reset environment and get initial state
    current_prompt_ids = eval_env.reset()
    if current_prompt_ids is None:
        logger.error("Evaluation environment failed initial reset.")
        # Return default values indicating failure
        return {"eval_mean_reward": 0.0, "eval_valid_format_pc": 0.0, "eval_steps_completed": 0}

    # Setup progress bar
    progress_context = (
        Progress(
            TextColumn("[magenta]Evaluating..."), # Use color
            BarColumn(),
            TextColumn("{task.completed}/{task.total} Steps"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) if RICH_AVAILABLE else contextlib.nullcontext()
    )

    eval_task = None
    iterable = range(eval_steps) # Number of prompts to evaluate

    with progress_context as progress:
        if progress:
            eval_task = progress.add_task("Eval", total=eval_steps)
            iterable = progress.track(iterable, task_id=eval_task) # Auto-update progress

        # Evaluation loop
        for i in iterable:
            if current_prompt_ids is None or shutdown_requested:
                logger.warning(f"Evaluation interrupted at step {i+1}.")
                break

            # Generate action (text)
            try:
                action_text, _, _, _, _ = agent.generate_action(
                    current_prompt_ids, temp=generation_temp
                )
            except Exception as e:
                 logger.error(f"Evaluation step {i+1}: generate_action failed: {e}", exc_info=True)
                 action_text = "[Eval Gen Error]" # Handle generation error gracefully

            # Step the environment
            step_result = eval_env.step(action_text)
            if step_result is None:
                logger.warning(f"Evaluation step {i+1}: env.step returned None. Attempting reset.")
                current_prompt_ids = eval_env.reset() # Try to recover by resetting
                continue # Skip accumulating results for this failed step

            # Accumulate results
            reward, metrics = step_result
            total_reward += reward
            total_valid_format += metrics.get("generation_valid_format", 0.0)
            all_metrics.append(metrics)
            actual_eval_steps += 1 # Count successful steps

            # Reset environment for the next prompt
            current_prompt_ids = eval_env.reset()

    # Restore training mode (important if training continues after eval)
    agent.actor.train()
    agent.critic.train()

    # Calculate final averaged metrics
    mean_reward = total_reward / actual_eval_steps if actual_eval_steps > 0 else 0.0
    valid_format_pc = (total_valid_format / actual_eval_steps * 100.0) if actual_eval_steps > 0 else 0.0

    # Example: Aggregate another metric like format score
    avg_format_score = 0.0
    if all_metrics and actual_eval_steps > 0:
         format_scores = [m.get("format_score", 0.0) for m in all_metrics if "format_score" in m]
         if format_scores:
              avg_format_score = sum(format_scores) / len(format_scores)

    logger.info(
        f"Evaluation finished. Mean Reward: {mean_reward:.4f}, Valid Format: {valid_format_pc:.2f}% "
        f"({actual_eval_steps}/{eval_steps} steps completed)"
    )

    # Return dictionary of evaluation results
    eval_results = {
        "eval_mean_reward": mean_reward,
        "eval_valid_format_pc": valid_format_pc,
        "eval_avg_format_score": avg_format_score,
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
    """Main MLX PPO training loop."""
    global shutdown_requested

    # --- Agent Initialization ---
    # Agent holds the models, optimizers, and PPO logic
    agent = PPOAgent(
        model, critic, tokenizer,
        actor_optimizer, critic_optimizer,
        gamma, gae_lambda, clip_epsilon, value_loss_coef, entropy_coef,
        max_gen_len, ppo_epochs, ppo_batch_size, grad_clip_norm,
    )

    # --- Setup Logging and Output ---
    metrics_logger = MetricsLogger(log_file=metrics_log_file)
    output_path = Path(output_dir)

    # --- Resume Logic ---
    global_step = 0
    num_updates = 0
    if resume_from:
        checkpoint_path = Path(resume_from)
        checkpoint_dir = None
        if checkpoint_path.is_file() and checkpoint_path.name == "checkpoint_latest.txt":
            try:
                checkpoint_dir_str = checkpoint_path.read_text().strip()
                checkpoint_dir = Path(checkpoint_dir_str)
                if not checkpoint_dir.is_dir():
                     logger.error(f"Latest checkpoint path points to non-existent directory: {checkpoint_dir}")
                     checkpoint_dir = None
            except Exception as e:
                logger.error(f"Could not read/resolve latest checkpoint file {checkpoint_path}: {e}")
        elif checkpoint_path.is_dir():
            checkpoint_dir = checkpoint_path
        else:
            logger.error(f"Invalid resume path: {resume_from}. Must be directory or checkpoint_latest.txt.")

        if checkpoint_dir and checkpoint_dir.is_dir():
            try:
                # Load state into the existing agent object
                global_step, num_updates = load_checkpoint(agent, checkpoint_dir)
                logger.info(f"Successfully resumed from checkpoint: {checkpoint_dir}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {checkpoint_dir}. Starting fresh.", exc_info=True)
                global_step, num_updates = 0, 0 # Reset counters
        else:
            logger.warning(f"Could not find valid checkpoint at {resume_from}. Starting fresh.")
    else:
        logger.info("Starting training from scratch.")

    # --- Training Setup ---
    rollout_buffer = RolloutBuffer()
    start_time = time.time()
    last_save_update = num_updates # Track last save relative to updates
    last_eval_update = num_updates # Track last eval relative to updates

    # Initial environment reset
    reset_result = train_env.reset()
    if reset_result is None:
        logger.critical("Failed initial training environment reset! Cannot start training.")
        return
    current_prompt_ids = reset_result

    logger.info(
        f"Starting training loop. Start Step: {global_step}, Start Update: {num_updates + 1}, Target Steps: {total_timesteps}"
    )

    # --- Progress Bar Setup ---
    progress_context = (
        Progress(
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            TextColumn("Steps: {task.completed}/{task.total}"), TextColumn("Upd: {task.fields[updates]}"),
            TextColumn("Rew: {task.fields[reward]:.2f}"), TextColumn("Loss: {task.fields[loss]:.3f}"), # Added Loss field
            TimeRemainingColumn(), TimeElapsedColumn(), console=console,
        ) if RICH_AVAILABLE else contextlib.nullcontext()
    )

    # --- Main Training Loop ---
    with progress_context as progress:
        main_task = None
        if progress:
            main_task = progress.add_task(
                "[green]Training", total=total_timesteps, completed=global_step,
                updates=num_updates, reward=0.0, loss=0.0, kl=0.0 # Add default fields
            )

        while global_step < total_timesteps and not shutdown_requested:
            # --- Rollout Phase ---
            rollout_phase_desc = f"[yellow]Rollout {num_updates + 1}..."
            if progress: progress.update(main_task, description=rollout_phase_desc)
            else: logger.info(rollout_phase_desc)

            rollout_start_time = time.time()
            collected_rewards = []
            collected_steps_in_rollout = 0
            rollout_buffer.clear()
            agent.actor.eval()
            agent.critic.eval()

            for step_in_rollout in range(num_rollout_steps):
                if global_step >= total_timesteps or shutdown_requested: break
                if current_prompt_ids is None: # Check if env needs reset
                    logger.warning(f"Rollout step {step_in_rollout+1}: current_prompt_ids is None. Resetting.")
                    reset_result = train_env.reset()
                    if reset_result is None:
                        logger.error("Rollout: Env reset failed permanently. Stopping.")
                        shutdown_requested = True; break
                    current_prompt_ids = reset_result
                    continue

                # Generate action
                try:
                    action_text, seq_log_prob, value, gen_ids, _ = agent.generate_action(
                        current_prompt_ids, temp=generation_temp
                    )
                except Exception as gen_e:
                    logger.error(f"Rollout step {step_in_rollout+1} (Global {global_step+1}): generate_action failed: {gen_e}", exc_info=True)
                    reset_result = train_env.reset(); current_prompt_ids = reset_result; continue # Try to recover

                # Environment step
                step_result = train_env.step(action_text)
                if step_result is None:
                    logger.warning(f"Rollout step {step_in_rollout+1} (Global {global_step+1}): env.step failed. Resetting.")
                    reset_result = train_env.reset(); current_prompt_ids = reset_result; continue # Try to recover

                reward, step_info = step_result
                collected_rewards.append(reward)
                rollout_buffer.add(current_prompt_ids, gen_ids, action_text, seq_log_prob, reward, True, value)

                global_step += 1
                collected_steps_in_rollout += 1
                if progress: progress.update(main_task, advance=1)

                # Log sample text periodically
                if generate_sample_text_every > 0 and global_step % generate_sample_text_every == 0:
                    prompt_for_log = train_env.current_prompt_text # Get prompt *before* reset
                    logger.info(f"\n--- Sample Step {global_step} ---\n"
                                f"Prompt: {prompt_for_log[:200]}...\n"
                                f"Gen: {action_text[:300]}...\nReward: {reward:.3f}\n" + "-"*27)

                # Reset for next step
                reset_result = train_env.reset()
                if reset_result is None:
                    logger.error(f"Rollout: Env reset failed after step {global_step}. Stopping rollout.")
                    break
                current_prompt_ids = reset_result


            # --- End Rollout Phase ---
            mean_reward_rollout = np.mean(collected_rewards) if collected_rewards else 0.0
            rollout_duration = time.time() - rollout_start_time
            logger.info(f"Rollout {num_updates + 1} ({collected_steps_in_rollout} steps) finished ({rollout_duration:.2f}s). Mean reward: {mean_reward_rollout:.3f}")
            if progress: progress.update(main_task, reward=mean_reward_rollout)

            if len(rollout_buffer) == 0:
                logger.warning("Rollout buffer empty, skipping PPO update.")
                if collected_steps_in_rollout == 0 and not shutdown_requested:
                    logger.error("Failed to collect any steps. Stopping.")
                    shutdown_requested = True
                continue

            # --- Compute Advantages ---
            compute_gae_desc = "[cyan]Calculating GAE..."
            if progress: progress.update(main_task, description=compute_gae_desc)
            else: logger.info(compute_gae_desc)
            try:
                if current_prompt_ids is None: # Need state for last_value
                     logger.warning("Cannot compute GAE: next state is None. Resetting.")
                     reset_result = train_env.reset()
                     if reset_result is None: raise ValueError("Failed reset for GAE.")
                     current_prompt_ids = reset_result
                last_value = agent.get_value(current_prompt_ids)
                rollout_buffer.compute_advantages_and_returns(last_value, gamma, gae_lambda)
            except Exception as e:
                logger.error(f"Failed GAE calculation: {e}. Skipping update.", exc_info=True)
                continue

            # --- PPO Update Phase ---
            ppo_update_desc = f"[blue]PPO Update {num_updates + 1}..."
            if progress: progress.update(main_task, description=ppo_update_desc)
            else: logger.info(ppo_update_desc)

            update_start_time = time.time()
            avg_losses = agent.update(rollout_buffer)
            update_duration = time.time() - update_start_time

            if not avg_losses: # Handle case where update failed internally
                 logger.error(f"PPO Update {num_updates + 1} failed to return losses. Skipping post-update steps.")
                 continue # Skip logging, eval, save for this update

            num_updates += 1 # Increment only after successful update logic
            if progress: progress.update(main_task, updates=num_updates, loss=avg_losses.get("total_ppo_loss", 0.0), kl=avg_losses.get("kl_approx", 0.0))
            logger.info(f"Update {num_updates} finished ({update_duration:.2f}s). Losses: { {k: f'{v:.4f}' for k,v in avg_losses.items()} }")

            # --- Log Metrics ---
            log_data = {
                "global_step": global_step, "update_step": num_updates, "timestamp": time.time(),
                "mean_rollout_reward": mean_reward_rollout, "rollout_steps": collected_steps_in_rollout,
                "rollout_duration_sec": rollout_duration, "update_duration_sec": update_duration,
                "memory_peak_gb": mx.get_peak_memory() / 1e9, "phase": "train", **avg_losses,
            }
            metrics_logger.log(log_data)

            # --- Evaluation ---
            if val_env and eval_freq > 0 and num_updates % eval_freq == 0:
                eval_desc = "[magenta]Evaluating..."
                if progress: progress.update(main_task, description=eval_desc)
                else: logger.info(eval_desc)
                eval_start_time = time.time()
                num_eval_steps = len(val_env) # Evaluate on full validation set
                eval_metrics = evaluate(agent, val_env, eval_steps=num_eval_steps, generation_temp=0.0)
                eval_duration = time.time() - eval_start_time
                logger.info(f"Evaluation took {eval_duration:.2f}s.")
                eval_log = {"global_step": global_step, "update_step": num_updates, "timestamp": time.time(),
                            "phase": "eval", "eval_duration_sec": eval_duration, **eval_metrics}
                metrics_logger.log(eval_log)
                last_eval_update = num_updates

            # --- Save Checkpoint ---
            if save_freq > 0 and num_updates % save_freq == 0:
                save_desc = f"[green]Saving Ckpt {num_updates}..."
                if progress: progress.update(main_task, description=save_desc)
                else: logger.info(save_desc)
                save_checkpoint(agent, output_path, global_step, num_updates)
                last_save_update = num_updates

            # --- Cleanup ---
            rollout_buffer.clear()
            gc.collect()

        # --- End of Training Loop ---
        if progress:
             final_desc = "[bold green]Training Finished" if not shutdown_requested else "[bold yellow]Training Interrupted"
             progress.update(main_task, description=final_desc)

    # --- Post-Training ---
    train_duration = time.time() - start_time
    logger.info(f"Training loop finished. Total Updates: {num_updates}, Total Steps: {global_step}")
    logger.info(f"Total training time: {train_duration / 3600:.2f} hours.")
    if shutdown_requested: logger.warning("Training interrupted by user (SIGINT).")

    # --- Final Save ---
    if num_updates > last_save_update or shutdown_requested:
        logger.info("Saving final model state...")
        save_checkpoint(agent, output_path, global_step, num_updates)

    logger.info(f"Script finished. Outputs saved in {output_path.resolve()}")


# === CLI Interface (Enhanced) ===
@click.command()
# Required Paths
@click.option(
    "--model-path", required=True, type=str,
    help="Path or HF repo ID of the base Llama model (MLX format)."
)
@click.option(
    "--train-dataset-path", required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help='Path to training JSONL (with "prompt", "completion").'
)
@click.option(
    "--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path),
    help="Directory to save checkpoints, logs, and final model."
)
# Optional Paths
@click.option(
    "--val-dataset-path", default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Optional path to validation JSONL."
)
@click.option(
    "--resume-from", default=None, type=click.Path(exists=True, path_type=Path),
    help='Path to checkpoint directory or "checkpoint_latest.txt" to resume.'
)
@click.option(
    "--metrics-log-file", default="metrics.jsonl", type=str, show_default=True,
    help="Filename for metrics log within the output directory."
)
# Training Duration & Control
@click.option(
    "--total-timesteps", default=2, type=int, show_default=True,
    help="Total environment interaction steps (global steps) for training."
)
@click.option("--seed", default=42, type=int, show_default=True, help="Random seed.")
@click.option(
    "--lora-layers", default=-1, type=int, show_default=True,
    help="Number of layers to apply LoRA to (from the top). -1=full fine-tune."
)
# Rollout & PPO Hyperparameters
@click.option(
    "--num-rollout-steps", default=256, type=int, show_default=True,
    help="Steps collected per rollout before PPO update."
)
@click.option(
    "--ppo-batch-size", default=128, type=int, show_default=True,
    help="Mini-batch size for PPO update phase."
)
@click.option(
    "--ppo-epochs", default=4, type=int, show_default=True,
    help="Optimization epochs over rollout data per PPO update."
)
@click.option(
    "--actor-lr", default=2e-6, type=float, show_default=True,
    help="Learning rate for the actor (LLM)."
)
@click.option(
    "--critic-lr", default=1e-5, type=float, show_default=True,
    help="Learning rate for the critic."
)
@click.option("--gamma", default=0.99, type=float, show_default=True, help="Discount factor (GAE).")
@click.option("--gae-lambda", default=0.95, type=float, show_default=True, help="Lambda factor (GAE).")
@click.option("--clip-epsilon", default=0.2, type=float, show_default=True, help="PPO clipping epsilon.")
@click.option("--value-loss-coef", default=0.5, type=float, show_default=True, help="Value loss coefficient.")
@click.option("--entropy-coef", default=0.01, type=float, show_default=True, help="Entropy bonus coefficient.")
@click.option(
    "--grad-clip-norm", default=1.0, type=float, show_default=True,
    help="Max norm for gradient clipping (0 or negative to disable)."
)
# Generation & Environment Hyperparameters
@click.option(
    "--max-prompt-len", default=750, type=int, show_default=True,
    help="Max length for tokenized prompts (truncation applied)."
)
@click.option(
    "--max-gen-len", default=1024, type=int, show_default=True,
    help="Max tokens generated per environment step."
)
@click.option(
    "--generation-temp", default=0.3, type=float, show_default=True,
    help="Temperature for sampling during rollouts."
)
# Reporting & Saving Frequencies
@click.option(
    "--save-freq", default=10, type=int, show_default=True,
    help="Save checkpoint every N PPO UPDATES (0 to disable)."
)
@click.option(
    "--eval-freq", default=25, type=int, show_default=True,
    help="Evaluate every N PPO UPDATES (0 to disable, requires validation set)."
)
@click.option(
    "--generate-sample-text-every", default=20, type=int, show_default=True,
    help="Log sample generation every N global steps (0 to disable)."
)
# Verbosity
@click.option(
    "--log-level", default="DEBUG",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level."
)
def cli_main(**kwargs):
    """Fine-tunes a Llama model using PPO and MLX."""
    # --- Configure Logging (as early as possible) ---
    log_level_name = kwargs.pop("log_level").upper()
    log_level = getattr(logging, log_level_name)
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler) # Remove existing handlers
    # Add new handler (Rich or basic)
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_time_format="[%Y-%m-%d %H:%M:%S]"
    if RICH_AVAILABLE:
        rich_handler = RichHandler(rich_tracebacks=True, show_path=log_level <= logging.DEBUG,
                                   console=console, markup=True, log_time_format=log_time_format)
        rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]")) # Simple format for rich
        root_logger.addHandler(rich_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(log_format, datefmt=log_time_format))
        root_logger.addHandler(stream_handler)

    global logger # Make logger accessible globally if needed elsewhere
    logger = logging.getLogger(__name__)
    logger.info(f"Logging level set to {log_level_name}. Rich logging: {RICH_AVAILABLE}")

    # --- Dependency Checks ---
    if not MLX_LM_AVAILABLE: logger.critical("mlx-lm not found. pip install mlx-lm"); sys.exit(1)
    if not REWARD_FUNC_AVAILABLE: logger.critical("reward.py not found or import failed."); sys.exit(1)

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

        model_config = load_config(Path(model_path_str))
        model.config = model_config

        model.train()

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

        # ───────────────────────────────────────────────────────────────────────────────
        # Ensure the tokenizer has a *safe* pad token that is NOT the same as eos_token.
        # Drop‑in replacement for the old block — no helper function required.
        # Variables expected to exist: `tokenizer`, `logger`.
        # If `model` is also in scope we’ll resize its embedding matrix when needed.
        # ───────────────────────────────────────────────────────────────────────────────

        if (
            tokenizer.pad_token_id is None
            or tokenizer.pad_token_id == getattr(tokenizer, "eos_token_id", None)
        ):
            # Prefer an unused, reserved token that’s already in the Llama‑3 vocab
            RESERVED_PAD = "<|finetune_right_pad_id|>"

            if RESERVED_PAD in tokenizer.get_vocab():
                tokenizer.pad_token = RESERVED_PAD
                logger.info(
                    f"Set pad_token_id to existing reserved token “{RESERVED_PAD}” "
                    f"(id={tokenizer.pad_token_id})."
                )
            else:
                # Fall back: add a brand‑new pad token, then resize model embeddings
                tokenizer.add_special_tokens({"pad_token": RESERVED_PAD})
                logger.warning(
                    f"Added new pad token “{RESERVED_PAD}” (id={tokenizer.pad_token_id}); "
                    "resizing model embeddings."
                )
                if "model" in locals():
                    model.resize_token_embeddings(len(tokenizer))

        else:
            logger.debug(
                f"pad_token_id already set to {tokenizer.pad_token_id} "
                "and distinct from eos_token_id."
            )

        # Keep model.config in sync (if `model` exists)
        if "model" in locals() and getattr(model.config, "pad_token_id", None) != tokenizer.pad_token_id:
            model.config['pad_token_id'] = tokenizer.pad_token_id

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
