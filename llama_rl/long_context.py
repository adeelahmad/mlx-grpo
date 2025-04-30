# long_context.py

import asyncio
import logging
import time
from functools import partial
from typing import Optional, List, Any, AsyncGenerator, Union, Generator, Callable

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

# Assuming mlx_lm is installed and accessible for GenerationResponse
try:
    from mlx_lm.generate import GenerationResponse

    # Import cache types if needed for advanced management (currently relies on model internal cache)
    # from mlx_lm.models.cache import Cache
    MLX_LM_AVAILABLE_LC = True
except ImportError:
    logging.error(
        "mlx_lm or GenerationResponse not found in long_context.py. Please install it: pip install mlx-lm"
    )
    MLX_LM_AVAILABLE_LC = False
    # Define dummy GenerationResponse if import fails
    GenerationResponse = type("GenerationResponse", (), {})
    # Cache = type("Cache", (), {}) # Dummy cache type if needed


class LongContextHandler:
    """
    Handles processing of long contexts beyond the model's native context window
    using a sliding window approach with efficient caching during generation.
    Relies on the model's internal handling of KV cache when passed.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: nn.Module,
        max_window_size: int = 2048,
        overlap_ratio: float = 0.25,
        use_hierarchical: bool = False,  # Keep param, but unused
        cache_manager: Optional[Any] = None,
    ):  # Keep param, but unused
        """
        Initializes the LongContextHandler.

        Args:
            tokenizer: The tokenizer instance (must be HF Tokenizer, not wrapper).
            model: The language model instance (mlx.nn.Module).
            max_window_size (int): The maximum number of tokens in a single sliding window.
            overlap_ratio (float): The fraction of overlap between consecutive windows.
            use_hierarchical (bool): Flag for potential future hierarchical strategies (unused currently).
            cache_manager: An optional external cache manager (unused currently, relies on model's cache mechanism).
        """
        if not MLX_LM_AVAILABLE_LC:
            raise ImportError(
                "mlx_lm.generate.GenerationResponse required for LongContextHandler."
            )

        self.tokenizer = tokenizer
        self.model = model
        # Ensure max_window_size is reasonable (e.g., > 0)
        self.max_window_size = max(128, int(max_window_size))  # Min window size 128
        # Ensure overlap is valid (0 <= ratio < 1)
        self.overlap_ratio = max(
            0.0, min(0.9, float(overlap_ratio))
        )  # Clamp overlap ratio
        self.use_hierarchical = use_hierarchical  # Store but unused
        # self.cache_manager = cache_manager # Store but unused

        # Get model config if available (optional, for info)
        self.hidden_size = None  # Not strictly needed for generation focus
        try:
            if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
                self.hidden_size = model.config.hidden_size
            elif hasattr(model, "hidden_size"):
                self.hidden_size = model.hidden_size
            if self.hidden_size:
                logging.debug(
                    f"LongContextHandler: Detected model hidden size: {self.hidden_size}"
                )
            else:
                logging.warning(
                    "LongContextHandler: Could not detect model hidden size (not needed for generation)."
                )
        except Exception as e_hs:
            logging.warning(f"LongContextHandler: Error detecting hidden size: {e_hs}")

        # Ensure EOS token ID is available
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.eos_token_id is None:
            # Attempt common fallbacks
            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token:
                try:
                    self.eos_token_id = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.eos_token
                    )
                    logging.info(
                        f"LongContextHandler: Found eos_token_id via eos_token: {self.eos_token_id}"
                    )
                except Exception:
                    logging.error(
                        "LongContextHandler: Failed to get eos_token_id from eos_token."
                    )
                    self.eos_token_id = -1  # Mark as unavailable
            else:
                logging.error(
                    "LongContextHandler: Tokenizer does not have a defined eos_token_id or eos_token. End-of-sequence detection might fail."
                )
                self.eos_token_id = -1  # Mark as unavailable
        # Handle list of EOS tokens if applicable (though less common for base ID)
        if isinstance(self.eos_token_id, list):
            logging.warning(
                f"LongContextHandler: Tokenizer eos_token_id is a list: {self.eos_token_id}. Using the first ID for simple checks."
            )
            # Use the first ID for simple checks, or adapt logic if needed
            # self.eos_token_id_internal = self.eos_token_id[0]

    def process_long_text(
        self, text: str, processing_type: str = "generation", **kwargs
    ):
        """
        Process text potentially longer than the model's context window.
        Mainly a wrapper for the generation methods. Embeddings are basic.

        Args:
            text (str): The input text to process.
            processing_type (str): "embeddings" or "generation".
            **kwargs: Additional arguments passed to the generation method
                      (e.g., max_new_tokens, sampler, logits_processors).

        Returns:
            For embeddings: mx.array of shape (1, hidden_size) or None if error.
            For generation: Generated text string (consumes the sync generator).
        """
        # Tokenize the text
        try:
            tokens = self.tokenizer.encode(text)
            num_tokens = len(tokens)
        except Exception as e:
            logging.error(f"LongContextHandler: Error tokenizing input text: {e}")
            raise ValueError("Tokenization failed") from e

        # Delegate based on type
        if processing_type == "embeddings":
            if not self.hidden_size:
                logging.error("Cannot get embeddings: hidden_size unknown.")
                return None
            if num_tokens <= self.max_window_size:
                return self._get_embeddings_for_tokens(tokens)
            else:
                return self._get_embeddings_with_sliding_window(tokens)

        elif processing_type == "generation":
            # Consume the synchronous generator to return a single string
            generated_text = ""
            try:
                # Pass kwargs to the generator
                gen_stream = self.generate_with_sliding_window_sync(tokens, **kwargs)
                for response in gen_stream:
                    if response.token != -1:  # Append text from valid tokens
                        generated_text += response.text
                    if response.finish_reason:  # Stop if generator signals end
                        break
            except Exception as e_gen:
                logging.error(
                    f"LongContextHandler: Error during generation in process_long_text: {e_gen}",
                    exc_info=True,
                )
                return f"[Generation Error: {e_gen}]"
            return generated_text

        else:
            raise ValueError(f"Unsupported processing_type: {processing_type}")

    def _get_embeddings_with_sliding_window(
        self, tokens: List[int]
    ) -> Optional[mx.array]:
        """Extract embeddings using sliding window approach (simple averaging)."""
        if not self.hidden_size:
            logging.error("Cannot calculate embeddings without knowing hidden_size.")
            return None

        window_size = self.max_window_size
        stride = int(window_size * (1 - self.overlap_ratio))
        if stride <= 0:
            stride = max(
                1, window_size // 2
            )  # Ensure stride is positive and at least 1
        window_embeddings = []
        window_weights = []
        num_tokens = len(tokens)

        # Process each window
        i = 0
        while i < num_tokens:
            start_idx = i
            end_idx = min(i + window_size, num_tokens)
            window_tokens = tokens[start_idx:end_idx]

            # Skip tiny windows unless it's the very last segment
            min_meaningful_len = max(
                10, int(window_size * 0.1)
            )  # Define a minimum meaningful length
            if len(window_tokens) < min_meaningful_len and end_idx != num_tokens:
                logging.debug(
                    f"Skipping tiny embedding window {start_idx}-{end_idx} (len {len(window_tokens)})"
                )
                i += stride  # Move to next stride start
                continue

            # Determine window position: beginning, middle, or end
            position = "middle"
            if start_idx == 0:
                position = "beginning"
            if end_idx == num_tokens:
                position = "end"

            # Assign higher weight to beginning and end windows
            weight = 1.5 if position in ["beginning", "end"] else 1.0

            # Get embeddings for this window
            logging.debug(
                f"Processing embedding window {start_idx}-{end_idx} (pos: {position}, weight: {weight})"
            )
            embedding = self._get_embeddings_for_tokens(window_tokens)
            if embedding is not None:  # Check if embedding succeeded
                window_embeddings.append(embedding)
                window_weights.append(weight)

            # Ensure we cover the end of the sequence if stride skips it
            if end_idx == num_tokens:
                break  # Reached the end

            # Move to next window start
            i += stride
            # If the next window would start beyond the tokens, break
            if i >= num_tokens:
                break
            # If the stride leaves a small gap at the end, ensure the last window includes the end
            if (
                i < num_tokens
                and num_tokens - i < stride
                and num_tokens - i < window_size
            ):
                # Force the next window to start such that it includes the last token
                i = max(0, num_tokens - window_size)

        # Combine window embeddings with weighted average
        if not window_embeddings:
            logging.warning("No embeddings generated from sliding window.")
            return mx.zeros(
                (1, self.hidden_size), dtype=mx.float32
            )  # Use float32 for safety

        total_weight = sum(window_weights)
        if total_weight == 0:  # Avoid division by zero
            logging.warning(
                "Total weight for embeddings is zero. Using simple average."
            )
            # Fallback: simple average
            try:
                stacked_embeddings = mx.stack(window_embeddings, axis=0)
                combined_embedding = mx.mean(stacked_embeddings, axis=0)
            except Exception as e_stack:
                logging.error(
                    f"Error stacking embeddings for fallback average: {e_stack}"
                )
                return mx.zeros((1, self.hidden_size), dtype=mx.float32)
        else:
            try:
                # Ensure embeddings are stackable (should be [1, hidden_size])
                stacked_embeddings = mx.stack(
                    [emb.reshape(1, -1) for emb in window_embeddings], axis=0
                )  # Shape [N, 1, hidden_size]
                weights_tensor = mx.array(
                    window_weights, dtype=stacked_embeddings.dtype
                ).reshape(
                    -1, 1, 1
                )  # Shape [N, 1, 1]
                weighted_sum = mx.sum(
                    stacked_embeddings * weights_tensor, axis=0
                )  # Shape [1, hidden_size]
                combined_embedding = weighted_sum / total_weight
            except Exception as e_combine:
                logging.error(
                    f"Error combining embeddings: {e_combine}. Falling back to mean."
                )
                try:
                    stacked_embeddings = mx.stack(window_embeddings, axis=0)
                    combined_embedding = mx.mean(stacked_embeddings, axis=0)
                except Exception as e_stack_fallback:
                    logging.error(
                        f"Error stacking embeddings for fallback average after combine error: {e_stack_fallback}"
                    )
                    return mx.zeros((1, self.hidden_size), dtype=mx.float32)

        return combined_embedding.reshape(
            1, self.hidden_size
        )  # Ensure correct final shape [1, H]

    def generate_with_sliding_window_sync(
        self,
        tokens: List[int],
        max_new_tokens: int = 512,
        sampler: Optional[Callable] = None,
        logits_processors: Optional[List[Callable]] = None,
    ) -> Generator[GenerationResponse, None, None]:
        """
        Synchronously generate tokens using a sliding window approach,
        yielding GenerationResponse objects for each new token.

        Relies on the model's internal cache handling. It passes the cache
        object between steps, assuming the model updates and returns it correctly.

        Args:
            tokens (List[int]): The initial prompt tokens.
            max_new_tokens (int): Maximum number of new tokens to generate.
            sampler (Optional[Callable]): A function to sample the next token from logits.
                                          If None, uses argmax.
            logits_processors (Optional[List[Callable]]): A list of functions to process
                                                          logits before sampling.

        Yields:
            GenerationResponse: An object containing the generated text chunk, token ID,
                                and other generation metadata for each step.
        """
        if not MLX_LM_AVAILABLE_LC:
            logging.error(
                "mlx_lm.generate.GenerationResponse not available. Cannot generate."
            )
            yield GenerationResponse(
                text="Error: MLX_LM not available", token=-1, finish_reason="error"
            )
            return

        prompt_token_count = len(tokens)
        generated_tokens = list(tokens)  # Full sequence including prompt
        sliding_window_size = self.max_window_size
        eos_token_id_internal = self.eos_token_id
        is_eos_list = isinstance(eos_token_id_internal, list)

        # KV cache state for this generation sequence
        current_kv_cache = None
        gen_token_count = 0
        start_time = time.time()
        overall_gen_tps_calculator = 0.0

        logging.debug(
            f"Starting sync sliding window generation. Prompt tokens: {prompt_token_count}, Max new: {max_new_tokens}, Window: {sliding_window_size}"
        )

        try:
            self.model.eval()  # Ensure model is in eval mode

            for i in range(max_new_tokens):
                step_start_time = time.time()
                current_total_len = len(generated_tokens)

                # Determine the window of tokens to feed the model
                if current_total_len <= sliding_window_size:
                    # Initial phase: process the tokens generated so far
                    # If cache exists, only process the last token
                    if (
                        current_kv_cache is not None
                        and current_total_len > prompt_token_count
                    ):
                        # Feed only the last generated token if cache is valid
                        window_tokens_indices = [generated_tokens[-1]]
                        inputs = mx.array([window_tokens_indices])
                        use_cache_for_step = current_kv_cache
                        logging.debug(
                            f"Step {i+1}: Using cache, processing 1 new token."
                        )
                    else:
                        # First token(s) or cache invalid: process entire current sequence
                        window_tokens_indices = generated_tokens
                        inputs = mx.array([window_tokens_indices])
                        use_cache_for_step = None  # Rebuild cache
                        logging.debug(
                            f"Step {i+1}: No cache or first step, processing {len(window_tokens_indices)} tokens."
                        )
                else:
                    # Sliding window phase: process the last `sliding_window_size` tokens
                    start_idx = current_total_len - sliding_window_size
                    window_tokens_indices = generated_tokens[start_idx:]
                    inputs = mx.array([window_tokens_indices])
                    # In sliding phase, we typically need to rebuild cache state based on the window
                    # unless a more sophisticated cache trimming/management is used.
                    # For simplicity, assume we rebuild cache for the window.
                    use_cache_for_step = None  # Rebuild cache for the slided window
                    logging.debug(
                        f"Step {i+1}: Sliding window active, processing {len(window_tokens_indices)} tokens from index {start_idx}."
                    )

                # --- Model Forward Pass ---
                try:
                    # Pass the current cache state; model should return updated cache
                    # Handle models that might not return cache explicitly
                    model_output = self.model(inputs, cache=use_cache_for_step)

                    if isinstance(model_output, tuple) and len(model_output) == 2:
                        (
                            outputs,
                            current_kv_cache,
                        ) = model_output  # Model returned logits and cache
                    elif isinstance(model_output, mx.array):
                        outputs = model_output  # Model returned only logits
                        # Cache wasn't returned, subsequent steps might not use it effectively
                        if use_cache_for_step is not None:
                            logging.warning(
                                "Model did not return cache object; cache usage might be ineffective."
                            )
                        current_kv_cache = None  # Assume cache is lost/invalid
                    else:
                        raise TypeError(
                            f"Unexpected model output type: {type(model_output)}"
                        )

                except Exception as e_model:
                    logging.error(
                        f"Error during model forward pass in sliding window (step {i+1}): {e_model}",
                        exc_info=True,
                    )
                    yield GenerationResponse(
                        text=f" Error: Model forward pass failed: {e_model}",
                        token=-1,
                        finish_reason="error",
                    )
                    return

                # --- Token Sampling ---
                # Get logits for the very last token position in the output sequence
                next_token_logits = outputs[:, -1, :]

                # Apply logit processors (if provided)
                if logits_processors:
                    processed_logits = next_token_logits  # Start with original logits
                    try:
                        # Processors might modify logits in place or return new ones
                        for processor in logits_processors:
                            # Assuming processor takes (tokens, logits) - adapt if different interface
                            # Pass the *input* tokens for context if needed by processor
                            processed_logits = processor(inputs, processed_logits)
                        next_token_logits = (
                            processed_logits  # Use the final processed logits
                        )
                    except Exception as e_proc:
                        logging.error(
                            f"Error applying logit processor: {e_proc}. Using unprocessed logits.",
                            exc_info=True,
                        )
                        # Continue with unprocessed logits

                # Apply sampler (if provided) or default sampling
                if sampler:
                    try:
                        # Sampler typically returns sampled token ID
                        next_token = sampler(next_token_logits)
                        # Ensure next_token is an integer ID
                        if isinstance(next_token, mx.array):
                            next_token = next_token.item()
                        next_token = int(next_token)
                    except Exception as e_sample:
                        logging.error(
                            f"Error applying sampler: {e_sample}. Falling back to argmax.",
                            exc_info=True,
                        )
                        next_token = mx.argmax(next_token_logits, axis=-1).item()
                else:
                    # Default to argmax if no sampler provided
                    next_token = mx.argmax(next_token_logits, axis=-1).item()

                # --- Append and Yield ---
                generated_tokens.append(next_token)
                gen_token_count += 1

                try:
                    # Decode only the newly generated token
                    decoded_text = self.tokenizer.decode([next_token])
                except Exception as e_decode:
                    logging.error(f"Error decoding token ID {next_token}: {e_decode}")
                    decoded_text = "[DECODE_ERR]"

                step_end_time = time.time()
                step_duration = step_end_time - step_start_time
                current_gen_tps = 1.0 / step_duration if step_duration > 0 else 0
                # Calculate overall TPS smoothed over time
                total_duration = step_end_time - start_time
                overall_gen_tps_calculator = (
                    gen_token_count / total_duration if total_duration > 0 else 0.0
                )

                # Construct and yield a GenerationResponse object
                response_obj = GenerationResponse(
                    text=decoded_text,
                    token=next_token,
                    logprobs=None,  # Logprobs complex with sliding window, omit for now
                    prompt_tokens=prompt_token_count,  # Initial prompt tokens
                    prompt_tps=0.0,  # Prompt TPS not tracked here
                    generation_tokens=gen_token_count,
                    generation_tps=overall_gen_tps_calculator,  # Use overall TPS
                    peak_memory=0.0,  # Memory not tracked here
                    finish_reason=None,
                    from_draft=False,
                    # Include cache state if needed by consumer? Probably not.
                    # prompt_cache = current_kv_cache
                )
                yield response_obj

                # --- Check for Stop Condition ---
                stop_condition = False
                if is_eos_list:
                    if next_token in eos_token_id_internal:
                        stop_condition = True
                elif next_token == eos_token_id_internal:
                    stop_condition = True

                if stop_condition:
                    logging.debug(
                        f"LongContextHandler: EOS token {next_token} generated. Stopping."
                    )
                    # Yield final response indicating stop reason
                    yield GenerationResponse(
                        text="",
                        token=-1,
                        finish_reason="stop",
                        generation_tokens=gen_token_count,
                        generation_tps=overall_gen_tps_calculator,
                    )
                    return  # End generation

            # --- Max Tokens Reached ---
            logging.debug(
                f"LongContextHandler: Max new tokens ({max_new_tokens}) reached."
            )
            total_duration_end = time.time() - start_time
            overall_gen_tps_final = (
                gen_token_count / total_duration_end if total_duration_end > 0 else 0.0
            )
            yield GenerationResponse(
                text="",
                token=-1,
                finish_reason="length",
                generation_tokens=gen_token_count,
                generation_tps=overall_gen_tps_final,
            )

        except Exception as e_main:
            logging.error(
                f"LongContextHandler: Unexpected error during synchronous generation: {e_main}",
                exc_info=True,
            )
            yield GenerationResponse(
                text=f" Error: {e_main}",
                token=-1,
                finish_reason="error",
                generation_tokens=gen_token_count,
            )
        finally:
            # Optional: Clean up cache if necessary
            # current_kv_cache = None
            logging.debug("Finished sync sliding window generation.")

    async def async_generate_with_sliding_window(
        self,
        tokens: List[int],
        max_new_tokens: int = 512,
        sampler: Optional[Callable] = None,
        logits_processors: Optional[List[Callable]] = None,
    ) -> AsyncGenerator[GenerationResponse, None]:
        """
        Asynchronously generate tokens using a sliding window approach,
        yielding GenerationResponse objects. Uses asyncio run_in_executor for model calls.

        Args:
            tokens (List[int]): The initial prompt tokens.
            max_new_tokens (int): Maximum number of new tokens to generate.
            sampler (Optional[Callable]): Sampler function. Uses argmax if None.
            logits_processors (Optional[List[Callable]]): List of logit processing functions.

        Yields:
            GenerationResponse: Response object for each generated token.
        """
        if not MLX_LM_AVAILABLE_LC:
            logging.error(
                "mlx_lm.generate.GenerationResponse not available. Cannot generate."
            )
            yield GenerationResponse(
                text="Error: MLX_LM not available", token=-1, finish_reason="error"
            )
            return

        prompt_token_count = len(tokens)
        generated_tokens = list(tokens)
        sliding_window_size = self.max_window_size
        eos_token_id_internal = self.eos_token_id
        is_eos_list = isinstance(eos_token_id_internal, list)

        current_kv_cache = None
        loop = asyncio.get_running_loop()
        gen_token_count = 0
        start_time = time.time()
        overall_gen_tps_calculator = 0.0

        logging.debug(
            f"Starting async sliding window generation. Prompt tokens: {prompt_token_count}, Max new: {max_new_tokens}, Window: {sliding_window_size}"
        )

        try:
            # Ensure model is in eval mode (assuming it's thread-safe or handled externally)
            # self.model.eval() # Might cause issues if called concurrently

            for i in range(max_new_tokens):
                step_start_time = time.time()
                current_total_len = len(generated_tokens)

                # Determine window and cache usage (same logic as sync)
                if current_total_len <= sliding_window_size:
                    if (
                        current_kv_cache is not None
                        and current_total_len > prompt_token_count
                    ):
                        window_tokens_indices = [generated_tokens[-1]]
                        inputs = mx.array([window_tokens_indices])
                        use_cache_for_step = current_kv_cache
                        log_msg = f"Step {i+1}: Using cache, processing 1 new token."
                    else:
                        window_tokens_indices = generated_tokens
                        inputs = mx.array([window_tokens_indices])
                        use_cache_for_step = None
                        log_msg = f"Step {i+1}: No cache or first step, processing {len(window_tokens_indices)} tokens."
                else:
                    start_idx = current_total_len - sliding_window_size
                    window_tokens_indices = generated_tokens[start_idx:]
                    inputs = mx.array([window_tokens_indices])
                    use_cache_for_step = None  # Rebuild cache for the slided window
                    log_msg = f"Step {i+1}: Sliding window active, processing {len(window_tokens_indices)} tokens from index {start_idx}."
                # logging.debug(log_msg) # Can be very verbose

                # --- Model Forward Pass (Async) ---
                try:
                    # Use run_in_executor for the potentially blocking model call
                    # Ensure partial captures the *current* cache state for the executor context
                    model_call = partial(self.model, inputs, cache=use_cache_for_step)
                    model_output = await loop.run_in_executor(None, model_call)

                    # Process output (same logic as sync)
                    if isinstance(model_output, tuple) and len(model_output) == 2:
                        outputs, current_kv_cache = model_output
                    elif isinstance(model_output, mx.array):
                        outputs = model_output
                        if use_cache_for_step is not None:
                            logging.warning(
                                "Model did not return cache object; cache usage might be ineffective."
                            )
                        current_kv_cache = None
                    else:
                        raise TypeError(
                            f"Unexpected model output type: {type(model_output)}"
                        )

                except Exception as e_model:
                    logging.error(
                        f"Error during async model forward pass (step {i+1}): {e_model}",
                        exc_info=True,
                    )
                    yield GenerationResponse(
                        text=f" Error: Model forward pass failed: {e_model}",
                        token=-1,
                        finish_reason="error",
                    )
                    return

                # --- Token Sampling (Sync - typically fast) ---
                next_token_logits = outputs[:, -1, :]
                if logits_processors:
                    processed_logits = next_token_logits
                    try:
                        for processor in logits_processors:
                            processed_logits = processor(inputs, processed_logits)
                        next_token_logits = processed_logits
                    except Exception as e_proc:
                        logging.error(
                            f"Error applying logit processor: {e_proc}. Using unprocessed logits.",
                            exc_info=True,
                        )

                if sampler:
                    try:
                        next_token = sampler(next_token_logits)
                        if isinstance(next_token, mx.array):
                            next_token = next_token.item()
                        next_token = int(next_token)
                    except Exception as e_sample:
                        logging.error(
                            f"Error applying sampler: {e_sample}. Falling back to argmax.",
                            exc_info=True,
                        )
                        next_token = mx.argmax(next_token_logits, axis=-1).item()
                else:
                    next_token = mx.argmax(next_token_logits, axis=-1).item()

                # --- Append and Yield ---
                generated_tokens.append(next_token)
                gen_token_count += 1

                try:
                    decoded_text = self.tokenizer.decode([next_token])
                except Exception as e_decode:
                    logging.error(f"Error decoding token ID {next_token}: {e_decode}")
                    decoded_text = "[DECODE_ERR]"

                step_end_time = time.time()
                total_duration = step_end_time - start_time
                overall_gen_tps_calculator = (
                    gen_token_count / total_duration if total_duration > 0 else 0.0
                )

                response_obj = GenerationResponse(
                    text=decoded_text,
                    token=next_token,
                    logprobs=None,
                    prompt_tokens=prompt_token_count,
                    prompt_tps=0.0,
                    generation_tokens=gen_token_count,
                    generation_tps=overall_gen_tps_calculator,
                    peak_memory=0.0,
                    finish_reason=None,
                )
                yield response_obj  # Yield the response object asynchronously

                # --- Check for Stop Condition ---
                stop_condition = False
                if is_eos_list:
                    if next_token in eos_token_id_internal:
                        stop_condition = True
                elif next_token == eos_token_id_internal:
                    stop_condition = True

                if stop_condition:
                    logging.debug(
                        f"LongContextHandler (Async): EOS token {next_token} generated. Stopping."
                    )
                    yield GenerationResponse(
                        text="",
                        token=-1,
                        finish_reason="stop",
                        generation_tokens=gen_token_count,
                        generation_tps=overall_gen_tps_calculator,
                    )
                    return

            # --- Max Tokens Reached ---
            logging.debug(
                f"LongContextHandler (Async): Max new tokens ({max_new_tokens}) reached."
            )
            total_duration_end = time.time() - start_time
            overall_gen_tps_final = (
                gen_token_count / total_duration_end if total_duration_end > 0 else 0.0
            )
            yield GenerationResponse(
                text="",
                token=-1,
                finish_reason="length",
                generation_tokens=gen_token_count,
                generation_tps=overall_gen_tps_final,
            )

        except Exception as e_main:
            logging.error(
                f"LongContextHandler: Unexpected error during async generation: {e_main}",
                exc_info=True,
            )
            yield GenerationResponse(
                text=f" Error: {e_main}",
                token=-1,
                finish_reason="error",
                generation_tokens=gen_token_count,
            )
        finally:
            logging.debug("Finished async sliding window generation.")

    def _get_embeddings_for_tokens(self, tokens: List[int]) -> Optional[mx.array]:
        """
        Extract embeddings for a single window of tokens.
        NOTE: This implementation is model-specific and might need adjustments.
        It assumes a transformer structure with `embed_tokens`, `layers`, and `norm`.
        Returns None on failure.
        """
        if not self.hidden_size:
            logging.error("Cannot get embeddings: hidden_size unknown.")
            return None
        if not tokens:
            logging.warning("Attempted to get embeddings for empty token list.")
            return mx.zeros((1, self.hidden_size), dtype=mx.float32)

        try:
            inputs = mx.array([tokens])
            # 1. Get token embeddings (adapt based on model structure)
            embed_layer = None
            if hasattr(self.model, "embed_tokens") and callable(
                self.model.embed_tokens
            ):
                embed_layer = self.model.embed_tokens
            elif hasattr(self.model, "model") and hasattr(
                self.model.model, "embed_tokens"
            ):  # Common in HF mlx models
                embed_layer = self.model.model.embed_tokens
            elif hasattr(self.model, "transformer") and hasattr(
                self.model.transformer, "wte"
            ):  # GPT-like
                embed_layer = self.model.transformer.wte
            # Add more checks if needed for other model types

            if embed_layer is None:
                logging.error("Could not find embedding layer in the model.")
                return None
            x = embed_layer(inputs)

            # 2. Pass through transformer layers (adapt based on model structure)
            layers_attr = None
            if hasattr(self.model, "layers"):
                layers_attr = self.model.layers
            elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                layers_attr = self.model.model.layers
            elif hasattr(self.model, "transformer") and hasattr(
                self.model.transformer, "h"
            ):
                layers_attr = self.model.transformer.h
            # Add more checks if needed

            if layers_attr is None:
                logging.warning(
                    "Could not find transformer layers attribute. Using embedding output directly."
                )
                hidden_states = x  # Use embedding output if no layers found
            else:
                # This loop structure assumes standard pre/post-norm blocks
                for layer in layers_attr:
                    # Example for LLaMA-like structure in mlx_lm, adjust if needed
                    if (
                        hasattr(layer, "self_attn")
                        and hasattr(layer, "mlp")
                        and hasattr(layer, "input_layernorm")
                        and hasattr(layer, "post_attention_layernorm")
                    ):
                        residual = x
                        x_norm = layer.input_layernorm(x)
                        attn_output = layer.self_attn(
                            x_norm, mask=None, cache=None
                        )  # No mask/cache for embeddings
                        x = residual + attn_output
                        residual = x
                        x_norm = layer.post_attention_layernorm(x)
                        mlp_output = layer.mlp(x_norm)
                        x = residual + mlp_output
                    else:
                        # Fallback: just call the layer if structure unknown
                        logging.warning(
                            f"Unknown layer structure: {type(layer)}. Calling layer directly (might fail)."
                        )
                        try:
                            # Try calling with just input, might need adaptation
                            x = layer(x)
                        except Exception as e_layer:
                            logging.error(
                                f"Failed to call unknown layer structure: {e_layer}"
                            )
                            return None  # Abort embedding extraction

                # 3. Final normalization (if exists)
                final_norm_layer = None
                if hasattr(self.model, "norm") and callable(self.model.norm):
                    final_norm_layer = self.model.norm
                elif hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
                    final_norm_layer = self.model.model.norm
                elif hasattr(self.model, "transformer") and hasattr(
                    self.model.transformer, "ln_f"
                ):
                    final_norm_layer = self.model.transformer.ln_f

                if final_norm_layer:
                    hidden_states = final_norm_layer(x)
                else:
                    logging.warning(
                        "Could not find final normalization layer. Using output of last layer."
                    )
                    hidden_states = x  # Use output of last layer if no final norm

            # 4. Pooling (mean pooling over sequence length)
            if (
                hidden_states.ndim == 3 and hidden_states.shape[0] == 1
            ):  # Expect [1, seq_len, hidden_size]
                pooled_embedding = mx.mean(
                    hidden_states, axis=1
                )  # Mean over seq_len dim -> [1, hidden_size]
                # Ensure correct dtype (e.g., float32) if needed downstream
                return pooled_embedding.astype(mx.float32)
            else:
                logging.error(
                    f"Unexpected hidden state shape for pooling: {hidden_states.shape}"
                )
                return None

        except Exception as e:
            logging.error(f"Error extracting embeddings for tokens: {e}", exc_info=True)
            return None

    def _sample_token(
        self, logits: mx.array, temperature: float = 0.7, top_k: int = 50
    ) -> int:
        """
        Sample next token ID from logits using temperature and top-k sampling.
        (Internal helper, kept from user input - ensure it's robust)
        """
        # --- Input Validation ---
        if logits.ndim > 1:
            if logits.shape[:-1] == (1,) * (logits.ndim - 1):
                logits = logits.reshape(-1)
            else:
                raise ValueError(
                    f"Input logits must be 1D or squeezable to 1D, but got shape {logits.shape}"
                )
        elif logits.ndim == 0:
            raise ValueError("Input logits cannot be a scalar.")

        vocab_size = logits.size

        try:
            # --- Temperature Scaling ---
            if temperature == 0:  # Greedy
                next_token_idx = mx.argmax(logits, axis=-1)
                return int(next_token_idx.item())
            elif temperature < 0:
                raise ValueError("Temperature must be non-negative.")
            else:
                scaled_logits = logits / temperature

            # --- Top-K Filtering ---
            if top_k > 0 and top_k < vocab_size:
                top_k_indices = mx.argpartition(scaled_logits, -top_k)[-top_k:]
                mask = mx.zeros_like(scaled_logits, dtype=mx.bool_)
                # Use scatter for robust index assignment
                updates = mx.ones_like(top_k_indices, dtype=mx.bool_)
                # Ensure indices are correctly shaped for scatter if needed
                scatter_indices = (
                    top_k_indices[None, :] if top_k_indices.ndim == 1 else top_k_indices
                )
                mask = mx.scatter(
                    mask, scatter_indices, updates, axes=[0]
                )  # Assumes 1D logits
                final_logits = mx.where(mask, scaled_logits, mx.array(-mx.inf))
            else:
                final_logits = scaled_logits  # No top-k or top-k >= vocab_size

            # --- Sampling ---
            try:
                # Use mx.random.categorical - implicitly handles softmax if logits are passed
                # Ensure float32 for stability if needed
                final_logits_float = (
                    final_logits.astype(mx.float32)
                    if final_logits.dtype != mx.float32
                    else final_logits
                )
                next_token_idx = mx.random.categorical(final_logits_float)
            except Exception as e_categorical:
                logging.warning(
                    f"mx.random.categorical failed: {e_categorical}. Falling back to argmax on filtered probs."
                )
                # Fallback: Argmax on probabilities after softmax (safer than argmax on -inf logits)
                probs = mx.softmax(final_logits, axis=-1)
                next_token_idx = mx.argmax(probs, axis=-1)

            return int(next_token_idx.item())

        except Exception as e_main:
            logging.error(
                f"Token sampling failed unexpectedly: {e_main}. Using fallback argmax on original logits.",
                exc_info=True,
            )
            try:
                return int(mx.argmax(logits).item())  # Ultimate fallback
            except Exception as e_fallback:
                logging.error(
                    f"Ultimate fallback argmax also failed: {e_fallback}. Returning token 0.",
                    exc_info=True,
                )
                return 0
