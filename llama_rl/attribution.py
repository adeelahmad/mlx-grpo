import logging
import math
import re
from typing import List, Tuple, Optional, Any, Union, Dict

# Assume 'mxnet' or 'mlx.core' is imported as 'mx' based on the original code.
# If using MLX (which seems likely given mx.stop_gradient, mx.grad):
import mlx.core as mx
import mlx.nn as nn

# Assume 'mlx.core' is imported as 'mx'
import mlx.core as mx
import mlx.nn as nn

# Import TokenizerWrapper for instanceof check
try:
    from mlx_lm.tokenizer_utils import TokenizerWrapper
except ImportError:
    logging.warning(
        "Could not import TokenizerWrapper from mlx_lm.tokenizer_utils. Type checking might be incomplete."
    )

    # Define a dummy class if import fails to avoid NameError, although functionality will be limited
    class TokenizerWrapper:
        pass


# If using MXNet (less likely given context, but adapting requires significant changes):
# import mxnet.ndarray as mx
# from mxnet import autograd

# --- Placeholder for External Dependencies ---
# These classes need to be defined elsewhere in your project.
# Example stubs are provided for context.


class MockTokenizer:
    """Placeholder for a real tokenizer (like HuggingFace's)."""

    def encode(self, text: str) -> List[int]:
        # Simple character-based encoding for demonstration
        return [ord(c) for c in text][:512]  # Limit sequence length

    def decode(self, token_ids: List[int]) -> str:
        return "".join(chr(t) for t in token_ids)


class MockModelConfig:
    """Placeholder for model configuration."""

    hidden_size: int = 768  # Example hidden size
    vocab_size: int = 30000  # Example vocab size


class MockLLMModel(nn.Module):  # Inherit from nn.Module for MLX
    """Placeholder for a real Language Model (like a transformer)."""

    def __init__(self, config: MockModelConfig):
        super().__init__()
        self.config = config
        # Simplified layers for demonstration
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Simulate transformer blocks implicitly in forward pass
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """Simulates model forward pass, supporting input_ids or inputs_embeds."""
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            # Validate input_ids tensor
            if not isinstance(input_ids, mx.array) or input_ids.ndim < 2:
                logging.warning(
                    f"MockModel received invalid input_ids shape: {input_ids.shape if isinstance(input_ids, mx.array) else type(input_ids)}"
                )
                # Return zero logits matching expected shape: (batch, seq_len, vocab_size)
                # Use a small seq_len if possible, or estimate from hidden_size if desperate
                seq_len = 5  # Arbitrary small seq len for fallback
                return mx.zeros((1, seq_len, self.config.vocab_size))

            # Ensure input_ids is integer type
            if input_ids.dtype != mx.int32 and input_ids.dtype != mx.int64:
                input_ids = input_ids.astype(mx.int32)  # Cast if needed

            # Check for out-of-bounds token IDs
            if mx.max(input_ids) >= self.config.vocab_size or mx.min(input_ids) < 0:
                logging.warning(
                    f"Input IDs out of vocabulary range [0, {self.config.vocab_size-1}]"
                )
                # Clamp or handle appropriately - for now, proceed cautiously or return zeros
                # Returning zeros might be safer if vocab errors are critical
                seq_len = input_ids.shape[1]
                return mx.zeros((1, seq_len, self.config.vocab_size))

            inputs_embeds = self.embed_tokens(input_ids)

        # Simplified forward pass: just pass embeddings to LM head
        # A real model would have transformer layers here.
        # Shape of inputs_embeds: (batch_size, sequence_length, hidden_size)
        # Shape of logits: (batch_size, sequence_length, vocab_size)
        logits = self.lm_head(inputs_embeds)
        return logits

    # Expose embedding layer directly if needed (like original code assumed)
    @property
    def model(self):
        # Provides access like `llm_model.model.embed_tokens`
        return self


# --- Observation Handler ---
class EnhancedObservationHandler:
    """Handles text processing, tokenization, chunking, and embedding generation."""

    def __init__(
        self, llm_model, llm_model_config=None, tokenizer=None, hidden_size=None
    ):
        """
        Initializes the handler, ensuring correct tokenizer assignment and validation.
        """
        func_name = "EnhancedObservationHandler.__init__"
        logging.debug(f"{func_name}: Initializing...")
        self.llm_model = llm_model
        logging.debug(f"{func_name}: Received tokenizer type: {type(tokenizer)}")

        # --- Determine the actual underlying tokenizer object ---
        actual_tokenizer = None
        tokenizer_source = "provided explicitly"
        if tokenizer is None:
            tokenizer_source = "inferred from llm_model"
            logging.debug(
                f"{func_name}: Tokenizer not provided, attempting to infer from llm_model."
            )
            tokenizer_maybe_wrapped = getattr(llm_model, "tokenizer", None)
            if tokenizer_maybe_wrapped is None:
                logging.error(
                    f"{func_name}: Tokenizer not provided and could not be inferred from llm_model."
                )
                raise ValueError(
                    "Tokenizer not provided and could not be inferred from llm_model."
                )
            tokenizer_to_check = tokenizer_maybe_wrapped
        else:
            tokenizer_to_check = tokenizer

        # Check if the object is the mlx_lm wrapper
        if isinstance(tokenizer_to_check, TokenizerWrapper):
            logging.debug(
                f"{func_name}: Tokenizer ({tokenizer_source}) is TokenizerWrapper, accessing underlying tokenizer."
            )
            actual_tokenizer = getattr(tokenizer_to_check, "_tokenizer", None)
            if actual_tokenizer is None:
                logging.error(
                    f"{func_name}: TokenizerWrapper ({tokenizer_source}) found but its underlying '_tokenizer' is None."
                )
                raise ValueError(
                    "TokenizerWrapper found but its underlying '_tokenizer' is None."
                )
        else:
            actual_tokenizer = tokenizer_to_check
            logging.debug(
                f"{func_name}: Using tokenizer ({tokenizer_source}) directly."
            )

        # --- CRITICAL VALIDATION on the actual_tokenizer ---
        is_callable_tokenizer = callable(actual_tokenizer)
        has_encode_method = hasattr(actual_tokenizer, "encode") and callable(
            actual_tokenizer.encode
        )
        if not is_callable_tokenizer or not has_encode_method:
            logging.error(
                f"{func_name}: Invalid actual tokenizer object. Must be callable and have 'encode'. Type: {type(actual_tokenizer)}"
            )
            raise TypeError(
                f"Could not obtain a valid tokenizer. Final Type: {type(actual_tokenizer)}"
            )

        self.tokenizer = actual_tokenizer
        logging.debug(
            f"{func_name}: Successfully validated and assigned actual tokenizer (Type: {type(self.tokenizer)})"
        )

        # --- Handle Pad Token ID ---
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            logging.debug(f"{func_name}: pad_token_id not found. Trying eos_token_id.")
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if pad_token_id is None:
                logging.warning(
                    f"{func_name}: Actual tokenizer has no pad_token_id or eos_token_id. Attempting to add/use [PAD]."
                )
                try:
                    # Check vocab using get_vocab() if available
                    vocab = {}
                    if hasattr(self.tokenizer, "get_vocab") and callable(
                        self.tokenizer.get_vocab
                    ):
                        vocab = self.tokenizer.get_vocab()

                    if "[PAD]" not in vocab:
                        logging.debug(
                            f"{func_name}: [PAD] not in vocab. Adding special token."
                        )
                        # Use add_special_tokens if available
                        if hasattr(self.tokenizer, "add_special_tokens") and callable(
                            self.tokenizer.add_special_tokens
                        ):
                            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                        else:
                            logging.warning(
                                f"{func_name}: Tokenizer lacks add_special_tokens method. Cannot add [PAD]."
                            )
                            raise ValueError("Cannot add [PAD] token.")
                    else:
                        logging.debug(f"{func_name}: [PAD] token found in vocab.")

                    # Try converting the token to ID
                    if hasattr(self.tokenizer, "convert_tokens_to_ids") and callable(
                        self.tokenizer.convert_tokens_to_ids
                    ):
                        pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
                    else:
                        logging.warning(
                            f"{func_name}: Tokenizer lacks convert_tokens_to_ids method. Cannot get ID for [PAD]."
                        )
                        raise ValueError("Cannot get ID for [PAD] token.")

                    # Ensure attributes are set on the tokenizer object itself for consistency
                    setattr(self.tokenizer, "pad_token", "[PAD]")
                    setattr(self.tokenizer, "pad_token_id", pad_token_id)
                    logging.debug(f"{func_name}: Set PAD token with ID: {pad_token_id}")

                except Exception as e_pad:
                    logging.error(
                        f"{func_name}: Failed to add/set PAD token: {e_pad}. Using fallback ID 0.",
                        exc_info=True,
                    )
                    pad_token_id = 0  # Last resort fallback
            else:
                logging.debug(
                    f"{func_name}: Using eos_token_id {pad_token_id} as fallback pad_token_id."
                )
                # Also set the pad_token attributes if using EOS as PAD
                setattr(self.tokenizer, "pad_token_id", pad_token_id)
                if hasattr(self.tokenizer, "eos_token") and not hasattr(
                    self.tokenizer, "pad_token"
                ):
                    setattr(self.tokenizer, "pad_token", self.tokenizer.eos_token)

        self._pad_token_id_internal = pad_token_id  # Store for internal use if needed
        logging.debug(
            f"{func_name}: Final pad_token_id determined: {self._pad_token_id_internal}"
        )

        # --- Configure other attributes ---
        self.model_config = (
            llm_model_config if llm_model_config else getattr(llm_model, "config", None)
        )
        if not self.model_config:
            logging.error(
                f"{func_name}: LLM model configuration not provided or found."
            )
            raise ValueError("LLM model configuration not provided or found.")

        config_hidden_size = None
        if isinstance(self.model_config, dict):
            config_hidden_size = self.model_config.get("hidden_size")
        elif hasattr(self.model_config, "hidden_size"):
            config_hidden_size = self.model_config.hidden_size
        # Use provided hidden_size if valid, else infer, else default
        self.hidden_size = (
            hidden_size
            if isinstance(hidden_size, int) and hidden_size > 0
            else config_hidden_size
        )
        if not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            default_hs = 3072  # Default if inference fails
            logging.warning(
                f"{func_name}: Could not infer valid hidden_size. Using default={default_hs}"
            )
            self.hidden_size = default_hs

        max_pos = None
        if isinstance(self.model_config, dict):
            max_pos = self.model_config.get("max_position_embeddings")
        elif hasattr(self.model_config, "max_position_embeddings"):
            max_pos = self.model_config.max_position_embeddings
        self.max_pos_embeddings = (
            max_pos if isinstance(max_pos, int) and max_pos > 0 else 4096
        )  # Default

        self.chunk_size = min(512, self.max_pos_embeddings)
        if self.chunk_size <= 0:
            self.chunk_size = 512  # Ensure positive
        self.chunk_overlap = max(0.0, min(0.9, 0.25))  # Clamp overlap

        logging.debug(
            f"{func_name}: Initialized - hidden_size={self.hidden_size}, "
            f"max_pos_embeddings={self.max_pos_embeddings}, chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, tokenizer_type={type(self.tokenizer)}"
        )

    def get_observation(self, text: str) -> mx.array:
        """Generates the observation vector from text, handling chunking."""
        func_name = "get_observation"
        # Use float32 consistently for observations
        output_dtype = mx.float32
        zero_obs = mx.zeros(
            (1, self.hidden_size), dtype=output_dtype
        )  # Precompute zero obs

        if not text or not text.strip():
            logging.warning(f"{func_name}: Received empty text, returning zeros.")
            return zero_obs
        try:
            # --- Explicitly check self.tokenizer before calling encode ---
            if not hasattr(self.tokenizer, "encode") or not callable(
                self.tokenizer.encode
            ):
                logging.error(
                    f"{func_name}: CRITICAL - self.tokenizer invalid or lacks 'encode'. Type: {type(self.tokenizer)}"
                )
                raise TypeError(
                    "self.tokenizer does not have a callable 'encode' method."
                )

            logging.debug(f"{func_name}: Encoding text: '{text[:50]}...'")
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            num_tokens = len(input_ids)
            logging.debug(f"{func_name}: Encoded into {num_tokens} tokens.")

            if num_tokens == 0:
                logging.warning(
                    f"{func_name}: Text resulted in zero tokens, returning zeros."
                )
                return zero_obs

            # Decide chunking
            if num_tokens > self.chunk_size:
                logging.debug(
                    f"{func_name}: Using chunked observation for {num_tokens} tokens."
                )
                state = self._chunked_observation(input_ids)
            else:
                logging.debug(
                    f"{func_name}: Using standard observation for {num_tokens} tokens."
                )
                state = self._standard_observation(input_ids)

            # --- Final Validation ---
            if state is None:
                logging.error(
                    f"{func_name}: Observation calculation returned None. Returning zeros."
                )
                return zero_obs
            if not isinstance(state, mx.array):
                logging.error(
                    f"{func_name}: Observation is not mx.array (Type: {type(state)}). Returning zeros."
                )
                return zero_obs
            if state.shape != (1, self.hidden_size):
                logging.error(
                    f"{func_name}: Observation shape mismatch: Got {state.shape}, expected {(1, self.hidden_size)}. Returning zeros."
                )
                return zero_obs
            if state.dtype != output_dtype:
                logging.debug(
                    f"{func_name}: Casting observation from {state.dtype} to {output_dtype}."
                )
                state = state.astype(output_dtype)
            if not mx.all(mx.isfinite(state)):
                logging.error(
                    f"{func_name}: Final observation contains NaN/Inf. Returning zeros."
                )
                return zero_obs

            logging.debug(
                f"{func_name}: Returning valid observation tensor, shape={state.shape}, dtype={state.dtype}"
            )
            return state

        except TypeError as e_type:
            logging.error(
                f"{func_name}: TypeError (likely tokenizer issue): {e_type}",
                exc_info=True,
            )
            return zero_obs
        except Exception as e:
            logging.error(f"{func_name}: Unexpected error: {e}", exc_info=True)
            return zero_obs

    def _standard_observation(self, token_ids: List[int]) -> Optional[mx.array]:
        """Processes a single chunk of tokens and returns the mean hidden state."""
        func_name = "_standard_observation"
        if not token_ids:
            logging.debug(f"{func_name}: Received empty token_ids.")
            return None
        try:
            logging.debug(f"{func_name}: Processing {len(token_ids)} tokens.")
            hidden_states = self._process_tokens(token_ids)  # Returns float32
            if hidden_states is None or hidden_states.size == 0:
                logging.warning(
                    f"{func_name}: _process_tokens returned None or empty array."
                )
                return None

            logging.debug(
                f"{func_name}: Calculating mean over sequence dim (axis=1) from shape {hidden_states.shape}."
            )
            state_mx = mx.mean(hidden_states, axis=1)  # Shape: (1, hidden_size)
            logging.debug(
                f"{func_name}: Mean state calculated, shape={state_mx.shape}."
            )
            # Return as float32 (matches _process_tokens output)
            return state_mx.astype(mx.float32)
        except Exception as e:
            logging.error(f"{func_name}: Error: {e}", exc_info=True)
            return None

    def _chunked_observation(self, all_tokens: List[int]) -> Optional[mx.array]:
        """Processes long sequences by chunking and combining observations."""
        func_name = "_chunked_observation"
        if not all_tokens:
            logging.debug(f"{func_name}: Received empty all_tokens.")
            return None

        output_dtype = mx.float32
        zero_obs = mx.zeros((1, self.hidden_size), dtype=output_dtype)  # Precompute
        chunk_stride = max(1, int(self.chunk_size * (1 - self.chunk_overlap)))
        chunk_observations = []
        weights = []

        logging.debug(
            f"{func_name}: Chunking {len(all_tokens)} tokens with size={self.chunk_size}, stride={chunk_stride}"
        )

        for i in range(0, len(all_tokens), chunk_stride):
            chunk_tokens = all_tokens[i : i + self.chunk_size]
            # Skip very small trailing chunks
            if len(chunk_tokens) < self.chunk_size * 0.1 and len(chunk_tokens) < 20:
                logging.debug(
                    f"{func_name}: Skipping small chunk (size {len(chunk_tokens)}) at index {i}."
                )
                continue
            logging.debug(
                f"{func_name}: Processing chunk {i // chunk_stride + 1} (indices {i} to {i + len(chunk_tokens) - 1})"
            )
            try:
                chunk_state = self._standard_observation(chunk_tokens)
                if (
                    chunk_state is not None
                    and chunk_state.size > 0
                    and chunk_state.shape == (1, self.hidden_size)
                ):
                    chunk_observations.append(chunk_state)
                    # Simple weighting: slightly higher weight for first/last chunks
                    weight = (
                        1.1 if i == 0 or (i + chunk_stride >= len(all_tokens)) else 1.0
                    )
                    weights.append(weight)
                    logging.debug(
                        f"{func_name}:  -> Added chunk observation with weight {weight}."
                    )
                else:
                    logging.warning(
                        f"{func_name}: Processing chunk starting at index {i} yielded invalid observation (None, empty, or wrong shape)."
                    )
            except Exception as e:
                logging.error(
                    f"{func_name}: Error processing chunk starting at index {i}: {e}",
                    exc_info=True,
                )
                continue  # Skip chunk on error

        if not chunk_observations:
            logging.warning(f"{func_name}: Resulted in no valid chunk observations.")
            return zero_obs

        # Combine observations: Weighted average
        try:
            logging.debug(
                f"{func_name}: Combining {len(chunk_observations)} chunk observations."
            )
            # Shape check was done when appending, but double check count vs weights
            if len(chunk_observations) != len(weights):
                logging.warning(
                    f"{func_name}: Mismatch observations ({len(chunk_observations)}) vs weights ({len(weights)}). Using equal weights."
                )
                weights = [1.0] * len(chunk_observations)

            stacked_obs = mx.stack(
                chunk_observations, axis=0
            )  # -> (num_chunks, 1, hidden_size)
            weights_mx = mx.array(weights, dtype=output_dtype).reshape(
                -1, 1, 1
            )  # -> (num_chunks, 1, 1)

            weighted_sum = mx.sum(
                stacked_obs * weights_mx, axis=0
            )  # Sum over chunks -> (1, hidden_size)
            total_weight = sum(weights)

            if total_weight > 1e-6:
                combined_obs = weighted_sum / total_weight
                logging.debug(
                    f"{func_name}: Combined observation using weighted average (total weight={total_weight:.2f})."
                )
            else:
                logging.warning(
                    f"{func_name}: Total weight near zero. Using unweighted mean."
                )
                combined_obs = mx.mean(
                    stacked_obs, axis=0
                )  # Fallback -> (1, hidden_size)

            # Final validation before returning combined obs
            if combined_obs.shape != (1, self.hidden_size) or not mx.all(
                mx.isfinite(combined_obs)
            ):
                logging.error(
                    f"{func_name}: Combined observation has invalid shape {combined_obs.shape} or non-finite values. Returning zeros."
                )
                return zero_obs

            return combined_obs.astype(output_dtype)

        except Exception as e:
            logging.error(
                f"{func_name}: Error combining chunk observations: {e}", exc_info=True
            )
            return zero_obs

    # --- Add the _validate_tensor helper method to the class ---
    def _validate_tensor(
        self, tensor: Optional[mx.array], name: str = "tensor"
    ) -> bool:
        """Internal helper: Validate tensor for NaN/Inf and basic shape."""
        if tensor is None:
            # logging.warning(f"_validate_tensor: {name} is None") # Can be too verbose
            return False
        if not isinstance(tensor, mx.array):
            logging.warning(f"_validate_tensor: {name} type {type(tensor)} != mx.array")
            return False
        # Check for non-finite values only if it's a floating point type
        if mx.issubdtype(tensor.dtype, mx.floating):
            if mx.any(mx.isnan(tensor)).item():
                logging.warning(f"_validate_tensor: {name} contains NaN")
                return False
            if mx.any(mx.isinf(tensor)).item():
                logging.warning(f"_validate_tensor: {name} contains Inf")
                return False
        # Check shape after checking for non-finite values
        if tensor.ndim == 0 or 0 in tensor.shape:
            logging.warning(f"_validate_tensor: {name} invalid shape {tensor.shape}")
            return False
        return True

    def _process_tokens(self, token_ids: List[int]) -> Optional[mx.array]:
        """Processes token IDs through the LLM's embedding and transformer layers."""
        func_name = "_process_tokens"
        # Use float32 for observation consistency and intermediate processing stability
        output_dtype = mx.float32
        zero_output = None  # Define later based on shape

        if not token_ids:
            logging.debug(f"{func_name}: Received empty token_ids.")
            # Cannot determine hidden_size if token_ids is empty, return None
            return None

        # Ensure input is int32 for standard embedding lookup
        input_mx = mx.array([token_ids], dtype=mx.int32)
        zero_output = mx.zeros(
            (1, len(token_ids), self.hidden_size), dtype=output_dtype
        )  # Define zero output shape

        # --- Embedding Lookup ---
        try:
            logging.debug(
                f"{func_name}: Performing embedding lookup for {input_mx.shape} tokens."
            )
            # Dynamically find embedding layer
            embed_layer = None
            if hasattr(self.llm_model, "embed_tokens"):
                embed_layer = self.llm_model.embed_tokens
            elif hasattr(self.llm_model, "language_model.model.embed_tokens"):
                embed_layer = self.llm_model.language_model.model.embed_tokens
            elif hasattr(self.llm_model, "model") and hasattr(
                self.llm_model.model, "embed_tokens"
            ):
                embed_layer = self.llm_model.model.embed_tokens

            if embed_layer is None or not callable(embed_layer):
                embed_layer = self.llm_model.language_model.model.embed_tokens
                # raise AttributeError("Model does not have a recognized callable embedding layer ('embed_tokens' or 'model.embed_tokens').")

            x = embed_layer(input_mx)
            logging.debug(f"{func_name}: Embedding lookup successful, shape={x.shape}.")
        except Exception as e_embed:
            logging.error(
                f"{func_name}: Error during embedding lookup: {e_embed}", exc_info=True
            )
            return zero_output  # Return zeros on embedding error

        # --- Process through Transformer Layers ---
        mask = None  # Typically no mask needed for non-causal observation processing
        hidden_states = x
        if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "layers"):
            logging.debug(
                f"{func_name}: Processing through {len(self.llm_model.model.layers)} transformer layers."
            )
            # Ensure eval mode for consistency if model tracks it
            original_training_state = getattr(self.llm_model, "training", False)
            if original_training_state:
                self.llm_model.eval()
            try:
                for idx, layer in enumerate(self.llm_model.model.layers):
                    logging.debug(
                        f"{func_name}:  Processing layer {idx} ({type(layer).__name__})..."
                    )
                    try:
                        # Try passing mask=None, fallback if it fails
                        hidden_states = layer(hidden_states, mask=mask)
                    except TypeError:  # Catch if layer doesn't accept mask
                        logging.debug(
                            f"{func_name}:   Layer {idx} does not accept mask, calling without it."
                        )
                        hidden_states = layer(hidden_states)
                    except Exception as e_layer:
                        logging.error(
                            f"{func_name}: Error processing layer {idx}: {e_layer}",
                            exc_info=True,
                        )
                        # Return intermediate state on layer error? Or zeros? Intermediate might be better.
                        logging.warning(
                            f"{func_name}: Returning hidden state from before failed layer {idx}."
                        )
                        break  # Stop processing layers
            finally:
                if original_training_state:
                    self.llm_model.train()  # Restore state
        else:
            logging.debug(
                f"{func_name}: LLM model structure lacks 'model.layers'. Skipping layer processing."
            )

        # --- Apply Final Normalization ---
        if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "norm"):
            logging.debug(f"{func_name}: Applying final normalization.")
            try:
                hidden_states = self.llm_model.model.norm(hidden_states)
            except Exception as e_norm:
                logging.error(
                    f"{func_name}: Error applying final normalization: {e_norm}",
                    exc_info=True,
                )
                # Use pre-norm state on error
        else:
            logging.debug(f"{func_name}: LLM model lacks final 'model.norm'.")

        # Final validation and type casting
        if not isinstance(hidden_states, mx.array) or hidden_states.size == 0:
            logging.error(
                f"{func_name}: Invalid hidden_states after processing. Returning zeros."
            )
            return zero_output

        logging.debug(
            f"{func_name}: Final hidden_states shape={hidden_states.shape}. Casting to {output_dtype}."
        )
        # Ensure final output is float32
        return hidden_states.astype(output_dtype)


# --- Dynamic Chunk Selector ---
class DynamicChunkSelector:
    """
    Selects important text chunks based on heuristics or token-level attributions.
    """

    def __init__(
        self, llm_model: Any, tokenizer: Any, hidden_size: Optional[int] = None
    ):
        """Initializes the chunk selector."""
        func_name = "DynamicChunkSelector.__init__"
        logging.debug(f"{func_name}: Initializing...")

        # --- Validate Model ---
        if not isinstance(llm_model, nn.Module):  # Basic check
            logging.warning(
                f"{func_name}: llm_model type is {type(llm_model)}, expected nn.Module subclass."
            )
        if not hasattr(llm_model, "config"):
            raise ValueError(f"{func_name}: llm_model must have a 'config' attribute.")
        self.llm_model = llm_model

        # --- Validate Tokenizer ---
        # Use the same robust validation as EnhancedObservationHandler
        actual_tokenizer = None
        tokenizer_source = "provided explicitly"
        if tokenizer is None:
            tokenizer_source = "inferred from llm_model"
            logging.debug(f"{func_name}: Tokenizer not provided, attempting inference.")
            tokenizer_maybe_wrapped = getattr(llm_model, "tokenizer", None)
            if tokenizer_maybe_wrapped is None:
                raise ValueError("Tokenizer not provided or inferrable.")
            tokenizer_to_check = tokenizer_maybe_wrapped
        else:
            tokenizer_to_check = tokenizer

        if isinstance(tokenizer_to_check, TokenizerWrapper):
            logging.debug(
                f"{func_name}: Tokenizer ({tokenizer_source}) is Wrapper, accessing underlying."
            )
            actual_tokenizer = getattr(tokenizer_to_check, "_tokenizer", None)
            if actual_tokenizer is None:
                raise ValueError("TokenizerWrapper found but '_tokenizer' is None.")
        else:
            actual_tokenizer = tokenizer_to_check
            logging.debug(
                f"{func_name}: Using tokenizer ({tokenizer_source}) directly."
            )

        is_callable_tokenizer = callable(actual_tokenizer)
        has_encode_method = hasattr(actual_tokenizer, "encode") and callable(
            actual_tokenizer.encode
        )
        if not is_callable_tokenizer or not has_encode_method:
            raise TypeError(
                f"{func_name}: Invalid actual tokenizer (Type: {type(actual_tokenizer)}). Must be callable and have 'encode'."
            )
        self.tokenizer = actual_tokenizer
        logging.debug(
            f"{func_name}: Validated tokenizer (Type: {type(self.tokenizer)})"
        )

        # --- Infer Config Values ---
        model_config = llm_model.config
        try:
            # Infer hidden size
            config_hidden_size = getattr(model_config, "hidden_size", None)
            self.hidden_size = (
                hidden_size
                if isinstance(hidden_size, int) and hidden_size > 0
                else config_hidden_size
            )
            if not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
                default_hs = 3072
                logging.warning(
                    f"{func_name}: Invalid hidden_size. Using default={default_hs}"
                )
                self.hidden_size = default_hs

            # Infer vocab size
            config_vocab_size = getattr(model_config, "vocab_size", None)
            if not isinstance(config_vocab_size, int) or config_vocab_size <= 0:
                default_vs = 128260
                # logging.warning(f"{func_name}: Invalid vocab_size from config. Using default={default_vs}")
                self.vocab_size = default_vs
            else:
                self.vocab_size = config_vocab_size

        except AttributeError as e:
            logging.error(
                f"{func_name}: Failed to get required attributes from model config: {e}"
            )
            raise ValueError(
                f"Model config missing required attributes (e.g., hidden_size, vocab_size): {e}"
            )

        # Validate embedding layer existence
        if not hasattr(self.llm_model, "embed_tokens") and not (
            hasattr(self.llm_model, "model")
            and hasattr(self.llm_model.model, "embed_tokens")
        ):
            raise ValueError(
                f"{func_name}: llm_model (or llm_model.model) must have 'embed_tokens'."
            )

        # Constants for chunk selection
        self.MIN_CHUNK_SIZE = 10
        self.MAX_CHUNK_SIZE = 100
        self.MAX_TOTAL_PROPORTION = 0.6
        self.MIN_TOTAL_TOKENS = 50

        logging.debug(
            f"{func_name}: Initialized with hidden_size={self.hidden_size}, vocab_size={self.vocab_size}"
        )

    def _validate_tensor(
        self, tensor: Optional[mx.array], name: str = "tensor"
    ) -> bool:
        """Internal helper: Validate tensor for NaN/Inf and basic shape."""
        if tensor is None:
            logging.warning(f"_validate_tensor: {name} is None")
            return False
        if not isinstance(tensor, mx.array):
            logging.warning(f"_validate_tensor: {name} type {type(tensor)} != mx.array")
            return False
        if mx.any(mx.isnan(tensor)).item():
            logging.warning(f"_validate_tensor: {name} contains NaN")
            return False
        if mx.any(mx.isinf(tensor)).item():
            logging.warning(f"_validate_tensor: {name} contains Inf")
            return False
        if tensor.ndim == 0 or 0 in tensor.shape:
            logging.warning(f"_validate_tensor: {name} invalid shape {tensor.shape}")
            return False
        return True

    def _find_last_special_token_position(self, text: str) -> int:
        """Find the token position after the last special token <|...|>."""
        func_name = "_find_last_special_token_position"
        if not text:
            return 0
        try:
            special_token_pattern = r"<\|.*?\|>"
            matches = list(re.finditer(special_token_pattern, text))
            if not matches:
                logging.debug(f"{func_name}: No special tokens found.")
                return 0

            last_match = matches[-1]
            last_special_token_end_char_pos = last_match.end()
            logging.debug(
                f"{func_name}: Last special token ends at char pos {last_special_token_end_char_pos}."
            )

            # Encode prefix to find corresponding token position
            text_before_last_token = text[:last_special_token_end_char_pos]
            tokens_before_last = self.tokenizer.encode(
                text_before_last_token, add_special_tokens=False
            )  # Use consistent tokenization
            num_tokens = len(tokens_before_last)
            logging.debug(f"{func_name}: Prefix encodes to {num_tokens} tokens.")
            return num_tokens
        except Exception as e:
            logging.error(
                f"{func_name}: Error finding special token position: {e}", exc_info=True
            )
            return 0  # Default to start if error occurs

    # --- Heuristic Chunk Selection ---
    def get_chunks(self, gen_text: str, tgt_text: str) -> List[Tuple[int, int]]:
        """Identifies important chunks using heuristics (position, rare tokens)."""
        func_name = "get_chunks (heuristic)"
        logging.debug(f"{func_name}: Starting heuristic chunk selection.")
        try:
            if not gen_text or not tgt_text:
                logging.debug(
                    f"{func_name}: Empty input texts, returning default small chunk."
                )
                return [(0, min(self.MIN_CHUNK_SIZE, 5))]

            try:
                input_ids_gen = self.tokenizer.encode(
                    gen_text, add_special_tokens=False
                )
                input_ids_tgt = self.tokenizer.encode(
                    tgt_text, add_special_tokens=False
                )
            except Exception as e:
                logging.warning(
                    f"{func_name}: Error tokenizing texts: {e}. Fallback chunk."
                )
                return [(0, self.MIN_CHUNK_SIZE)]

            gen_len_total = len(input_ids_gen)
            start_position = self._find_last_special_token_position(gen_text)
            # Also check target text special tokens? Assume gen_text is primary for now.
            # start_position = max(start_position, self._find_last_special_token_position(tgt_text))
            logging.debug(
                f"{func_name}: Effective start position after special tokens: {start_position}"
            )

            L = gen_len_total - start_position  # Length of content after special tokens

            if L <= 0:
                logging.debug(
                    f"{func_name}: No tokens available after special tokens. Returning empty chunk at start_pos."
                )
                return [(start_position, start_position)]

            # --- Position-Based Chunks ---
            logging.debug(
                f"{func_name}: Getting position-based chunks for effective length L={L}."
            )
            raw_position_chunks = self._get_chunks_by_position(L)
            position_chunks = [
                (s + start_position, e + start_position) for s, e in raw_position_chunks
            ]
            chunks = position_chunks
            logging.debug(f"{func_name}: Initial position chunks (offset): {chunks}")

            # --- Rare Token Chunks ---
            if L > self.MIN_CHUNK_SIZE * 5:
                logging.debug(
                    f"{func_name}: Analyzing rare tokens (L={L} > {self.MIN_CHUNK_SIZE * 5})."
                )
                try:
                    # Only analyze tokens *after* start_position
                    tokens_to_analyze = input_ids_gen[
                        start_position : start_position + L
                    ]
                    if not tokens_to_analyze:
                        raise ValueError("No tokens to analyze for rarity.")

                    token_counts = {}
                    for token in tokens_to_analyze:
                        token_counts[token] = token_counts.get(token, 0) + 1

                    if len(token_counts) > 5:
                        sorted_freqs = sorted(
                            token_counts.items(), key=lambda item: item[1]
                        )
                        num_rare_threshold = max(1, len(sorted_freqs) // 10)
                        rare_tokens = set(
                            t for t, c in sorted_freqs[:num_rare_threshold] if c <= 2
                        )
                        logging.debug(
                            f"{func_name}: Found {len(rare_tokens)} rare tokens (<=2 occurrences, bottom 10%)."
                        )

                        if rare_tokens:
                            # Get indices relative to the start of tokens_to_analyze, then add start_position
                            rare_indices_relative = [
                                i
                                for i, token in enumerate(tokens_to_analyze)
                                if token in rare_tokens
                            ]
                            rare_positions = [
                                idx + start_position for idx in rare_indices_relative
                            ]
                            logging.debug(
                                f"{func_name}: Found {len(rare_positions)} occurrences of rare tokens."
                            )

                            if rare_positions:
                                rare_chunks_potential = []
                                current_group = []
                                last_pos = -self.MIN_CHUNK_SIZE
                                for pos in rare_positions:
                                    if pos > last_pos + (self.MIN_CHUNK_SIZE // 2):
                                        if current_group:
                                            s = max(
                                                start_position,
                                                min(current_group)
                                                - self.MIN_CHUNK_SIZE // 4,
                                            )
                                            e = min(
                                                gen_len_total,
                                                max(current_group)
                                                + 1
                                                + self.MIN_CHUNK_SIZE // 4,
                                            )
                                            if e - s < self.MIN_CHUNK_SIZE:
                                                e = min(
                                                    gen_len_total,
                                                    s + self.MIN_CHUNK_SIZE,
                                                )
                                            if e > s:
                                                rare_chunks_potential.append((s, e))
                                        current_group = [pos]
                                    else:
                                        current_group.append(pos)
                                    last_pos = pos
                                if current_group:  # Add last group
                                    s = max(
                                        start_position,
                                        min(current_group) - self.MIN_CHUNK_SIZE // 4,
                                    )
                                    e = min(
                                        gen_len_total,
                                        max(current_group)
                                        + 1
                                        + self.MIN_CHUNK_SIZE // 4,
                                    )
                                    if e - s < self.MIN_CHUNK_SIZE:
                                        e = min(gen_len_total, s + self.MIN_CHUNK_SIZE)
                                    if e > s:
                                        rare_chunks_potential.append((s, e))

                                if rare_chunks_potential:
                                    logging.debug(
                                        f"{func_name}: Potential rare chunks before merge: {rare_chunks_potential}"
                                    )
                                    rare_chunks_potential.sort()
                                    merged_rare = [rare_chunks_potential[0]]
                                    for cur_s, cur_e in rare_chunks_potential[1:]:
                                        prev_s, prev_e = merged_rare[-1]
                                        if cur_s < prev_e:
                                            merged_rare[-1] = (
                                                prev_s,
                                                max(prev_e, cur_e),
                                            )
                                        else:
                                            merged_rare.append((cur_s, cur_e))
                                    logging.debug(
                                        f"{func_name}: Adding merged rare chunks: {merged_rare}"
                                    )
                                    chunks.extend(merged_rare)
                except Exception as e_rare:
                    logging.warning(
                        f"{func_name}: Error analyzing rare tokens: {e_rare}",
                        exc_info=True,
                    )

            # --- Final Processing ---
            if not chunks:
                logging.warning(
                    f"{func_name}: No chunks generated, fallback to initial chunk."
                )
                return [
                    (
                        start_position,
                        min(start_position + self.MAX_CHUNK_SIZE, gen_len_total),
                    )
                ]

            logging.debug(f"{func_name}: Merging all {len(chunks)} potential chunks.")
            chunks.sort()
            merged_chunks = [chunks[0]]
            for current_start, current_end in chunks[1:]:
                prev_start, prev_end = merged_chunks[-1]
                if current_start <= prev_end:
                    merged_chunks[-1] = (prev_start, max(prev_end, current_end))
                else:
                    merged_chunks.append((current_start, current_end))
            logging.debug(
                f"{func_name}: Merged to {len(merged_chunks)} chunks: {merged_chunks}"
            )

            logging.debug(f"{func_name}: Refining chunks (max size, min size).")
            refined_chunks = []
            for start, end in merged_chunks:
                start = max(
                    start, start_position
                )  # Ensure chunks don't precede start_position
                chunk_len = end - start
                if chunk_len > self.MAX_CHUNK_SIZE:
                    logging.debug(
                        f"{func_name}: Splitting chunk ({start},{end}) len={chunk_len} > max={self.MAX_CHUNK_SIZE}"
                    )
                    step = self.MAX_CHUNK_SIZE
                    overlap = self.MIN_CHUNK_SIZE // 2
                    for i in range(start, end, step - overlap):
                        chunk_end = min(i + step, end)
                        if chunk_end > i:
                            refined_chunks.append((i, chunk_end))
                elif chunk_len >= self.MIN_CHUNK_SIZE:
                    refined_chunks.append((start, end))
                elif chunk_len > 0:
                    logging.debug(
                        f"{func_name}: Padding small chunk ({start},{end}) len={chunk_len} to min={self.MIN_CHUNK_SIZE}"
                    )
                    padded_end = min(gen_len_total, start + self.MIN_CHUNK_SIZE)
                    refined_chunks.append((start, padded_end))
            logging.debug(
                f"{func_name}: Refined to {len(refined_chunks)} chunks: {refined_chunks}"
            )

            logging.debug(f"{func_name}: Limiting total proportion.")
            total_tokens_refined = sum(e - s for s, e in refined_chunks)
            effective_length = gen_len_total - start_position
            max_allowed_tokens = max(
                self.MIN_TOTAL_TOKENS, int(effective_length * self.MAX_TOTAL_PROPORTION)
            )
            logging.debug(
                f"{func_name}: Total tokens={total_tokens_refined}, Effective length={effective_length}, Max allowed={max_allowed_tokens}"
            )

            if total_tokens_refined > max_allowed_tokens:
                logging.warning(
                    f"{func_name}: Chunks exceed proportion ({total_tokens_refined}/{effective_length}). Prioritizing."
                )
                if len(refined_chunks) >= 3:
                    mid_idx = len(refined_chunks) // 2
                    priority_set = {
                        refined_chunks[0],
                        refined_chunks[mid_idx],
                        refined_chunks[-1],
                    }
                    refined_chunks = sorted(list(priority_set))
                elif len(refined_chunks) > 1:
                    refined_chunks = [refined_chunks[0], refined_chunks[-1]]
                logging.debug(
                    f"{func_name}: Prioritized to {len(refined_chunks)} chunks: {refined_chunks}"
                )

            if not refined_chunks:
                logging.warning(f"{func_name}: Final chunk list empty. Fallback.")
                return [
                    (
                        start_position,
                        min(start_position + self.MAX_CHUNK_SIZE, gen_len_total),
                    )
                ]

            logging.debug(f"{func_name}: Selected {len(refined_chunks)} final chunks.")
            return refined_chunks

        except Exception as e:
            logging.error(f"{func_name}: Unhandled error: {e}", exc_info=True)
            try:  # Attempt robust fallback
                start_pos = self._find_last_special_token_position(gen_text)
                gen_len = len(self.tokenizer.encode(gen_text, add_special_tokens=False))
                return [(start_pos, min(start_pos + self.MIN_CHUNK_SIZE, gen_len))]
            except Exception:
                return [(0, self.MIN_CHUNK_SIZE)]  # Absolute fallback

    # --- Attribution-Based Chunk Selection ---
    def get_chunks_with_attribution(
        self,
        gen_text: str,
        tgt_text: str,
        attribution_threshold_ratio: float = 0.1,
        use_abs_value: bool = True,
        target_for_attribution: Optional[int] = None,
        steps: int = 25,
    ) -> List[Tuple[int, int]]:
        """Selects chunks based on token attribution scores (Integrated Gradients)."""
        func_name = "get_chunks_with_attribution"
        logging.debug(f"{func_name}: Starting attribution-based chunk selection.")
        try:
            # --- Find Start Position ---
            start_position = self._find_last_special_token_position(gen_text)
            logging.debug(f"{func_name}: Effective start position: {start_position}")

            # --- Get Attributions for text *after* special tokens ---
            gen_text_processed = ""
            input_ids_full = self.tokenizer.encode(gen_text, add_special_tokens=False)
            if start_position < len(input_ids_full):
                tokens_after_special = input_ids_full[start_position:]
                gen_text_processed = self.tokenizer.decode(
                    tokens_after_special,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            else:
                logging.warning(
                    f"{func_name}: start_position {start_position} >= text length {len(input_ids_full)}. No text to analyze."
                )
                return self.get_chunks(gen_text, tgt_text)  # Fallback

            if not gen_text_processed.strip():
                logging.warning(
                    f"{func_name}: No processable text after special tokens. Falling back."
                )
                return self.get_chunks(gen_text, tgt_text)

            logging.debug(
                f"{func_name}: Getting attributions for processed text: '{gen_text_processed[:50]}...'"
            )
            attributions = self.get_token_attributions(
                text=gen_text_processed,
                target_token_id=target_for_attribution,
                target_position=-1,
                steps=steps,
                baseline_type="zero",
            )

            if not attributions:
                logging.warning(
                    f"{func_name}: Failed to get attributions. Falling back to heuristic."
                )
                return self.get_chunks(gen_text, tgt_text)

            # --- Identify Important Tokens ---
            token_indices_rel, scores = zip(
                *attributions
            )  # Indices relative to gen_text_processed
            scores_np = mx.array(scores)
            eval_scores = mx.abs(scores_np) if use_abs_value else scores_np
            logging.debug(
                f"{func_name}: Attribution scores range (eval): [{mx.min(eval_scores).item():.4f}, {mx.max(eval_scores).item():.4f}]"
            )

            if eval_scores.size == 0:
                logging.warning(f"{func_name}: No attribution scores. Falling back.")
                return self.get_chunks(gen_text, tgt_text)

            sorted_scores = mx.sort(eval_scores)[::-1]
            threshold_idx = min(
                max(0, int(len(sorted_scores) * attribution_threshold_ratio)),
                len(sorted_scores) - 1,
            )
            score_threshold = sorted_scores[threshold_idx].item()

            # Adjust threshold if near zero but scores exist
            if score_threshold < 1e-6 and mx.max(eval_scores).item() > 1e-5:
                min_tokens_select = max(1, len(sorted_scores) // 20)  # Top 5%
                threshold_idx = min(min_tokens_select, len(sorted_scores) - 1)
                score_threshold = sorted_scores[threshold_idx].item()
                logging.debug(
                    f"{func_name}: Adjusted score threshold to {score_threshold:.4f}"
                )

            important_indices_rel_mx = mx.where(eval_scores >= score_threshold)[0]
            important_indices_rel = important_indices_rel_mx.tolist()
            if not important_indices_rel:
                logging.warning(
                    f"{func_name}: No tokens met threshold {score_threshold:.4f}. Falling back."
                )
                return self.get_chunks(gen_text, tgt_text)

            logging.debug(
                f"{func_name}: Identified {len(important_indices_rel)} important relative indices."
            )

            # --- Group Indices into Chunks (Adjusting for start_position) ---
            important_indices_abs = sorted(
                [idx + start_position for idx in important_indices_rel]
            )
            logging.debug(
                f"{func_name}: Absolute important indices (first 10): {important_indices_abs[:10]}"
            )

            chunks = []
            start_idx_grp = important_indices_abs[0]
            end_idx_grp = important_indices_abs[0]
            gen_len_total = len(input_ids_full)  # Use full length for bounds checking

            for i in range(1, len(important_indices_abs)):
                current_idx = important_indices_abs[i]
                # Merge if gap is small
                if current_idx <= end_idx_grp + (self.MIN_CHUNK_SIZE // 4) + 1:
                    end_idx_grp = current_idx
                else:  # End of group, create chunk
                    chunk_s = max(
                        start_position, start_idx_grp - self.MIN_CHUNK_SIZE // 2
                    )
                    chunk_e = min(
                        gen_len_total, end_idx_grp + 1 + self.MIN_CHUNK_SIZE // 2
                    )
                    if chunk_e - chunk_s < self.MIN_CHUNK_SIZE:
                        chunk_e = min(gen_len_total, chunk_s + self.MIN_CHUNK_SIZE)
                    if chunk_e > chunk_s:
                        chunks.append((chunk_s, chunk_e))
                        logging.debug(
                            f"{func_name}: Created chunk from group: ({chunk_s}, {chunk_e})"
                        )
                    start_idx_grp = current_idx  # Start new group
                    end_idx_grp = current_idx

            # Add last group
            chunk_s = max(start_position, start_idx_grp - self.MIN_CHUNK_SIZE // 2)
            chunk_e = min(gen_len_total, end_idx_grp + 1 + self.MIN_CHUNK_SIZE // 2)
            if chunk_e - chunk_s < self.MIN_CHUNK_SIZE:
                chunk_e = min(gen_len_total, chunk_s + self.MIN_CHUNK_SIZE)
            if chunk_e > chunk_s:
                chunks.append((chunk_s, chunk_e))
                logging.debug(
                    f"{func_name}: Created final chunk: ({chunk_s}, {chunk_e})"
                )

            # --- Refine and Merge ---
            if not chunks:
                logging.warning(
                    f"{func_name}: Attribution grouping yielded no chunks. Falling back."
                )
                return self.get_chunks(gen_text, tgt_text)

            logging.debug(
                f"{func_name}: Merging {len(chunks)} attribution-based chunks."
            )
            chunks.sort()
            merged_chunks = [chunks[0]]
            for cur_s, cur_e in chunks[1:]:
                prev_s, prev_e = merged_chunks[-1]
                if cur_s < prev_e:
                    merged_chunks[-1] = (prev_s, max(prev_e, cur_e))
                else:
                    merged_chunks.append((cur_s, cur_e))
            logging.debug(
                f"{func_name}: Merged to {len(merged_chunks)} chunks: {merged_chunks}"
            )

            logging.debug(f"{func_name}: Refining chunks (max size).")
            final_chunks = []
            for start, end in merged_chunks:
                if end - start > self.MAX_CHUNK_SIZE:
                    step = self.MAX_CHUNK_SIZE
                    overlap = self.MIN_CHUNK_SIZE // 2
                    for i in range(start, end, step - overlap):
                        chunk_end = min(i + step, end)
                        if chunk_end > i:
                            final_chunks.append((i, chunk_end))
                elif end > start:
                    final_chunks.append((start, end))
            logging.debug(
                f"{func_name}: Refined to {len(final_chunks)} chunks: {final_chunks}"
            )

            # --- Limit Proportion ---
            logging.debug(f"{func_name}: Limiting total proportion.")
            effective_length = gen_len_total - start_position
            total_tokens = sum(e - s for s, e in final_chunks)
            max_allowed = max(
                self.MIN_TOTAL_TOKENS, int(effective_length * self.MAX_TOTAL_PROPORTION)
            )
            logging.debug(
                f"{func_name}: Total tokens={total_tokens}, Effective length={effective_length}, Max allowed={max_allowed}"
            )

            if total_tokens > max_allowed:
                logging.warning(
                    f"{func_name}: Attrib chunks exceed proportion. Prioritizing."
                )
                if len(final_chunks) >= 3:
                    mid_idx = len(final_chunks) // 2
                    priority_set = {
                        final_chunks[0],
                        final_chunks[mid_idx],
                        final_chunks[-1],
                    }
                    final_chunks = sorted(list(priority_set))
                elif len(final_chunks) > 1:
                    final_chunks = [final_chunks[0], final_chunks[-1]]
                logging.debug(
                    f"{func_name}: Prioritized to {len(final_chunks)} chunks: {final_chunks}"
                )

            if not final_chunks:
                logging.warning(
                    f"{func_name}: Final attribution chunk list empty. Fallback."
                )
                return self.get_chunks(gen_text, tgt_text)

            logging.debug(
                f"{func_name}: Selected {len(final_chunks)} final chunks using attributions."
            )
            return final_chunks

        except Exception as e:
            logging.error(f"{func_name}: Error: {e}", exc_info=True)
            logging.warning(f"{func_name}: Falling back to heuristic due to error.")
            return self.get_chunks(gen_text, tgt_text)
        finally:
            mx.synchronize()

    # --- Perplexity Calculation ---
    def calculate_perplexity(self, input_ids: List[int]) -> float:
        """Calculate perplexity for a sequence of token IDs."""
        func_name = "calculate_perplexity"
        DEFAULT_PERPLEXITY = 10000.0  # Use a higher default to indicate issues clearly
        MIN_PROB = 1e-9  # Not directly used with cross_entropy loss

        if not input_ids or len(input_ids) < 2:
            logging.debug(
                f"{func_name}: Requires >= 2 tokens, got {len(input_ids)}. Returning default perplexity."
            )
            return DEFAULT_PERPLEXITY

        try:
            logging.debug(
                f"{func_name}: Calculating perplexity for sequence length {len(input_ids)}."
            )
            input_tensor = mx.array([input_ids], dtype=mx.int32)

            # Model call should ideally be within stop_gradient if not needed for grads elsewhere
            # with mx.stop_gradient(): # Uncomment if safe to do so
            logging.debug(f"{func_name}: Performing model forward pass.")
            logits = self.llm_model(input_tensor)
            logits = mx.eval(logits)  # Ensure computation
            logging.debug(f"{func_name}: Logits received, shape={logits.shape}.")

            if not self._validate_tensor(logits, f"{func_name}_logits"):
                return DEFAULT_PERPLEXITY
            if (
                logits.ndim != 3
                or logits.shape[0] != 1
                or logits.shape[1] != input_tensor.shape[1]
                or logits.shape[2] != self.vocab_size
            ):
                logging.warning(
                    f"{func_name}: Logits shape mismatch. Expected (1, {input_tensor.shape[1]}, {self.vocab_size}), Got: {logits.shape}"
                )
                return DEFAULT_PERPLEXITY

            # Shift for next-token prediction loss
            shift_logits = logits[:, :-1, :]
            shift_labels = input_tensor[:, 1:]
            logging.debug(
                f"{func_name}: Shifted shapes: Logits={shift_logits.shape}, Labels={shift_labels.shape}"
            )

            if (
                shift_logits.shape[1] != shift_labels.shape[1]
                or shift_logits.shape[1] == 0
            ):
                logging.warning(
                    f"{func_name}: Shape mismatch after shifting or zero length."
                )
                return DEFAULT_PERPLEXITY

            # Calculate cross-entropy loss (average over sequence)
            logging.debug(f"{func_name}: Calculating mean cross-entropy loss.")
            loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
            loss = mx.eval(loss)

            if not self._validate_tensor(loss, f"{func_name}_loss") or loss.size != 1:
                logging.warning(f"{func_name}: Invalid loss calculated: {loss}")
                return DEFAULT_PERPLEXITY

            loss_val = loss.item()
            logging.debug(f"{func_name}: Mean cross-entropy loss: {loss_val:.4f}")

            # Perplexity = exp(loss)
            perplexity = math.exp(loss_val)
            # Clamp to avoid extremely large values, but allow reasonably high ones
            perplexity = min(perplexity, DEFAULT_PERPLEXITY * 10)
            logging.debug(f"{func_name}: Calculated perplexity: {perplexity:.2f}")
            return perplexity

        except Exception as e:
            logging.error(
                f"{func_name}: Error calculating perplexity: {e}", exc_info=True
            )
            return DEFAULT_PERPLEXITY
        finally:
            mx.synchronize()

    # --- Internal Helper for Heuristics ---
    def _get_chunks_by_position(self, L: int) -> List[Tuple[int, int]]:
        """Simple position-based heuristic method."""
        func_name = "_get_chunks_by_position"
        logging.debug(f"{func_name}: Getting chunks for length L={L}.")
        if L <= 0:
            return [(0, 0)]

        chunks = []
        min_len_for_mid = self.MIN_CHUNK_SIZE * 3
        min_len_for_end = self.MIN_CHUNK_SIZE * 4

        # 1. Beginning chunk (always add if L > 0)
        start1 = 0
        end1 = min(self.MIN_CHUNK_SIZE * 2, L)
        chunks.append((start1, end1))
        logging.debug(f"{func_name}: Added beginning chunk: {(start1, end1)}")

        # 2. Midpoint chunk
        if L >= min_len_for_mid:
            mid_point = L // 2
            start2 = max(0, mid_point - self.MIN_CHUNK_SIZE)
            end2 = min(L, mid_point + self.MIN_CHUNK_SIZE)
            # Avoid adding identical or fully contained chunks
            is_new = True
            for s, e in chunks:
                if s <= start2 and e >= end2:
                    is_new = False
                    break  # Contained
                if start2 <= s and end2 >= e:
                    is_new = False
                    break  # Contains existing (skip)
            if is_new and end2 > start2:
                chunks.append((start2, end2))
                logging.debug(f"{func_name}: Added midpoint chunk: {(start2, end2)}")

        # 3. End chunk
        if L >= min_len_for_end:
            start3 = max(0, L - self.MIN_CHUNK_SIZE * 2)
            end3 = L
            is_new = True
            for s, e in chunks:
                if s <= start3 and e >= end3:
                    is_new = False
                    break
                if start3 <= s and end3 >= e:
                    is_new = False
                    break
            if is_new and end3 > start3:
                chunks.append((start3, end3))
                logging.debug(f"{func_name}: Added end chunk: {(start3, end3)}")

        # Merge overlapping/adjacent chunks
        if not chunks:
            return [(0, min(self.MAX_CHUNK_SIZE, L))]
        chunks.sort()
        merged = [chunks[0]]
        for current_start, current_end in chunks[1:]:
            prev_start, prev_end = merged[-1]
            if current_start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, current_end))
            else:
                merged.append((current_start, current_end))
        logging.debug(f"{func_name}: Merged position chunks: {merged}")

        # Ensure max size by splitting
        final_chunks = []
        for start, end in merged:
            if end - start > self.MAX_CHUNK_SIZE:
                logging.debug(f"{func_name}: Splitting large chunk ({start},{end})")
                for i in range(start, end, self.MAX_CHUNK_SIZE):
                    final_chunks.append((i, min(i + self.MAX_CHUNK_SIZE, end)))
            elif end > start:
                final_chunks.append((start, end))

        if not final_chunks:
            return [(0, min(self.MAX_CHUNK_SIZE, L))]
        logging.debug(f"{func_name}: Final position chunks after split: {final_chunks}")
        return final_chunks

    # --- Integrated Gradients Implementation ---
    def get_token_attributions(
        self,
        text: str,
        target_token_id: Optional[int] = None,
        target_position: int = -1,
        baseline_type: str = "zero",
        steps: int = 25,
    ) -> List[Tuple[int, float]]:
        """Compute token-level attribution scores using Integrated Gradients (IG)."""
        func_name = "get_token_attributions"
        logging.debug(
            f"{func_name}: Starting IG for text='{text[:50]}...', target_id={target_token_id}, target_pos={target_position}, steps={steps}"
        )
        if not text:
            logging.warning(f"{func_name}: Empty text.")
            return []

        try:
            # --- 1. Tokenization ---
            input_ids = self.tokenizer.encode(
                text, add_special_tokens=False
            )  # Usually False for internal analysis
            if not input_ids:
                logging.warning(f"{func_name}: Tokenization empty.")
                return []
            input_tensor = mx.array([input_ids], dtype=mx.int32)
            seq_len = input_tensor.shape[1]
            logging.debug(f"{func_name}: Input shape: {input_tensor.shape}")

            # Adjust target position
            adj_target_position = (
                target_position + seq_len if target_position < 0 else target_position
            )
            if not (0 <= adj_target_position < seq_len):
                logging.error(
                    f"{func_name}: Target position {adj_target_position} out of bounds (0-{seq_len-1})."
                )
                return []
            logging.debug(
                f"{func_name}: Adjusted target position: {adj_target_position}"
            )

            # --- 2. Get Input Embeddings ---
            try:
                embed_layer = None
                if hasattr(self.llm_model, "language_model.model.embed_tokens"):
                    embed_layer = self.llm_model.language_model.model.embed_tokens
                if hasattr(self.llm_model, "embed_tokens"):
                    embed_layer = self.llm_model.embed_tokens
                elif hasattr(self.llm_model, "model") and hasattr(
                    self.llm_model.model, "embed_tokens"
                ):
                    embed_layer = self.llm_model.model.embed_tokens
                if embed_layer is None or not callable(embed_layer):
                    raise AttributeError("No valid embedding layer found.")
                input_emb = embed_layer(input_tensor)
                input_emb = mx.eval(input_emb)
                logging.debug(f"{func_name}: Input embeddings shape: {input_emb.shape}")
            except Exception as e:
                print(self.llm_model)
                logging.error(
                    f"{func_name}: Failed getting input embeddings: {e}", exc_info=True
                )
                return []
            if not self._validate_tensor(input_emb, f"{func_name}_InputEmb"):
                return []

            # --- 3. Define Baseline ---
            logging.debug(f"{func_name}: Creating baseline type: {baseline_type}")
            baseline_emb = None
            if baseline_type == "zero":
                baseline_emb = mx.zeros_like(input_emb)
            elif baseline_type == "pad":
                pad_id = getattr(
                    self.tokenizer, "pad_token_id", 0
                )  # Use internal or default 0
                if pad_id is None:
                    pad_id = 0
                    logging.warning(f"{func_name}: PAD token ID is None, using 0.")
                baseline_ids = mx.full((1, seq_len), pad_id, dtype=mx.int32)
                try:
                    baseline_emb = embed_layer(baseline_ids)
                    baseline_emb = mx.eval(baseline_emb)
                except Exception as e:
                    logging.error(
                        f"{func_name}: Failed getting PAD baseline: {e}. Falling back to zero."
                    )
                    baseline_emb = mx.zeros_like(input_emb)
            else:
                logging.warning(
                    f"{func_name}: Unsupported baseline '{baseline_type}'. Using zero."
                )
                baseline_emb = mx.zeros_like(input_emb)
            if not self._validate_tensor(baseline_emb, f"{func_name}_BaselineEmb"):
                return []
            logging.debug(
                f"{func_name}: Baseline embeddings shape: {baseline_emb.shape}"
            )

            # --- 4. Define Forward Function for Grad ---
            def model_forward_for_grad(embeddings: mx.array) -> mx.array:
                logits = self.llm_model(inputs_embeds=embeddings)
                target_logits = logits[
                    :, adj_target_position, :
                ]  # Use adjusted position
                if target_token_id is not None:
                    if not (0 <= target_token_id < self.vocab_size):
                        raise ValueError(f"Target ID {target_token_id} out of vocab.")
                    scalar_output = target_logits[:, target_token_id]
                else:
                    scalar_output = mx.sum(target_logits, axis=-1)
                return scalar_output

            # --- 5. Create Grad Function ---
            grad_fn = mx.grad(model_forward_for_grad, argnums=0)
            logging.debug(f"{func_name}: Gradient function created.")

            # --- 6. Integrated Gradients Calculation ---
            delta = input_emb - baseline_emb
            total_gradients = mx.zeros_like(input_emb)
            logging.debug(f"{func_name}: Starting IG loop with {steps} steps.")
            for i in range(steps + 1):
                alpha = i / steps
                interpolated_emb = baseline_emb + alpha * delta
                interpolated_emb = mx.eval(interpolated_emb)
                try:
                    grads = grad_fn(interpolated_emb)
                    grads = mx.eval(grads)
                    if not self._validate_tensor(grads, f"{func_name}_Grads_Step{i}"):
                        logging.warning(
                            f"{func_name}: Invalid grads at step {i}. Skipping."
                        )
                        continue
                    total_gradients += grads
                except Exception as e:
                    logging.warning(
                        f"{func_name}: Error computing grads step {i}: {e}. Skipping."
                    )
            logging.debug(f"{func_name}: IG loop finished.")

            avg_gradients = total_gradients / (steps + 1)
            integrated_grads = delta * avg_gradients
            integrated_grads = mx.eval(integrated_grads)
            logging.debug(
                f"{func_name}: Integrated gradients calculated, shape={integrated_grads.shape}"
            )
            if not self._validate_tensor(
                integrated_grads, f"{func_name}_IntegratedGrads"
            ):
                return []

            # --- 7. Calculate Attribution Scores ---
            token_attributions = mx.sum(integrated_grads, axis=2)[
                0
            ]  # Sum over hidden dim -> (seq_len,)
            token_attributions = mx.eval(token_attributions)
            attribution_scores = token_attributions.tolist()
            attributions = list(zip(range(seq_len), attribution_scores))
            logging.debug(
                f"{func_name}: Successfully computed {len(attributions)} token attributions."
            )
            return attributions

        except Exception as e:
            logging.error(f"{func_name}: Unhandled error: {e}", exc_info=True)
            return []
        finally:
            mx.synchronize()


# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Setup Mocks (replace with your actual model and tokenizer)
    mock_config = MockModelConfig()
    mock_model = MockLLMModel(mock_config)
    mock_tokenizer = MockTokenizer()

    # Initialize the selector
    selector = DynamicChunkSelector(llm_model=mock_model, tokenizer=mock_tokenizer)

    # Example texts
    generated_text = "The quick brown fox jumps over the lazy dog near the river bank where unusual flowers bloom."
    target_text = "A fast dark fox leaps above a sleepy canine close to the water's edge with rare blossoms."

    print("\n--- Testing Heuristic Chunk Selection (`get_chunks`) ---")
    heuristic_chunks = selector.get_chunks(generated_text, target_text)
    print(f"Heuristic Chunks ({len(heuristic_chunks)}): {heuristic_chunks}")
    for i, (start, end) in enumerate(heuristic_chunks):
        chunk_tokens = selector.tokenizer.encode(generated_text)[start:end]
        print(
            f"  Chunk {i+1} ({start}:{end}): '{selector.tokenizer.decode(chunk_tokens)}'",
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    print("\n--- Testing Perplexity Calculation ---")
    # Ensure model has parameters initialized if needed by MLX backend
    mx.eval(mock_model.parameters())  # Evaluate parameters once

    # Calculate perplexity for the generated text
    gen_ids = mock_tokenizer.encode(generated_text)
    perplexity = selector.calculate_perplexity(gen_ids)
    print(f"Perplexity for generated text: {perplexity:.2f}")

    print(
        "\n--- Testing Attribution-Based Chunk Selection (`get_chunks_with_attribution`) ---"
    )
    # Note: IG requires actual gradients. The mock model's grads will be simplistic.
    # In a real scenario, ensure the model's forward pass and gradients are meaningful.
    attribution_chunks = selector.get_chunks_with_attribution(
        generated_text,
        target_text,
        attribution_threshold_ratio=0.2,  # Select top 20% scoring tokens
        steps=10,  # Fewer steps for quick test
    )
    print(f"Attribution Chunks ({len(attribution_chunks)}): {attribution_chunks}")
    for i, (start, end) in enumerate(attribution_chunks):
        chunk_tokens = selector.tokenizer.encode(generated_text)[start:end]
        print(
            f"  Chunk {i+1} ({start}:{end}): '{selector.tokenizer.decode(chunk_tokens)}'",
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    print("\n--- Testing Edge Case: Short Texts ---")
    short_gen = "Test."
    short_tgt = "Check."
    short_chunks_h = selector.get_chunks(short_gen, short_tgt)
    print(f"Heuristic Chunks (Short): {short_chunks_h}")
    short_chunks_a = selector.get_chunks_with_attribution(short_gen, short_tgt)
    print(f"Attribution Chunks (Short): {short_chunks_a}")

    print("\n--- Testing Edge Case: Empty Texts ---")
    empty_chunks_h = selector.get_chunks("", "")
    print(f"Heuristic Chunks (Empty): {empty_chunks_h}")
    empty_chunks_a = selector.get_chunks_with_attribution("", "")
    print(f"Attribution Chunks (Empty): {empty_chunks_a}")
