# -*- coding: utf-8 -*-
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple # Added Tuple
import logging
import math
import types # Added for patching
import os # Added for path checking
from pathlib import Path # Added for path checking

# Assuming utils are in the same directory or accessible via PYTHONPATH
try:
    # Placeholder for utils if not fully defined elsewhere yet
    # Ensure these files exist even if empty for the import to succeed
    if not os.path.exists("llama_rl"): os.makedirs("llama_rl")
    if not os.path.exists("llama_rl/utils.py"): Path("llama_rl/utils.py").touch()
    from llama_rl.utils import patch_llm_model_with_update_shared
except ImportError:
    logging.warning("Could not import patch_llm_model_with_update_shared from llama_rl.utils. Patching functions might fail if called.")
    # Define a dummy function if import fails
    def patch_llm_model_with_update_shared(model):
        logging.error("patch_llm_model_with_update_shared is not implemented (dummy function called).")
        return model
except NameError: # Handle case where os/Path might not be imported yet if utils doesn't exist
     logging.warning("Could not import patch_llm_model_with_update_shared from llama_rl.utils (likely missing file). Patching functions might fail if called.")
     def patch_llm_model_with_update_shared(model):
        logging.error("patch_llm_model_with_update_shared is not implemented (dummy function called).")
        return model


# ======================================================================
# Policy Network Definitions (With LayerNorm added)
# ======================================================================

class ActionPolicyNetwork(nn.Module):
    """MLP to process LLM state and output a new state representation, with LayerNorm."""

    def __init__(
        self,
        input_dim: int = 3072,
        output_dim: int = 3072,
        hidden_dim: int = 3072,
        num_layers: int = 8,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("ActionPolicyNetwork requires at least 1 layer.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers  # Store num_layers for the loop
        self.hidden_dim = hidden_dim  # Store hidden_dim for LayerNorm

        layer_sizes = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        # Initialize Linear layers and register them
        for i, (idim, odim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer = nn.Linear(idim, odim)
            setattr(self, f"layer_{i}", layer)

        # Initialize LayerNorm layers (one for each hidden layer except the last output)
        # and register them
        if num_layers > 1:  # Only add norms if there are hidden layers
            for i in range(num_layers - 1):
                norm_layer = nn.LayerNorm(
                    hidden_dim
                )  # Norm applied to hidden dim output
                setattr(self, f"norm_{i}", norm_layer)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through the network using correctly registered layers and LayerNorm."""
        original_ndim = x.ndim
        b, s = 1, 1 # Defaults for reshaping back
        if original_ndim == 1:
            x = x.reshape(1, -1)  # Ensure batch dimension [1, D]
        elif original_ndim == 3:
            # Input like [Batch, Seq, Dim], process each sequence element independently
            b, s, d = x.shape
            x = x.reshape(b * s, d)
        elif original_ndim != 2:
             raise ValueError(f"ActionPolicyNetwork expects input ndim 1, 2, or 3, got {original_ndim}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dim mismatch ActionPolicy: Expected {self.input_dim}, got {x.shape[-1]}"
            )

        # Iterate through layers
        for i in range(self.num_layers):
            try:
                layer = getattr(self, f"layer_{i}")  # Access Linear layer
            except AttributeError:
                logging.error(f"Layer 'layer_{i}' not found during forward pass!")
                raise RuntimeError(
                    f"Internal error: Layer attribute layer_{i} missing."
                )

            x = layer(x)

            # Apply activation and LayerNorm to all layers except the last one
            if i < self.num_layers - 1:
                x = nn.gelu(x)
                if hasattr(self, f"norm_{i}"): # Check if norm layer exists for this index
                    try:
                        norm_layer = getattr(self, f"norm_{i}")
                        x = norm_layer(x)  # Apply LayerNorm
                    except AttributeError:
                        # This case should ideally be prevented by __init__ logic
                        logging.error(f"LayerNorm 'norm_{i}' attribute exists check failed!")
                        raise RuntimeError(f"Internal error: LayerNorm attribute norm_{i} missing.")
                # If num_layers == 1, no norm layer exists, skip norm application

        # Final output shape check
        if x.shape[-1] != self.output_dim:
            raise RuntimeError(
                f"Output dim mismatch ActionPolicy after loop: Expected {self.output_dim}, got {x.shape[-1]}"
            )

        # Reshape back if original input was 3D
        if original_ndim == 3:
            x = x.reshape(b, s, self.output_dim)
        # Note: If input was 1D, output remains 2D [1, D]. If input was 2D [B,D], output is [B,D]

        return x


class TokenPolicyNetwork(nn.Module):
    """MLP head for reconstructing LLM hidden states, with LayerNorm."""

    def __init__(
        self,
        input_dim: int = 3072,
        output_dim: int = 3072,
        hidden_dim: int = 3072,
        num_layers: int = 8,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("TokenPolicyNetwork requires at least 1 layer.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers  # Store num_layers for the loop
        self.hidden_dim = hidden_dim  # Store hidden_dim for LayerNorm

        layer_sizes = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        # Initialize Linear layers and register them
        for i, (idim, odim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer = nn.Linear(idim, odim)
            setattr(self, f"layer_{i}", layer)

        # Initialize LayerNorm layers and register them
        if num_layers > 1:
            for i in range(num_layers - 1):
                norm_layer = nn.LayerNorm(hidden_dim)
                setattr(self, f"norm_{i}", norm_layer)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass, handling potential sequence inputs using registered layers and LayerNorm."""
        original_ndim = x.ndim
        b, s = 1, 1 # Defaults for reshaping back
        if original_ndim == 1:
            x = x.reshape(1, -1)  # Ensure batch dimension [1, D]
        elif original_ndim == 3:
            b, s, d = x.shape
            x = x.reshape(b * s, d) # Flatten sequence for MLP processing
        elif original_ndim != 2:
             raise ValueError(f"TokenPolicyNetwork expects input ndim 1, 2, or 3, got {original_ndim}")


        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dim mismatch TokenPolicy: Expected {self.input_dim}, got {x.shape[-1]}"
            )

        # Iterate through layers
        for i in range(self.num_layers):
            try:
                layer = getattr(self, f"layer_{i}")  # Access Linear layer
            except AttributeError:
                logging.error(f"Layer 'layer_{i}' not found during forward pass!")
                raise RuntimeError(
                    f"Internal error: Layer attribute layer_{i} missing."
                )

            x = layer(x)

            # Apply activation and LayerNorm to all layers except the last one
            if i < self.num_layers - 1:
                x = nn.gelu(x)
                if hasattr(self, f"norm_{i}"):
                    try:
                        norm_layer = getattr(self, f"norm_{i}")
                        x = norm_layer(x)  # Apply LayerNorm
                    except AttributeError:
                        logging.error(f"LayerNorm 'norm_{i}' not found!")
                        raise RuntimeError(f"Internal error: LayerNorm attribute norm_{i} missing.")


        # Final output shape check
        if x.shape[-1] != self.output_dim:
            raise RuntimeError(
                f"Output dim mismatch TokenPolicy after loop: Expected {self.output_dim}, got {x.shape[-1]}"
            )

        # Reshape back if original input was 3D
        if original_ndim == 3:
            x = x.reshape(b, s, self.output_dim)
        # Note: If input was 1D, output remains 2D [1, D]. If input was 2D [B,D], output is [B,D]

        return x


# ======================================================================
# Attention Module - Compatible with existing framework
# ======================================================================
class SelfAttention(nn.Module):
    """Self-attention module with update_shared support."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0: raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout # Note: Dropout is often inactive during eval

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass. Expects input shape [Batch, SeqLen, Dim]."""
        if x.ndim != 3:
             # If input is 2D [Batch, Dim], add SeqLen=1 dimension
             if x.ndim == 2:
                 x = x.reshape(x.shape[0], 1, x.shape[1])
             else:
                 raise ValueError(f"SelfAttention expects input ndim 2 or 3, got {x.ndim}")

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: [Batch, SeqLen, Heads, HeadDim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention scores
        attn_weights = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale # [B, H, S, S]

        # Apply mask if provided
        if mask is not None: attn_weights = attn_weights + mask

        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Apply attention to values
        attn_output = mx.matmul(attn_weights, v) # [B, H, S, HeadDim]

        # Transpose back and reshape: [B, S, Dim]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)

        # Final projection
        output = self.out_proj(attn_output) # [B, S, Dim]

        return output

    def update_shared(self, params):
        self.update(params)


class AttentionPolicyNetwork(nn.Module):
    """Attention-based policy network that handles 2D or 3D input."""
    def __init__(
        self,
        input_dim: int = 3072,
        output_dim: int = 3072,
        hidden_dim: int = 3072,
        num_layers: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        if num_layers < 1: raise ValueError("Requires at least 1 layer.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.input_proj = None
        if input_dim != hidden_dim: self.input_proj = nn.Linear(input_dim, hidden_dim)

        for i in range(num_layers):
            setattr(self, f"attn_{i}", SelfAttention(hidden_dim, num_heads))
            setattr(self, f"norm1_{i}", nn.LayerNorm(hidden_dim))
            setattr(self, f"ff_linear1_{i}", nn.Linear(hidden_dim, hidden_dim * 4))
            setattr(self, f"ff_linear2_{i}", nn.Linear(hidden_dim * 4, hidden_dim))
            setattr(self, f"norm2_{i}", nn.LayerNorm(hidden_dim))

        self.output_proj = None
        if hidden_dim != output_dim: self.output_proj = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass handling [Batch, Dim] or [Batch, Seq, Dim] input."""
        # Check input dimension and potentially add sequence dimension
        original_ndim = x.ndim
        if original_ndim == 2: # Input is [Batch, Dim]
            # Add sequence dimension: [Batch, Seq=1, Dim]
            x = x.reshape(x.shape[0], 1, x.shape[1])
        elif original_ndim != 3: # Expects 3D input otherwise
            raise ValueError(f"AttentionPolicyNetwork expects input ndim 2 or 3, got {original_ndim}")

        # Now x is guaranteed to be 3D: [Batch, Seq, Dim]
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Input dim mismatch: Expected {self.input_dim}, got {x.shape[-1]}")

        # Apply input projection if needed
        if self.input_proj is not None: x = self.input_proj(x) # Shape becomes [B, S, HiddenDim]

        # Process through transformer layers (Pre-Norm)
        for i in range(self.num_layers):
            try:
                # --- Attention Block ---
                norm1 = getattr(self, f"norm1_{i}")
                normed_x1 = norm1(x) # [B, S, HiddenDim]
                attn = getattr(self, f"attn_{i}")
                attn_output = attn(normed_x1) # Input/Output: [B, S, HiddenDim]
                x = x + attn_output # Residual

                # --- FFN Block ---
                norm2 = getattr(self, f"norm2_{i}")
                normed_x2 = norm2(x) # [B, S, HiddenDim]
                ff_linear1 = getattr(self, f"ff_linear1_{i}")
                ff_linear2 = getattr(self, f"ff_linear2_{i}")
                ff_output = ff_linear2(nn.gelu(ff_linear1(normed_x2))) # [B, S, HiddenDim]
                x = x + ff_output # Residual
            except AttributeError as e:
                logging.error(f"Layer component not found during forward pass: {e}")
                raise RuntimeError(f"Internal error: Component missing - {e}")

        # Apply output projection if needed
        if self.output_proj is not None: x = self.output_proj(x) # Shape becomes [B, S, OutputDim]

        # Final output shape check
        if x.shape[-1] != self.output_dim:
            raise RuntimeError(f"Output dim mismatch: Expected {self.output_dim}, got {x.shape[-1]}")

        # If the original input was 2D, remove the sequence dimension before returning
        if original_ndim == 2:
            x = x.squeeze(1) # [Batch, OutputDim]

        # Return [B, S, OutputDim] or [B, OutputDim]
        return x

    def update_shared(self, params):
        """Updates parameters in-place."""
        try:
            if self.input_proj is not None and "input_proj" in params: self.input_proj.update(params["input_proj"])
            for i in range(self.num_layers):
                if f"attn_{i}" in params: getattr(self, f"attn_{i}").update_shared(params[f"attn_{i}"])
                if f"norm1_{i}" in params: getattr(self, f"norm1_{i}").update(params[f"norm1_{i}"])
                if f"ff_linear1_{i}" in params: getattr(self, f"ff_linear1_{i}").update(params[f"ff_linear1_{i}"])
                if f"ff_linear2_{i}" in params: getattr(self, f"ff_linear2_{i}").update(params[f"ff_linear2_{i}"])
                if f"norm2_{i}" in params: getattr(self, f"norm2_{i}").update(params[f"norm2_{i}"])
            if self.output_proj is not None and "output_proj" in params: self.output_proj.update(params["output_proj"])
        except Exception as e: logging.error(f"Error in AttentionPolicy update_shared: {e}", exc_info=True); raise


class AttentionTokenPolicyNetwork(nn.Module):
    """Attention-based token policy network."""
    def __init__(
        self,
        input_dim: int = 3072,
        output_dim: int = 3072, # e.g., vocab_size
        hidden_dim: int = 3072,
        num_layers: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        if num_layers < 1: raise ValueError("Requires at least 1 layer.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.input_proj = None
        if input_dim != hidden_dim: self.input_proj = nn.Linear(input_dim, hidden_dim)

        for i in range(num_layers):
            setattr(self, f"attn_{i}", SelfAttention(hidden_dim, num_heads))
            setattr(self, f"norm1_{i}", nn.LayerNorm(hidden_dim))
            setattr(self, f"ff_linear1_{i}", nn.Linear(hidden_dim, hidden_dim * 4))
            setattr(self, f"ff_linear2_{i}", nn.Linear(hidden_dim * 4, hidden_dim))
            setattr(self, f"norm2_{i}", nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass handling [Batch, Seq, Dim] or [Batch, Dim] input."""
        original_ndim = x.ndim
        if original_ndim == 2: # Input is [Batch, Dim]
            x = x.reshape(x.shape[0], 1, x.shape[1]) # Add Seq=1: [Batch, 1, Dim]
        elif original_ndim != 3:
            raise ValueError(f"AttentionTokenPolicy expects input ndim 2 or 3, got {original_ndim}")

        # Now x is guaranteed to be 3D: [Batch, Seq, Dim]
        if x.shape[-1] != self.input_dim: raise ValueError(f"Input dim mismatch: Expected {self.input_dim}, got {x.shape[-1]}")

        if self.input_proj is not None: x = self.input_proj(x)

        for i in range(self.num_layers):
            try:
                norm1 = getattr(self, f"norm1_{i}")
                normed_x1 = norm1(x)
                attn = getattr(self, f"attn_{i}")
                x = x + attn(normed_x1) # Residual

                norm2 = getattr(self, f"norm2_{i}")
                normed_x2 = norm2(x)
                ff_linear1 = getattr(self, f"ff_linear1_{i}")
                ff_linear2 = getattr(self, f"ff_linear2_{i}")
                x = x + ff_linear2(nn.gelu(ff_linear1(normed_x2))) # Residual
            except AttributeError as e:
                logging.error(f"Layer component not found: {e}")
                raise RuntimeError(f"Internal error: Component missing - {e}")

        x = self.output_proj(x) # Final projection to output_dim: [B, S, OutputDim]

        if x.shape[-1] != self.output_dim: raise RuntimeError(f"Output dim mismatch: Expected {self.output_dim}, got {x.shape[-1]}")

        # Squeeze seq dim if original input was 2D
        if original_ndim == 2:
            x = x.squeeze(1) # [B, OutputDim]

        return x # Return [B, S, OutputDim] or [B, OutputDim]

    def update_shared(self, params):
        """Updates parameters in-place."""
        try:
            if self.input_proj is not None and "input_proj" in params: self.input_proj.update(params["input_proj"])
            for i in range(self.num_layers):
                if f"attn_{i}" in params: getattr(self, f"attn_{i}").update_shared(params[f"attn_{i}"])
                if f"norm1_{i}" in params: getattr(self, f"norm1_{i}").update(params[f"norm1_{i}"])
                if f"ff_linear1_{i}" in params: getattr(self, f"ff_linear1_{i}").update(params[f"ff_linear1_{i}"])
                if f"ff_linear2_{i}" in params: getattr(self, f"ff_linear2_{i}").update(params[f"ff_linear2_{i}"])
                if f"norm2_{i}" in params: getattr(self, f"norm2_{i}").update(params[f"norm2_{i}"])
            if self.output_proj is not None and "output_proj" in params: self.output_proj.update(params["output_proj"])
        except Exception as e: logging.error(f"Error in AttentionTokenPolicy update_shared: {e}", exc_info=True); raise


class LinearWithShared(nn.Linear):
    """Extended Linear layer that includes the update_shared method."""
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__(input_dim, output_dim, bias)

    def update_shared(self, params):
        self.update(params)


class AttentionActionHead(nn.Module):
    """Action head with attention."""
    def __init__(self, input_dim: int, action_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.attention = SelfAttention(input_dim, num_heads)
        self.norm = nn.LayerNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, action_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. Expects [Batch, Seq, Dim] or [Batch, Dim] input."""
        original_ndim = x.ndim
        if original_ndim == 2: # Input is [Batch, Dim]
             x = x.reshape(x.shape[0], 1, x.shape[1]) # Add Seq=1: [Batch, 1, Dim]
        elif original_ndim != 3:
             raise ValueError(f"AttentionActionHead expects input ndim 2 or 3, got {original_ndim}")

        # Now x is [Batch, Seq, Dim]
        if x.shape[-1] != self.input_dim: raise ValueError(f"Input dim mismatch: Expected {self.input_dim}, got {x.shape[-1]}")

        x_norm = self.norm(x)
        attn_output = self.attention(x_norm) # [B, S, Dim]
        # Optional residual: x = x + attn_output
        action_logits = self.output_proj(attn_output) # [B, S, ActionDim]

        # Squeeze seq dim if original input was 2D
        if original_ndim == 2:
             action_logits = action_logits.squeeze(1) # [B, ActionDim]

        return action_logits # Return [B, S, ActionDim] or [B, ActionDim]

    def update_shared(self, params):
        """Updates parameters in-place."""
        try:
            if "attention" in params: self.attention.update_shared(params["attention"])
            if "norm" in params: self.norm.update(params["norm"])
            if "output_proj" in params: self.output_proj.update(params["output_proj"])
        except Exception as e: logging.error(f"Error in AttentionActionHead update_shared: {e}", exc_info=True); raise


# ======================================================================
# Value Network (Critic) Definition
# ======================================================================
class ValueNetwork(nn.Module):
    """ Simple MLP Critic Network """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        if num_layers < 1: raise ValueError("ValueNetwork requires at least 1 layer.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layer_sizes = [input_dim] + [hidden_dim] * (num_layers) # Hidden layers + final hidden

        self.layers = []
        current_dim = input_dim
        for i in range(num_layers):
            layer = nn.Linear(current_dim, hidden_dim)
            self.layers.append(layer)
            setattr(self, f"fc_{i}", layer) # Register layer
            self.layers.append(nn.ReLU())
            current_dim = hidden_dim

        # Final layer to output a single value
        final_layer = nn.Linear(hidden_dim, 1)
        self.layers.append(final_layer)
        setattr(self, "fc_final", final_layer) # Register final layer

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass for the critic. Handles [Batch, Dim] or [Batch, Seq, Dim]."""
        original_ndim = x.ndim
        b, s = 1, 1 # Defaults for reshaping back
        if original_ndim == 1:
            x = x.reshape(1, -1) # Add batch dim: [1, D]
        elif original_ndim == 3:
            b, s, d = x.shape
            x = x.reshape(b * s, d) # Flatten sequence: [B*S, D]
        elif original_ndim != 2: # Input is [B, D]
            raise ValueError(f"ValueNetwork expects ndim 1, 2 or 3, got {original_ndim}")

        if x.shape[-1] != self.input_dim: raise ValueError(f"Input dim mismatch ValueNetwork: Expected {self.input_dim}, got {x.shape[-1]}")

        # Explicitly call registered layers
        current_out = x
        for i in range(self.num_layers):
             layer = getattr(self, f"fc_{i}")
             current_out = nn.relu(layer(current_out))
        final_layer = getattr(self, "fc_final")
        value = final_layer(current_out) # [B*S, 1] or [B, 1]

        # Reshape back if input was 3D
        if original_ndim == 3:
             value = value.reshape(b, s, 1) # [B, S, 1]
        # If input was 1D, output is [1, 1]. If 2D, output is [B, 1].

        return value # Shape: [B, 1] or [B, S, 1]

    def update_shared(self, params):
        """Updates parameters in-place (standard update)."""
        self.update(params)


# ======================================================================
# Initialization Helper - Incorporating Critic
# ======================================================================

# Function to create action head (handles attention vs linear)
def create_action_head(
    input_dim: int, action_dim: int, use_attention: bool = False, num_heads: int = 4
) -> nn.Module:
    """Creates an action head, potentially using attention."""
    if use_attention:
        logging.debug(f"Creating AttentionActionHead ({input_dim=}, {action_dim=}, {num_heads=})")
        return AttentionActionHead(input_dim, action_dim, num_heads)
    else:
        logging.debug(f"Creating Linear Action Head ({input_dim=}, {action_dim=})")
        return LinearWithShared(input_dim, action_dim)


def create_attention_policy_networks(
    state_dim: int,
    action_dim: int,
    vocab_size: int, # Keep vocab_size for token policy if used
    hidden_dim: Optional[int] = None,
    num_layers: int = 4,
    num_heads: int = 8,
    use_attention_head: bool = False, # Option for action head type
) -> Dict[str, nn.Module]:
    """Creates and returns attention-based Actor and Critic network components."""
    if hidden_dim is None: hidden_dim = state_dim
    logging.debug(f"Creating Attention Networks: state={state_dim}, action={action_dim}, hidden={hidden_dim}, layers={num_layers}, heads={num_heads}")

    # Actor Base (Policy Network: state -> transformed state)
    # Input: [B, S, StateDim] or [B, StateDim], Output: [B, S, HiddenDim] or [B, HiddenDim]
    policy_network = AttentionPolicyNetwork(
        input_dim=state_dim, output_dim=hidden_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, num_heads=num_heads
    )

    # Actor Head (transformed state -> action logits)
    # Input: [B, S, HiddenDim] or [B, HiddenDim], Output: [B, S, ActionDim] or [B, ActionDim]
    action_head = create_action_head(
         input_dim=hidden_dim, action_dim=action_dim,
         use_attention=use_attention_head, num_heads=num_heads // 2
    )

    # Token policy (optional, transformed state -> vocab size/reconstruction)
    # Input: [B, S, HiddenDim] or [B, HiddenDim], Output: [B, S, VocabSize] or [B, VocabSize]
    token_policy = AttentionTokenPolicyNetwork(
        input_dim=hidden_dim, output_dim=vocab_size, hidden_dim=hidden_dim,
        num_layers=max(1, num_layers // 2), num_heads=num_heads
    )

    # Critic Network (Value Network: original state -> value)
    # Input: [B, S, StateDim] or [B, StateDim], Output: [B, S, 1] or [B, 1]
    critic_num_layers = max(1, num_layers // 2)
    value_network = ValueNetwork(
        input_dim=state_dim, hidden_dim=hidden_dim, num_layers=critic_num_layers
    )
    logging.debug(f"Created ValueNetwork with {critic_num_layers} layers.")

    return {
        "policy_network": policy_network, # Actor Base
        "action_head": action_head,       # Actor Head
        "token_policy": token_policy,     # Optional Token Policy
        "value_network": value_network,   # Critic
    }


# ======================================================================
# Patching Functions
# ======================================================================
def patch_linear_with_update_shared(linear_instance):
    """Monkey patches an existing Linear layer with update_shared method."""
    if not hasattr(linear_instance, "update_shared"):
        def update_shared_method(self, params): self.update(params)
        linear_instance.update_shared = types.MethodType(update_shared_method, linear_instance)
    return linear_instance

def patch_all_action_heads(agent):
    """Patches action head instances in GRPOAgent if they are standard nn.Linear."""
    if hasattr(agent, "action_head") and isinstance(agent.action_head, nn.Linear) and not isinstance(agent.action_head, LinearWithShared):
        agent.action_head = patch_linear_with_update_shared(agent.action_head)
        logging.debug("Patched main action_head with update_shared")
    if hasattr(agent, "action_head_old") and isinstance(agent.action_head_old, nn.Linear) and not isinstance(agent.action_head_old, LinearWithShared):
        agent.action_head_old = patch_linear_with_update_shared(agent.action_head_old)
        logging.debug("Patched action_head_old with update_shared")
    return agent

# ======================================================================
# Usage Example (Updated)
# ======================================================================
def example_usage(state_dim=3072, action_dim=135, vocab_size=32000):
    """Example of how to use the attention-based actor-critic networks."""

    print("\n--- Running Network Example Usage ---")
    networks = create_attention_policy_networks(
        state_dim=state_dim, action_dim=action_dim, vocab_size=vocab_size,
        hidden_dim=state_dim // 2, num_layers=2, num_heads=4, use_attention_head=True
    )

    policy_network = networks["policy_network"]
    action_head = networks["action_head"]
    token_policy = networks["token_policy"]
    value_network = networks["value_network"]

    # Example 1: Single state input (like in select_action)
    print("\nTesting with Single State Input (Batch=1, Seq=1):")
    single_state = mx.random.normal((state_dim,))
    single_state_batch = single_state.reshape(1, state_dim) # [1, D]
    print(f"Input State Shape: {single_state_batch.shape}")

    intermediate_state_single = policy_network(single_state_batch) # Should output [1, HiddenDim]
    action_logits_single = action_head(intermediate_state_single)  # Should output [1, ActionDim]
    value_prediction_single = value_network(single_state_batch) # Should output [1, 1]
    token_logits_single = token_policy(intermediate_state_single) # Should output [1, VocabSize]

    print("\nOutput Shapes (Single State):")
    print(f"  Intermediate State: {intermediate_state_single.shape}")
    print(f"  Action Logits: {action_logits_single.shape}")
    print(f"  Value Prediction: {value_prediction_single.shape}")
    print(f"  Token Logits: {token_logits_single.shape}")

    # Example 2: Batched sequence input (like in updates)
    print("\nTesting with Batched Sequence Input (Batch=2, Seq=3):")
    dummy_state_seq = mx.random.normal((2, 3, state_dim))
    print(f"Input State Shape: {dummy_state_seq.shape}")

    intermediate_state_seq = policy_network(dummy_state_seq) # Should output [2, 3, HiddenDim]
    action_logits_seq = action_head(intermediate_state_seq)  # Should output [2, 3, ActionDim]
    value_prediction_seq = value_network(dummy_state_seq) # Should output [2, 3, 1]
    token_logits_seq = token_policy(intermediate_state_seq) # Should output [2, 3, VocabSize]

    print("\nOutput Shapes (Batched Sequence):")
    print(f"  Intermediate State: {intermediate_state_seq.shape}")
    print(f"  Action Logits: {action_logits_seq.shape}")
    print(f"  Value Prediction: {value_prediction_seq.shape}")
    print(f"  Token Logits: {token_logits_seq.shape}")
    print("--- End Network Example Usage ---\n")

    return {
        "intermediate_state_shape_single": intermediate_state_single.shape,
        "action_logits_shape_single": action_logits_single.shape,
        "value_prediction_shape_single": value_prediction_single.shape,
        "intermediate_state_shape_seq": intermediate_state_seq.shape,
        "action_logits_shape_seq": action_logits_seq.shape,
        "value_prediction_shape_seq": value_prediction_seq.shape,
    }

# Example call (optional, for testing)
# if __name__ == "__main__":
#      example_usage()
