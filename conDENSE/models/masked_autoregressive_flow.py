"""Module for defining Masked Autoregressive Flow (MAF) bijectors."""
from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd


def get_MAF_bijector(
    event_shape: int = 2,
    hidden_units: List[int] = [64, 64],
    activation: str = "relu",
    bijector_label: str = "masked_autoregressive_flow",
    conditional: bool = False,
    conditional_event_shape: Optional[Tuple] = None,
) -> tfb.Bijector:
    """Create a single Masked Autoregressive Flow bijector.

    Args:
        event_shape (int): Dimensionality of the training data.
        hidden_units (List[int]): Number of neurons in the MADE network.
        activation (str): Activation function to be used in the MADE network.
        bijector_label (str): Label assigned to the created bijector.
        conditional (bool): Flag indicating if MAF receives conditional inputs.
        conditional_event_shape (Optional[Tuple]): Shape of conditional input. If conditional is False, this should be None.

    Returns:
        tfp.bijectors.Bijector: Masked Autoregressive Flow bijector.
    """
    made = tfb.AutoregressiveNetwork(
        params=2,
        event_shape=[event_shape],
        hidden_units=hidden_units,
        activation=activation,
        conditional=conditional,
        conditional_event_shape=conditional_event_shape,
        conditional_input_layers="first_layer",
    )
    return tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made, name=bijector_label)


def get_MAF_chain(
    base_distribution: tfd.Distribution = tfd.Normal(loc=0, scale=1),
    n_bijectors: int = 2,
    event_shape: int = 2,
    hidden_units: List[int] = [64, 64],
    activation: str = "relu",
    conditional: bool = False,
    conditional_event_shape: Optional[Tuple] = None,
) -> tfd.Distribution:
    """Create a chain of Masked Autoregressive Flow (MAF) bijectors.

    Args:
        base_distribution (tfd.Distribution): Base distribution to transform.
        n_bijectors (int): Number of MAF bijectors to chain together.
        event_shape (int): Dimensionality of training data.
        hidden_units (List[int]): Number of neurons in MADE network.
        activation (str): Activation function to be used in MADE network.
        conditional (bool): Flag indicating if MAF receives conditional inputs.
        conditional_event_shape (Optional[Tuple]): Shape of conditional input. If conditional is False, this should be None.
            Shape of conditional input must be (event_shape * n_samples).

    Returns:
        tfd.Distribution: Transformed distribution, base distribution
        transformed by chain of MAF bijectors.
    """
    permutations = list(range(1, event_shape)) + [0]
    bijectors = []
    for i in range(n_bijectors):
        masked_auto_i = get_MAF_bijector(
            event_shape,
            hidden_units=hidden_units,
            activation=activation,
            bijector_label=f"maf_{i}",
            conditional=conditional,
            conditional_event_shape=conditional_event_shape,
        )
        bijectors.extend([masked_auto_i, tfb.Permute(permutation=permutations)])
    # Discard last permute layer when chaining
    flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
    return tfd.TransformedDistribution(
        tfd.Sample(base_distribution, sample_shape=[event_shape]),
        bijector=flow_bijector,
        name="chained_maf",
    )