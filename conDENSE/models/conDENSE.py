"""conDENSE model definition."""

from typing import Dict, List, Optional, Tuple, Union
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow_probability import layers as tfpl
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from conDENSE.models import masked_autoregressive_flow as MAF


class conDENSE(Model):
    """conDENSE model class."""

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 10,
        window_size: int = 20,
        MAF_n_bijectors: int = 5,
        MAF_hidden_units: List[int] = [64, 64],
        MAF_conditional: bool = True,
        GRU_dim: Optional[int] = 5,
        VAE_hidden_dim: Optional[int] = 32,
        learning_rate: float = 0.001,
        reconstruction_discount: float = 0.1,
        regularization: bool = True,
    ):
        """Initialize conDENSE model.

        Args:
            n_features (int): Dimensionality of the MAF input, equal to the number of features per timestep.
            latent_dim (int): Number of units in the latent representation produced by the VAE.
                Each unit corresponds to a Gaussian distribution parameterized by mean and standard deviation values.
            window_size (int): Number of timesteps in the context window passed to the VAE.
            MAF_n_bijectors (int): Number of bijectors to include in the MAF.
            MAF_hidden_units (List[int]): Hidden units in each MAF bijector.
            MAF_conditional (bool): Flag indicating if MAF receives conditional inputs.
                Set this to False to produce an unconditional MAF. Note that this setting is no longer equivalent to conDENSE.
            GRU_dim (Optional[int]): Number of units in the VAE GRU.
            VAE_hidden_dim (Optional[int]): Number of units in the VAE hidden layer.
            learning_rate (float): Learning rate.
            reconstruction_discount (float): Discount weighting applied to reconstruction error in the cost function.
            regularization (bool): Whether or not to apply regularization to the latent dimension.
                This pushes representations towards the unit Gaussian.
        """
        super(conDENSE, self).__init__()
        self.compile_kwargs = {
            "optimizer": optimizers.legacy.Adam(learning_rate=learning_rate),
            "loss_fn": MeanSquaredError(reduction="none"),
        }

        self.MAF_conditional = MAF_conditional
        if MAF_conditional:
            self.create_GRU_VAE(
                n_features,
                latent_dim,
                window_size,
                GRU_dim,
                VAE_hidden_dim,
                reconstruction_discount,
                regularization,
            )
            MAF_conditional_shape = 2 * latent_dim
        else:
            MAF_conditional_shape = None

        self.maf = MAF.get_MAF_chain(
            n_bijectors=MAF_n_bijectors,
            hidden_units=MAF_hidden_units,
            activation="relu",
            event_shape=n_features,
            conditional=MAF_conditional,
            conditional_event_shape=MAF_conditional_shape,
        )

    def create_GRU_VAE(
        self,
        n_features: int,
        latent_dim: int,
        window_size: int,
        GRU_dim: Optional[int],
        VAE_hidden_dim: Optional[int],
        reconstruction_discount: float,
        regularization: bool,
    ):
        """Create a GRU VAE.

        Args:
            n_features (int): Dimensionality of the MAF input, equal to the number of features per timestep.
            latent_dim (int): Number of units in the latent representation produced by the VAE.
                Each unit corresponds to a Gaussian distribution parameterized by mean and standard deviation values.
            window_size (int): Number of timesteps in the context window passed to the VAE.
            GRU_dim (Optional[int]): Number of units in the VAE GRU.
            VAE_hidden_dim (Optional[int]): Number of units in the VAE hidden layer.
            reconstruction_discount (float): Discount weighting applied to reconstruction error in the cost function.
            regularization (bool): Whether or not to apply regularization to the latent dimension.
                This pushes representations towards the unit Gaussian.
        """
        if regularization:
            latent_layer = tfpl.MultivariateNormalTriL(
                latent_dim,
                activity_regularizer=tfpl.KLDivergenceRegularizer(
                    self._get_latent_prior(
                        latent_dim,
                    )
                ),
            )
        else:
            latent_layer = tfpl.MultivariateNormalTriL(latent_dim)
        self.reconstruction_discount = reconstruction_discount
        self.encoder = tf.keras.Sequential(
            [
                tfkl.InputLayer(
                    input_shape=(window_size, n_features),
                ),
                tfkl.GRU(
                    GRU_dim,
                    return_sequences=True,
                ),
                tfkl.Flatten(),
                tfkl.Dense(
                    VAE_hidden_dim,
                    activation="relu",
                ),
                tfkl.Dense(
                    VAE_hidden_dim / 2,
                    activation="relu",
                ),
                tfkl.Dense(
                    tfpl.MultivariateNormalTriL.params_size(latent_dim),
                    activation=None,
                ),
                latent_layer,
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tfkl.InputLayer(input_shape=(latent_dim)),
                tfkl.Dense(
                    VAE_hidden_dim / 2,
                    activation="relu",
                ),
                tfkl.Dense(
                    VAE_hidden_dim,
                    activation="relu",
                ),
                tfkl.Dense(
                    window_size * n_features,
                    activation="sigmoid",
                ),
                tfkl.Reshape((window_size, n_features)),
            ]
        )

    def predict(self, inputs: Tuple[np.ndarray]) -> tf.Tensor:
        """Calculate one anomaly score per timestep.

        Args:
            inputs (List[np.ndarray]): list of inputs, tensorflow wraps this
            in a tuple.

        Returns:
            tf.Tensor: anomaly scores.
        """
        if self.MAF_conditional:
            maf_input, vae_input = inputs
            encoded = self.encoder(vae_input)
            score = 1 - self.calculate_log_prob(
                maf_input, encoded.mean(), encoded.stddev()
            )
        else:
            maf_input = inputs.astype(np.float32)
            score = 1 - self.calculate_log_prob(maf_input)
        return score.numpy()

    def call(self, inputs: List[np.ndarray]) -> Union[Tuple[tf.Tensor], tf.Tensor]:
        """Call function.

        Args:
            inputs (List[np.ndarray]): list of inputs, tensorflow wraps this
            in a tuple.

        Returns:
            Union[tf.Tensor, Tuple[tf.Tensor]]: if vae is included two tensors are
                returned: log probabilities for each input and reconstructed inputs.
                If vae is not used then just the log probabilities are returned.
        """
        if self.MAF_conditional:
            maf_input = inputs[0][0]
            vae_input = inputs[0][1]
            encoded = self.encoder(vae_input)
            decoded = self.decoder(encoded)
            log_prob = self.calculate_log_prob(
                maf_input, encoded.mean(), encoded.stddev()
            )
            return log_prob, decoded
        else:
            maf_input = inputs
            log_prob = self.calculate_log_prob(maf_input)
            return log_prob

    def compile(self, optimizer: optimizers.Optimizer, loss_fn: Loss):
        """Compile model.

        Args:
            optimizer (optimizers.Optimizer): model optimizer
            loss_fn (Loss): loss function.
        """
        super(conDENSE, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, inputs: List[np.ndarray]) -> Dict[str, tf.Tensor]:
        """Train step.

        inputs (List[np.ndarray]): list of inputs, tensorflow wraps this
            in a tuple.

        Returns:
            Dict[str, tf.Tensor]: loss for training step.
        """
        if self.MAF_conditional:
            vae_input = inputs[0][1]
            with tf.GradientTape() as tape:
                log_prob, reconstruction = self.call(inputs)
                reconstruction = self.reconstruction_discount * self.loss_fn(
                    vae_input, reconstruction
                )
                reconstruction = tf.reduce_mean(reconstruction, axis=1)
                loss = tf.reduce_mean([-log_prob, reconstruction])
                grads = tape.gradient(
                    loss,
                    self.trainable_variables,
                )
            self.optimizer.apply_gradients(
                zip(
                    grads,
                    self.trainable_variables,
                )
            )
            return {"loss": loss}
        else:
            input = inputs[0]
            with tf.GradientTape() as tape:
                log_prob = self.call(input)
                loss = tf.reduce_mean([-log_prob])
                grads = tape.gradient(
                    loss,
                    self.maf.trainable_variables,
                )
            self.optimizer.apply_gradients(
                zip(
                    grads,
                    self.maf.trainable_variables,
                )
            )
            return {"loss": loss}

    def test_step(self, inputs: List[np.ndarray]) -> Dict[str, tf.Tensor]:
        """Test step.

        Args:
            inputs (List[np.ndarray]): list of inputs, tensorflow wraps this
            in a tuple.

        Returns:
            Dict[str, tf.Tensor]: loss for test step.
        """
        if self.MAF_conditional:
            vae_input = inputs[0][1]
            log_prob, reconstruction = self.call(inputs)
            reconstruction = self.reconstruction_discount * self.loss_fn(
                vae_input, reconstruction
            )
            reconstruction = tf.reduce_mean(reconstruction, axis=1)
            loss = tf.reduce_mean([-log_prob, reconstruction])
            return {"loss": loss}
        else:
            input = inputs[0]
            log_prob = self.call(input)
            loss = tf.reduce_mean([-log_prob])
            return {"loss": loss}

    def calculate_log_prob(
        self,
        input: np.ndarray,
        latent_mean: Optional[tf.Tensor] = None,
        latent_std: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Calculate log probability using conDENSE MAF.

        Args:
            input (np.ndarray): input to calculate log prob on
            latent_mean (Optional[tf.tensor]): means from vae latent distributions,
                these will be passed as conditional inputs to the MAF. Set to None
                if not using conditional MAF.
            latent_std (Optional[tf.tensor]): stand deviations from vae latent distributions,
                these will be passed as conditional inputs to the MAF. Set to None
                if not using conditional MAF.

        Returns:
            tf.Tensor: log probabilities for given inputs.
        """
        if self.MAF_conditional:
            conditional = tf.concat((latent_mean, latent_std), axis=1)
            return self.maf.log_prob(
                input,
                bijector_kwargs=self._make_bijector_kwargs(
                    self.maf.bijector, {"maf.": {"conditional_input": conditional}}
                ),
            )
        else:
            return self.maf.log_prob(input)

    def _make_bijector_kwargs(
        self, bijector: tfb.Bijector, name_to_kwargs: Dict[str, dict]
    ) -> Dict[str, dict]:
        """Recursive helper function to create kwargs for bijectors.

        If provided name is a substring of multiple bijector names then each bijector
        with that substring will be assigned the same kwargs.

        Args:
            bijector (tfb.Bijector): bijector (can be a chain of bijectors)
            name_to_kwargs (Dict[str, dict]): dictionary mapping bijector names
                to kwargs

        Returns:
            (Dict[str, dict]): dictionary of kwargs for each bijector.
        """
        if hasattr(bijector, "bijectors"):
            return {
                b.name: self._make_bijector_kwargs(b, name_to_kwargs)
                for b in bijector.bijectors
            }
        else:
            for name_regex, kwargs in name_to_kwargs.items():
                if re.match(name_regex, bijector.name):
                    return kwargs
        return {}

    def _get_latent_prior(self, latent_dim: int) -> tfd.Distribution:
        """Get prior distribution for VAE latent space.

        Args:
            latent_dim (int): size of latent representation

        Returns:
            tfd.Distribution: prior distribution for latent space.
        """
        return tfd.Independent(
            tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
        )
