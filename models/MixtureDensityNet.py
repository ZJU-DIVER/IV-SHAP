import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate, Embedding
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
from tensorflow.keras.initializers import RandomNormal
import numpy as np

class MixtureDensityNetwork:
    def __init__(self, num_components, input_dim, num_discrete_features, category_counts, embedding_dim):
        self.num_components = num_components
        self.input_dim = input_dim
        self.num_discrete_features = num_discrete_features
        self.category_counts = category_counts  # List of category counts for each discrete feature
        self.embedding_dim = embedding_dim
        self.model = self._build_model()

    def _slice_parameter_vectors(self, parameter_vector):
        """Return components of mean, variance, and weights for the mixture distribution"""
        mus = parameter_vector[:, :self.num_components]
        sigs = tf.exp(parameter_vector[:, self.num_components:2*self.num_components])+1e-6
        pis = tf.nn.softmax(parameter_vector[:, 2*self.num_components:], axis=1)
        return mus, sigs, pis

    def _mdn_loss(self, y_true, y_pred):
        """Custom loss function for MDN"""
        mus, sigs, pis = self._slice_parameter_vectors(y_pred)
        # Create normal distribution
        normal_dist = tf.distributions.Normal(mus, sigs)
        # Compute probability density
        pdfs = normal_dist.prob(tf.tile(y_true, [1, num_components]))
        #pdfs = tfp.distributions.Normal(mus, sigs).prob(tf.tile(y_true, [1, self.num_components]))
        weighted_pdfs = pis * pdfs
        loss = -tf.math.log(tf.reduce_sum(weighted_pdfs, axis=1)+1e-6)
        return tf.reduce_mean(loss)

    def _build_model(self):
        """Build the MDN model"""
        output_dim = self.num_components * 3
        # Continuous input
        inputs_continuous = Input(shape=(self.input_dim,))
        # Discrete input
        inputs_discrete = [Input(shape=(1,)) for _ in range(self.num_discrete_features)]
        embeddings = [Embedding(input_dim=self.category_counts[i], output_dim=self.embedding_dim)(inputs_discrete[i]) for i in range(self.num_discrete_features)]
        flattened = [Flatten()(embeddings[i]) for i in range(self.num_discrete_features)]
        concatenated = concatenate(flattened + [inputs_continuous])
        # Create a normal distribution initializer with a standard deviation of 0.01
        initializer = RandomNormal(mean=0.0, stddev=0.01)
        hidden = Dense(128, activation='relu', kernel_initializer=initializer)(concatenated)
        hidden = Dense(128, activation='relu')(hidden)
        output = Dense(output_dim, activation=None)(hidden)
        model = Model(inputs=inputs_discrete + [inputs_continuous], outputs=output)
        return model

    def compile(self,learning_rate=0.001):
        """Compile the MDN model"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=self._mdn_loss)

    def fit(self, X_train, Y_train, epochs=200, batch_size=128):
        """Train the MDN model"""
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        """Prediction function"""
        return self.model.predict(X_test)
    
    def sample_from_mdn_output(self, mdn_output, num_samples):
        """Generate sample values based on the distribution parameters output by the MDN"""
        num_components = mdn_output.shape[1] // 3
        mus = mdn_output[:, :num_components]
        sigs_raw = mdn_output[:, num_components:2*num_components]
        pis_raw = mdn_output[:, 2*num_components:]

        # Apply exp and softmax
        sigs = tf.exp(sigs_raw)
        pis = tf.nn.softmax(pis_raw, axis=1)
    
        # Create Gaussian mixture distribution
        cat = tfp.distributions.Categorical(probs=pis)
        components = [tfp.distributions.Normal(loc=mu, scale=sigma) for mu, sigma in zip(tf.unstack(mus, axis=1), tf.unstack(sigs, axis=1))]
        mixture = tfp.distributions.Mixture(cat=cat, components=components)

        # Sample from the Gaussian mixture distribution
        samples = mixture.sample(num_samples)
        return samples


