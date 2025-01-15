'''
This file will contain all of the libraries and functions that we write throughout our notebooks so that they can be accessed in the following notebooks
'''
import equinox as eqx
import jax
import jax.numpy as np
from jax.scipy.stats import norm
from jax import grad, vmap, jacfwd, random
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import field
from typing import Callable, List, Tuple
import optax
from tqdm.auto import tqdm
from jax.tree_util import tree_flatten, tree_map

'''
Models:
'''

class GaussianScoreModel(eqx.Module):
  mu: np.array = field(default_factory=lambda: np.array(0.0))
  log_sigma: np.array = field(default_factory=lambda: np.array(0.0))

  @eqx.filter_jit
  def __call__(self, x):
    score = jacfwd(norm.logpdf)
    return score(x, loc=self.mu, scale=np.exp(self.log_sigma))


class MixtureGaussian(eqx.Module):
  pi: np.array
  mus: np.array
  logsigmas: np.array

  def __init__(self, mus, logsigmas, pi):
    self.mus = mus
    self.logsigmas = logsigmas
    self.pi = pi

    # Check that mus, log_sigmas, and ws are of the same length.
    lengths = set(map(len, [mus, logsigmas, pi]))
    if len(lengths) != 1:
        raise ValueError(
            "`mus`, `log_sigmas` and `ws` must all be of the same length!"
        )

  def pdf(self, x):
    return np.sum(self.pi * norm.pdf(x, loc=self.mus, scale=np.exp(self.logsigmas)))

  def lpdf(self, x):
    return np.log(self.pdf(x))

  def sample(self, n_samples):
    # For each sample decide which gaussian to sample from based on weights
    props = np.round(n_samples * self.pi).astype(int)

    samples = [random.normal(key=random.key(45), shape=(prop, ), dtype=np.float32)*sigma + mu for prop,sigma,mu in  zip(props,np.exp(self.logsigmas), self.mus)]
    samples = np.concatenate(samples)
    samples = random.permutation(key=random.key(45), x=samples)
    return samples

  @eqx.filter_jit
  def __call__(self, x):
    score = jacfwd(self.lpdf)
    return score(x)


'''
Loss Functions and Training:
'''

def score_matching_loss(model, data):
  '''
  data should have shape (batch, :), at least a 2D tensor
  '''
  # Compute the hessians of the logpdf (or jacobian of score fn)
  hess = vmap(jacfwd(model))(data)

  # Get the diagonals of the Hessians, correspond to derivative of score fn w.r.t. i-th data dimension
  # If it's a 2D tensor, add another dimension so that when we call np.diagonal it reduces dim instead of increasing it
  if hess.shape[-1] == 1 and len(hess.shape)==2:
    hess = np.expand_dims(hess, -1)
  term1 = vmap(np.diagonal)(hess)

  term2 = 0.5 * vmap(model)(data)**2
  term2 = np.reshape(term2, term1.shape) # reshape to be like term 1

  inner = term1 + term2
  # sum along data dimensions (mapping np.sum to each individual data tensor ensures final shape is (batch, ))
  inner = vmap(np.sum)(inner)

  # Now average over the entire batch of data
  loss = np.mean(inner)
  return loss


def regularized_loss(model, data):
  sm_loss = score_matching_loss(model, data)

  # compute l2 norm
  params = eqx.filter(model, eqx.is_array_like)
  squared = tree_map(lambda x: np.power(x, 2), params)
  summed = tree_map(np.sum, squared)
  flattened, _ = tree_flatten(summed)

  l2_norm = np.mean(np.array(flattened)) * 0.05 # scale it down too

  return sm_loss + l2_norm


def fit(
    model: eqx.Module,
    data: np.array,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    steps: int,
    progress_bar: bool = True,
  ) -> Tuple[List[eqx.Module], List]:
  """
  Fit model to data

  data should have shape (batch, :), at least a 2D tensor

  returns updated model + loss training history
  """
  # Set up the state of the optimizer, filtering for weights which are inexact (which is apparently anything that is floating point)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

  # Here we call filter_value_and_grad s.t. it returns pair (value, grad). We jit compile it too to be faster
  dloss = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))

  @eqx.filter_jit
  def step(model, opt_state, data):
    """
    One step of training loop, jitted to run faster
    """

    loss, grads = dloss(model, data)

    # Next we compute the updates and obtain optimizer state
    updates, opt_state = optimizer.update(grads, opt_state)

    # Apply updates to the model
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

  loss_hist = []
  iterator = range(steps)
  if progress_bar: iterator = tqdm(iterator)

  for _ in iterator:
    model, opt_state, loss = step(model, opt_state, data)
    loss_hist.append(loss)
  return model, loss_hist

