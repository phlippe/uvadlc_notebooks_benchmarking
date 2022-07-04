import os
import time
import numpy as np
from typing import Sequence

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from flax.training import train_state, checkpoints

import optax

import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST

DATASET_PATH = "../data"
CHECKPOINT_PATH = "../saved_models/tutorial11_jax"
timestr = time.strftime("%Y_%m_%d__%H_%M_%S")
LOG_FILE = open(f'../logs/tutorial11_jax_{timestr}.txt', 'w')

main_rng = random.PRNGKey(42)

print("Device:", jax.devices()[0])


def image_to_numpy(img):
    img = np.array(img, dtype=np.int32)
    img = img[..., None]  # Make image [28, 28, 1]
    return img


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


train_dataset = MNIST(root=DATASET_PATH, train=True,
                      transform=image_to_numpy, download=True)
train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000],
                                                   generator=torch.Generator().manual_seed(42))

test_set = MNIST(root=DATASET_PATH, train=False,
                 transform=image_to_numpy, download=True)

train_exmp_loader = data.DataLoader(
    train_set, batch_size=256, shuffle=False, drop_last=False, collate_fn=numpy_collate)
train_data_loader = data.DataLoader(train_set,
                                    batch_size=128,
                                    shuffle=True,
                                    drop_last=True,
                                    collate_fn=numpy_collate,
                                    num_workers=8,
                                    persistent_workers=True)
val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False,
                             drop_last=False, num_workers=4, collate_fn=numpy_collate)
test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False,
                              drop_last=False, num_workers=4, collate_fn=numpy_collate)


class ImageFlow(nn.Module):
    # A list of flows (each a nn.Module) that should be applied on the images.
    flows: Sequence[nn.Module]
    # Number of importance samples to use during testing (see explanation below).
    import_samples: int = 8

    def __call__(self, x, rng, testing=False):
        if not testing:
            bpd, rng = self._get_likelihood(x, rng)
        else:
            # Perform importance sampling during testing => estimate likelihood M times for each image
            img_ll, rng = self._get_likelihood(x.repeat(self.import_samples, 0),
                                               rng,
                                               return_ll=True)
            img_ll = img_ll.reshape(-1, self.import_samples)

            # To average the probabilities, we need to go from log-space to exp, and back to log.
            # Logsumexp provides us a stable implementation for this
            img_ll = jax.nn.logsumexp(
                img_ll, axis=-1) - np.log(self.import_samples)

            # Calculate final bpd
            bpd = -img_ll * np.log2(np.exp(1)) / np.prod(x.shape[1:])
            bpd = bpd.mean()
        return bpd, rng

    def encode(self, imgs, rng):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, ldj = imgs, jnp.zeros(imgs.shape[0])
        for flow in self.flows:
            z, ldj, rng = flow(z, ldj, rng, reverse=False)
        return z, ldj, rng

    def _get_likelihood(self, imgs, rng, return_ll=False):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        z, ldj, rng = self.encode(imgs, rng)
        log_pz = jax.scipy.stats.norm.logpdf(z).sum(axis=(1, 2, 3))
        log_px = ldj + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        return (bpd.mean() if not return_ll else log_px), rng

    def sample(self, img_shape, rng, z_init=None):
        """
        Sample a batch of images from the flow.
        """
        # Sample latent representation from prior
        if z_init is None:
            rng, normal_rng = random.split(rng)
            z = random.normal(normal_rng, shape=img_shape)
        else:
            z = z_init

        # Transform z to x by inverting the flows
        ldj = jnp.zeros(img_shape[0])
        for flow in reversed(self.flows):
            z, ldj, rng = flow(z, ldj, rng, reverse=True)
        return z, rng


class Dequantization(nn.Module):
    # Small constant that is used to scale the original input for numerical stability.
    alpha: float = 1e-5
    # Number of possible discrete values (usually 256 for 8-bit image)
    quants: int = 256

    def __call__(self, z, ldj, rng, reverse=False):
        if not reverse:
            z, ldj, rng = self.dequant(z, ldj, rng)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            ldj += np.log(self.quants) * np.prod(z.shape[1:])
            z = jnp.floor(z)
            z = jax.lax.clamp(min=0., x=z, max=self.quants -
                              1.).astype(jnp.int32)
        return z, ldj, rng

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z-2*jax.nn.softplus(-z)).sum(axis=[1, 2, 3])
            z = nn.sigmoid(z)
        else:
            # Scale to prevent boundaries 0 and 1
            z = z * (1 - self.alpha) + 0.5 * self.alpha
            ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            ldj += (-jnp.log(z) - jnp.log(1-z)).sum(axis=[1, 2, 3])
            z = jnp.log(z) - jnp.log(1-z)
        return z, ldj

    def dequant(self, z, ldj, rng):
        # Transform discrete values to continuous volumes
        z = z.astype(jnp.float32)
        rng, uniform_rng = random.split(rng)
        z = z + random.uniform(uniform_rng, z.shape)
        z = z / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj, rng


class VariationalDequantization(Dequantization):
    # A list of flow transformations to use for modeling q(u|x)
    var_flows: Sequence[nn.Module] = None

    def dequant(self, z, ldj, rng):
        z = z.astype(jnp.float32)
        # We condition the flows on x, i.e. the original image
        img = (z / 255.0) * 2 - 1

        # Prior of u is a uniform distribution as before
        # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
        rng, uniform_rng = random.split(rng)
        deq_noise = random.uniform(uniform_rng, z.shape)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)
        if self.var_flows is not None:
            for flow in self.var_flows:
                deq_noise, ldj, rng = flow(
                    deq_noise, ldj, rng, reverse=False, orig_img=img)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        # After the flows, apply u as in standard dequantization
        z = (z + deq_noise) / 256.0
        ldj -= np.log(256.0) * np.prod(z.shape[1:])
        return z, ldj, rng


class CouplingLayer(nn.Module):
    network: nn.Module  # NN to use in the flow for predicting mu and sigma
    # Binary mask where 0 denotes that the element should be transformed, and 1 not.
    mask: np.ndarray
    c_in: int  # Number of input channels

    def setup(self):
        self.scaling_factor = self.param('scaling_factor',
                                         nn.initializers.zeros,
                                         (self.c_in,))

    def __call__(self, z, ldj, rng, reverse=False, orig_img=None):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            rng - PRNG state
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        z_in = z * self.mask
        if orig_img is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(jnp.concatenate([z_in, orig_img], axis=-1))
        s, t = nn_out.split(2, axis=-1)

        # Stabilize scaling output
        s_fac = jnp.exp(self.scaling_factor).reshape(1, 1, 1, -1)
        s = nn.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * jnp.exp(s)
            ldj += s.sum(axis=[1, 2, 3])
        else:
            z = (z * jnp.exp(-s)) - t
            ldj -= s.sum(axis=[1, 2, 3])

        return z, ldj, rng


def create_checkerboard_mask(h, w, invert=False):
    x, y = jnp.arange(h, dtype=jnp.int32), jnp.arange(w, dtype=jnp.int32)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    mask = jnp.fmod(xx + yy, 2)
    mask = mask.astype(jnp.float32).reshape(1, h, w, 1)
    if invert:
        mask = 1 - mask
    return mask


def create_channel_mask(c_in, invert=False):
    mask = jnp.concatenate([
        jnp.ones((c_in//2,), dtype=jnp.float32),
        jnp.zeros((c_in-c_in//2,), dtype=jnp.float32)
    ])
    mask = mask.reshape(1, 1, 1, c_in)
    if invert:
        mask = 1 - mask
    return mask


class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def __call__(self, x):
        return jnp.concatenate([nn.elu(x), nn.elu(-x)], axis=-1)


class GatedConv(nn.Module):
    """ This module applies a two-layer convolutional ResNet block with input gate """
    c_in: int  # Number of input channels
    c_hidden: int  # Number of hidden dimensions

    @nn.compact
    def __call__(self, x):
        out = nn.Sequential([
            ConcatELU(),
            nn.Conv(self.c_hidden, kernel_size=(3, 3)),
            ConcatELU(),
            nn.Conv(2*self.c_in, kernel_size=(1, 1))
        ])(x)
        val, gate = out.split(2, axis=-1)
        return x + val * nn.sigmoid(gate)


class GatedConvNet(nn.Module):
    c_hidden: int  # Number of hidden dimensions to use within the network
    c_out: int  # Number of output channels
    num_layers: int = 3  # Number of gated ResNet blocks to apply

    def setup(self):
        layers = []
        layers += [nn.Conv(self.c_hidden, kernel_size=(3, 3))]
        for layer_index in range(self.num_layers):
            layers += [GatedConv(self.c_hidden, self.c_hidden),
                       nn.LayerNorm()]
        layers += [ConcatELU(),
                   nn.Conv(self.c_out, kernel_size=(3, 3),
                           kernel_init=nn.initializers.zeros)]
        self.nn = nn.Sequential(layers)

    def __call__(self, x):
        return self.nn(x)


def create_simple_flow(use_vardeq=True):
    flow_layers = []
    if use_vardeq:
        vardeq_layers = [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=16),
                                       mask=create_checkerboard_mask(
                                           h=28, w=28, invert=(i % 2 == 1)),
                                       c_in=1) for i in range(4)]
        flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]
    else:
        flow_layers += [Dequantization()]

    for i in range(8):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=32),
                                      mask=create_checkerboard_mask(
                                          h=28, w=28, invert=(i % 2 == 1)),
                                      c_in=1)]

    flow_model = ImageFlow(flow_layers)
    return flow_model


class TrainerModule:

    def __init__(self, model_name, flow, lr=1e-3, seed=42):
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = flow
        # Prepare logging
        self.exmp_imgs = next(iter(train_exmp_loader))[0]
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.model_name)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model()

    def create_functions(self):
        # Training function
        def train_step(state, rng, batch):
            imgs, _ = batch
            def loss_fn(params): return self.model.apply(
                {'params': params}, imgs, rng, testing=False)
            (loss, rng), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params)  # Get loss and gradients for loss
            state = state.apply_gradients(grads=grads)  # Optimizer update step
            return state, rng, loss
        self.train_step = jax.jit(train_step)
        # Eval function, which is separately jitted for validation and testing

        def eval_step(state, rng, batch, testing):
            return self.model.apply({'params': state.params}, batch[0], rng, testing=testing)
        self.eval_step = jax.jit(eval_step, static_argnums=(3,))

    def init_model(self):
        # Initialize model
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng, init_rng, flow_rng = jax.random.split(self.rng, 3)
        params = self.model.init(init_rng, self.exmp_imgs, flow_rng)['params']
        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.exponential_decay(
            init_value=self.lr,
            transition_steps=len(train_data_loader),
            decay_rate=0.99,
            end_value=0.01*self.lr
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at 1
            optax.adam(lr_schedule)
        )
        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer)

    def train_model(self, train_loader, val_loader, num_epochs=500):
        # Train model for defined number of epochs
        best_eval = 1e6
        for epoch_idx in tqdm(range(1, num_epochs+1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 5 == 0:
                eval_bpd = self.eval_model(val_loader, testing=False)
                self.logger.add_scalar(
                    'val/bpd', eval_bpd, global_step=epoch_idx)
                if eval_bpd < best_eval:
                    best_eval = eval_bpd
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def train_epoch(self, data_loader, epoch):
        # Train model for one epoch, and log avg loss
        avg_loss = 0.
        for batch in tqdm(data_loader, leave=False):
            self.state, self.rng, loss = self.train_step(
                self.state, self.rng, batch)
            avg_loss += loss
        avg_loss /= len(data_loader)
        self.logger.add_scalar('train/bpd', avg_loss.item(), global_step=epoch)

    def eval_model(self, data_loader, testing=False):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            loss, self.rng = self.eval_step(
                self.state, self.rng, batch, testing=testing)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir, target=self.state.params, step=step)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            params = checkpoints.restore_checkpoint(
                ckpt_dir=self.log_dir, target=self.state.params)
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'),
                                                    target=self.state.params)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=self.state.tx)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))


def train_flow(flow, model_name="MNISTFlow"):
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(model_name, flow)
    start_time = time.time()
    trainer.train_model(train_data_loader,
                        val_loader,
                        num_epochs=200)
    train_time = time.time()
    print(f'{model_name} - Full training time:',
          time.strftime('%H:%M:%S', time.gmtime(train_time - start_time)),
          file=LOG_FILE, flush=True)


class SqueezeFlow(nn.Module):

    def __call__(self, z, ldj, rng, reverse=False):
        B, H, W, C = z.shape
        if not reverse:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, H//2, 2, W//2, 2, C)
            z = z.transpose((0, 1, 3, 2, 4, 5))
            z = z.reshape(B, H//2, W//2, 4*C)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, H, W, 2, 2, C//4)
            z = z.transpose((0, 1, 3, 2, 4, 5))
            z = z.reshape(B, H*2, W*2, C//4)
        return z, ldj, rng


class SplitFlow(nn.Module):

    def __call__(self, z, ldj, rng, reverse=False):
        if not reverse:
            z, z_split = z.split(2, axis=-1)
            ldj += jax.scipy.stats.norm.logpdf(z_split).sum(axis=[1, 2, 3])
        else:
            z_split = random.normal(rng, z.shape)
            z = jnp.concatenate([z, z_split], axis=-1)
            ldj -= jax.scipy.stats.norm.logpdf(z_split).sum(axis=[1, 2, 3])
        return z, ldj, rng


def create_multiscale_flow():
    flow_layers = []

    vardeq_layers = [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=16),
                                   mask=create_checkerboard_mask(
                                       h=28, w=28, invert=(i % 2 == 1)),
                                   c_in=1) for i in range(4)]
    flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]

    flow_layers += [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=32),
                                  mask=create_checkerboard_mask(
                                      h=28, w=28, invert=(i % 2 == 1)),
                                  c_in=1) for i in range(2)]
    flow_layers += [SqueezeFlow()]
    for i in range(2):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=8, c_hidden=48),
                                      mask=create_channel_mask(
                                          c_in=4, invert=(i % 2 == 1)),
                                      c_in=4)]
    flow_layers += [SplitFlow(),
                    SqueezeFlow()]
    for i in range(4):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=16, c_hidden=64),
                                      mask=create_channel_mask(
                                          c_in=8, invert=(i % 2 == 1)),
                                      c_in=8)]

    flow_model = ImageFlow(flow_layers)
    return flow_model


train_flow(create_simple_flow(use_vardeq=False), model_name="MNISTFlow_simple")
train_flow(create_simple_flow(use_vardeq=True), model_name="MNISTFlow_vardeq")
train_flow(create_multiscale_flow(), model_name="MNISTFlow_multiscale")
