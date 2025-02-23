# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
from torch.distributions import Normal
from ultralytics.nn.modules import Bottleneck
from ultralytics.utils import LOGGER
from ultralytics.nn.modules.adaptor import LightweightAdaptor


class TTAStrategy:
    def __init__(
        self,
        model,
        alpha=0.01,
        tau1=1.1,
        tau2=1.05,
        momentum=0.99,
        reduction_ratio=32,
        tta_lr=0.001,
        feature_layer=-3,
    ):
        """Initialize TTA strategy
        Args:
            model: YOLO model
            alpha: EMA update rate
            tau1: Primary threshold
            tau2: Secondary threshold
            momentum: EMA momentum
            reduction_ratio: Reduction ratio for adaptors
            tta_lr: Learning rate for TTA updates
            feature_layer: Index of the layer to extract features from
        """
        self.model = model
        self.alpha = alpha
        self.tau1 = tau1
        self.tau2 = tau2
        self.momentum = momentum
        self.reduction_ratio = reduction_ratio
        self.tta_lr = tta_lr
        self.feature_layer = feature_layer
        self.ema_mean = None
        self.ema_loss = None
        self.train_stats = {}
        self.current_stats = {}

    def extract_features(self, x):
        """Extract features from specified layer with detailed debug output"""
        features = None
        target_idx = len(self.model.model) + self.feature_layer

        LOGGER.info(f"Attempting to extract features from layer {target_idx}")

        for i, m in enumerate(self.model.model):
            x = m(x)
            LOGGER.info(f"Layer {i}: {m.__class__.__name__}, output shape {x.shape}")

            if i == target_idx:
                features = x
                LOGGER.info(f"Selected features from layer {i} with shape {features.shape}")
                break

        if features is None:
            LOGGER.warning(f"Target layer {target_idx} not found, using last layer output")
            features = x

        return features

    def collect_train_statistics(self, x):
        """Collect feature statistics from training set

        Args:
            x: Input tensor or features
        """
        try:
            if x.dim() == 4 and x.size(1) == 3:
                features = self.extract_features(x)
            else:
                features = x

            if features is None:
                raise ValueError("Failed to extract features")

            LOGGER.info(f"Processing features with shape: {features.shape}")

            mu = torch.mean(features, dim=[0, 2, 3])
            sigma = torch.std(features, dim=[0, 2, 3])

            self.train_stats["mean"] = mu
            self.train_stats["std"] = sigma
            LOGGER.info(f"Train statistics collected: mean={mu.mean().item():.3f}, std={sigma.mean().item():.3f}")

        except Exception as e:
            LOGGER.error(f"Error collecting statistics: {str(e)}")
            raise

    def compute_distribution(self, x):
        """Compute feature distribution
        Args:
            x: Input tensor or features
        Returns:
            Tuple of mean and standard deviation
        """
        if x.dim() == 4 and x.size(1) == 3:
            features = self.extract_features(x)
        else:
            features = x

        mu = torch.mean(features, dim=[0, 2, 3])
        sigma = torch.std(features, dim=[0, 2, 3])

        self.current_stats["mean"] = mu
        self.current_stats["std"] = sigma

        return mu, sigma

    def update_ema(self, current_mean):
        """Update exponential moving average"""
        if self.ema_mean is None:
            self.ema_mean = current_mean
        else:
            self.ema_mean = self.momentum * self.ema_mean + (1 - self.momentum) * current_mean

    def compute_domain_gap(self):
        """Compute domain gap"""
        if not self.train_stats or not self.current_stats:
            return float("inf")

        kl_div = self.kl_divergence(
            self.train_stats["mean"], self.train_stats["std"], self.current_stats["mean"], self.current_stats["std"]
        )
        return kl_div

    def kl_divergence(self, mu1, sigma1, mu2, sigma2):
        """Calculate KL divergence"""
        dist1 = Normal(mu1, sigma1)
        dist2 = Normal(mu2, sigma2)
        return torch.distributions.kl.kl_divergence(dist1, dist2).mean()

    def should_update(self, current_loss):
        """Determine if update is needed"""
        if self.ema_loss is None:
            self.ema_loss = current_loss
            return True

        old_ema = self.ema_loss
        self.ema_loss = self.momentum * self.ema_loss + (1 - self.momentum) * current_loss

        if current_loss / old_ema > self.tau1:
            LOGGER.info(f"Major distribution shift detected: {current_loss / old_ema:.3f} > {self.tau1}")
            return True

        if current_loss / old_ema > self.tau2:
            LOGGER.info(f"Minor distribution shift detected: {current_loss / old_ema:.3f} > {self.tau2}")
            return True

        return False

    def init_adaptors(self, model=None):
        """Initialize adaptors in TTAStrategy"""
        target_model = model if model is not None else self.model
        adaptor_count = 0

        in_channels = 256

        for module in target_model.modules():
            if isinstance(module, Bottleneck):
                adaptor = LightweightAdaptor(
                    in_channels=in_channels, reduction_ratio=self.reduction_ratio, out_channels=in_channels
                )
                module.adaptor = adaptor
                adaptor_count += 1

        LOGGER.info(f"Initialized {adaptor_count} TTA adaptors with {in_channels} channels")
        return adaptor_count
