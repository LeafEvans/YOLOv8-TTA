# Ultralytics YOLO 🚀, AGPL-3.0 license

from typing import Optional

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from .tta import TTAStrategy


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.tta_strategy: Optional[TTAStrategy] = None

    def setup_model(self, model, verbose=True):
        """Initialize model and set up TTA"""
        super().setup_model(model, verbose)

        if self.args.tta:
            if hasattr(model, "tta_strategy"):
                self.tta_strategy = model.tta_strategy  # type: TTAStrategy
                LOGGER.info("TTA strategy loaded from model")
            elif hasattr(model, "trainer") and hasattr(model.trainer, "tta_strategy"):
                self.tta_strategy = model.trainer.tta_strategy  # type: TTAStrategy
                LOGGER.info("TTA strategy loaded from model trainer")

    def preprocess(self, im):
        """Preprocess images and apply TTA"""
        im = super().preprocess(im)

        if self.args.tta and self.tta_strategy is not None:
            self._apply_tta(im)

        return im

    def _apply_tta(self, im):
        """Apply test-time adaptation strategy"""
        try:
            with torch.no_grad():
                # Extract features using TTAStrategy's extract_features method
                features = self.tta_strategy.extract_features(im)

                if features is None:
                    LOGGER.warning("No features extracted for TTA")
                    return

                # Calculate test distribution statistics
                test_mean, test_sigma = self.tta_strategy.compute_distribution(features)
                self.tta_strategy.update_ema(test_mean)

                # Calculate KL divergence between training and test distributions
                loss = self.tta_strategy.kl_divergence(
                    self.tta_strategy.train_stats["mean"], self.tta_strategy.train_stats["std"], test_mean, test_sigma
                )

                # Update model parameters if needed
                if self.tta_strategy.should_update(loss):
                    # Enable gradients for adaptor parameters
                    for name, param in self.model.named_parameters():
                        if "adaptor" in name:
                            param.requires_grad_(True)

                    # Backward propagation
                    loss.backward()

                    # Update parameters using gradient descent
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if "adaptor" in name and param.grad is not None:
                                param.data.sub_(param.grad * self.args.tta_lr)
                                param.grad = None

                    # Disable gradients after update
                    for param in self.model.parameters():
                        param.requires_grad_(False)

                    LOGGER.info(f"TTA update applied with loss: {loss.item():.4f}")

        except Exception as e:
            LOGGER.error(f"Error in TTA: {str(e)}")

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
