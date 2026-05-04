import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    Grad-CAM for CNNs.
    Works well with torchvision ResNet (target layer usually model.layer4[-1]).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out  # shape: [B, C, H, W]

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] is gradient wrt layer output
            self.gradients = grad_out[0]  # shape: [B, C, H, W]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        # register_full_backward_hook is preferred in new PyTorch
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        # cam shape [B, 1, H, W] or [B, H, W]
        if cam.dim() == 3:
            cam = cam.unsqueeze(1)
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return cam

    def __call__(self, x: torch.Tensor, class_idx=None):
        """
        x: input tensor [1,3,H,W]
        class_idx: for binary output (1 logit), keep None.
        returns: cam numpy [H,W] in [0,1]
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)  # [B, 1] for your model
        if logits.dim() == 2 and logits.size(1) == 1:
            score = logits[:, 0]  # binary logit
        else:
            # multi-class fallback
            if class_idx is None:
                class_idx = logits.argmax(dim=1)
            score = logits.gather(1, class_idx.view(-1, 1)).squeeze(1)

        # Backprop
        score.sum().backward(retain_graph=False)

        # Grad-CAM weights: global average pool gradients
        grads = self.gradients          # [B,C,H,W]
        acts = self.activations         # [B,C,H,W]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)

        cam = self._normalize_cam(cam)  # [B,1,H,W] in [0,1]
        cam = cam[0, 0].detach().cpu().numpy()
        return cam