import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Import CONFIG from main.py for global access
from main import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class YOLACTHead(nn.Module):
    def __init__(self, in_ch, proto_ch, mask_size, num_cls):
        """
        Implements a simplified YOLACT-style mask head.
        Args:
            in_ch (int): Number of input channels from the detector's neck features.
            proto_ch (int): Number of channels for mask prototypes.
            mask_size (int): Target spatial resolution for generated masks (e.g., 28x28).
            num_cls (int): Number of object classes.
        """
        super().__init__()
        # Prototype network: Generates a set of 'mask prototypes'
        self.proto_net = nn.Sequential(
            nn.Conv2d(in_ch, proto_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(proto_ch, proto_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(proto_ch, mask_size, 1)  # Output `mask_size` channels (prototypes)
        )
        
        # Mask prediction network: Predicts 'mask coefficients' per class per anchor
        # This is a simplification. In actual YOLACT, this is usually per-anchor
        # and combined with detection head. Here, we simplify to per-class.
        self.mask_pred = nn.Conv2d(in_ch, num_cls, 1)  # Output `num_cls` channels (coefficients)
        
        self.mask_size = mask_size
        self.num_cls = num_cls
        logger.info(f"YOLACTHead initialized: in_ch={in_ch}, proto_ch={proto_ch}, mask_size={mask_size}, num_cls={num_cls}")

    def forward(self, feats):
        """
        Forward pass for the YOLACT-style mask head.
        Args:
            feats (torch.Tensor): Feature maps from the detector's neck (e.g., P3, P4, P5).
                                  Shape: (B, C_in, H_feat, W_feat).
        Returns:
            tuple: (proto_masks, mask_coefficients)
                proto_masks (torch.Tensor): Generated mask prototypes. Shape: (B, mask_size, H_feat, W_feat).
                mask_coefficients (torch.Tensor): Predicted mask coefficients. Shape: (B, num_cls, H_feat, W_feat).
        """
        # Generate mask prototypes
        proto = self.proto_net(feats)
        
        # Predict mask coefficients
        masks = self.mask_pred(feats)
        
        logger.debug(f"YOLACTHead forward pass: feats_shape={feats.shape}, proto_shape={proto.shape}, masks_shape={masks.shape}")
        return proto, masks

def attach_mask_head(model):
    """
    Attaches the YOLACT-style mask head to the YOLO model.
    This function modifies the model in-place.
    Args:
        model (YOLO): The YOLO model instance (e.g., ultralytics.YOLO).
    """
    # Ensure CONFIG is accessible. If running standalone, provide a dummy.
    current_config = CONFIG if 'CONFIG' in globals() else {
        'mask_head': {
            'enabled': True,
            'type': 'yolact',
            'num_classes': 1,
            'prototype_channels': 32,
            'mask_size': 28
        }
    }
    if not current_config['mask_head'].get('enabled', False):
        logger.info("Mask head attachment skipped as it's disabled in CONFIG.")
        return

    # Determine input channels for the mask head.
    # This is highly dependent on YOLOv8's internal architecture.
    # Typically, it would be the output channels of the last neck layer (e.g., P5 or P4).
    # For Ultralytics YOLOv8, `model.model[-1]` is often the detection head.
    # We need features *before* the final detection head's classification/regression layers.
    # This is a conceptual access. You might need to inspect `model.model` structure.
    
    feat_ch = None
    try:
        # A common approach is to get channels from a specific neck output, e.g., P3, P4, P5
        # For simplicity, let's assume a default input channel size or try to infer.
        # If model.model[-1] is the Detect head, its input channels might be complex.
        
        # A more robust way: explicitly define which feature map output (e.g., P3, P4, P5)
        # the mask head should connect to and get its channel count.
        
        # For a generic YOLOv8, a common feature map channel for the head is 256.
        # Let's use a default if we can't infer.
        if hasattr(model, 'model') and isinstance(model.model, nn.Module):
            # Attempt to find a layer that might output features before the final detection
            # This is highly speculative without knowing the exact model definition.
            # As a fallback, use a common feature channel size.
            feat_ch = 256 # Common feature channel size for YOLOv8 neck outputs
            
            # Example of trying to infer from YOLOv8's internal structure (conceptual)
            # for module in reversed(model.model):
            #     if hasattr(module, 'cv2') and hasattr(module.cv2, 'out_channels'):
            #         feat_ch = module.cv2.out_channels
            #         break
            #     elif isinstance(module, nn.Sequential):
            #         for sub_module in reversed(module):
            #             if hasattr(sub_module, 'cv2') and hasattr(sub_module.cv2, 'out_channels'):
            #                 feat_ch = sub_module.cv2.out_channels
            #                 break
            #         if feat_ch is not None: break
        
        if feat_ch is None:
            logger.warning("Could not infer feature channels for mask head. Using default 256. This might be incorrect.")
            feat_ch = 256  # Default fallback
            
    except Exception as e:
        logger.error(f"Error inferring feature channels for mask head: {e}. Using default 256.", exc_info=True)
        feat_ch = 256  # Fallback if inference fails

    cfg = current_config['mask_head']
    
    # Attach the mask head as a new module to the model
    # It's crucial that `model` is an `nn.Module` or has a way to add submodules.
    # Ultralytics YOLO models are `nn.Module`s.
    model.mask_head = YOLACTHead(in_ch=feat_ch,
                                 proto_ch=cfg['prototype_channels'],
                                 mask_size=cfg['mask_size'],
                                 num_cls=cfg['num_classes'])
    
    # IMPORTANT: You need to modify the YOLOv8's forward method to pass the
    # appropriate neck features to `model.mask_head` and collect its output.
    # This is a significant change to Ultralytics' internal `forward` logic.
    # This `attach_mask_head` only adds the module, not the forward pass integration.
    logger.warning("Mask head attached. You MUST modify the YOLOv8 model's forward method to use this mask head.")
    logger.warning("Example: In YOLOv8's forward, after neck features, add: `proto, masks = self.mask_head(neck_feat)`")


def mask_loss(proto, masks, gt_masks):
    """
    Calculates the mask loss (e.g., BCEWithLogitsLoss).
    Args:
        proto (torch.Tensor): Mask prototypes from YOLACTHead.
        masks (torch.Tensor): Mask coefficients from YOLACTHead.
        gt_masks (torch.Tensor): Ground truth masks.
                                  Shape: (B, Num_Instances, H_mask, W_mask) or (B, H_mask, W_mask).
    Returns:
        torch.Tensor: Scalar mask loss.
    """
    # Ensure CONFIG is accessible
    current_config = CONFIG if 'CONFIG' in globals() else {'mask_head': {'num_classes': 1}}

    if proto is None or masks is None or gt_masks is None or proto.numel() == 0 or masks.numel() == 0 or gt_masks.numel() == 0:
        logger.warning("Mask loss input is empty. Returning 0.0 loss.")
        # Ensure the returned tensor is on the same device as inputs if possible
        device = proto.device if proto is not None else (masks.device if masks is not None else 'cpu')
        return torch.tensor(0.0, device=device)

    # This is a simplified mask loss.
    # In a real YOLACT setup, the final instance masks are generated by
    # matrix multiplication of prototypes and coefficients, then cropped
    # and resized to match ground truth.
    # Then BCEWithLogitsLoss is applied to the final instance masks.
    
    # For this stub, we'll assume `masks` are directly comparable to `gt_masks`
    # after some internal processing or that `gt_masks` are already in the correct format.
    
    # If `masks` are coefficients and `proto` are prototypes, the actual mask
    # generation for an instance 'i' would be:
    # `instance_mask_i = torch.sigmoid(torch.sum(proto * coefficients_for_instance_i, dim=1))`
    # This requires careful matching of predictions to ground truth instances.
    
    # For a simple placeholder, let's assume `masks` represent the final
    # predicted masks (e.g., after some internal processing in the model's forward).
    
    # Ensure `masks` and `gt_masks` have compatible shapes and types.
    # Resize `masks` to `gt_masks` size if necessary (e.g., for different resolutions)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Ensure gt_masks is float and has the same number of dimensions as masks
    if gt_masks.ndim == 3:  # (B, H, W)
        gt_masks_processed = gt_masks.unsqueeze(1).float()  # -> (B, 1, H, W)
    else:  # (B, N_instances, H, W)
        # Take the first instance for simplicity of the stub
        gt_masks_processed = gt_masks[:, 0:1, :, :].float()  # -> (B, 1, H, W)
    
    # Resize masks to match ground truth mask dimensions if needed
    if masks.shape[-2:] != gt_masks_processed.shape[-2:]:
        logger.warning(f"Masks shape {masks.shape} does not match gt_masks shape {gt_masks_processed.shape}. Resizing masks for loss calculation.")
        masks_for_loss = F.interpolate(masks, size=gt_masks_processed.shape[-2:], mode='bilinear', align_corners=False)
    else:
        masks_for_loss = masks

    # Ensure masks_for_loss also has a single channel for this simplified BCE if it's multi-channel
    if masks_for_loss.shape[1] > 1:
        masks_for_loss = masks_for_loss[:, 0:1, :, :] # Take first channel or average if multiple classes
    
    # Final check for shape compatibility before loss calculation
    if masks_for_loss.shape != gt_masks_processed.shape:
        logger.error(f"Shape mismatch after processing for BCE loss: Predicted {masks_for_loss.shape} vs Ground Truth {gt_masks_processed.shape}")
        # As a last resort, resize again if they still don't match, though this indicates a deeper issue
        masks_for_loss = F.interpolate(masks_for_loss, size=gt_masks_processed.shape[-2:], mode='bilinear', align_corners=False)


    loss = loss_fn(masks_for_loss, gt_masks_processed)
    logger.debug(f"Mask loss (stub) calculated: {loss.item():.4f}")
    return loss

if __name__ == '__main__':
    # Example usage for testing the segmentation module
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Dummy CONFIG for testing
    dummy_config_for_test = {
        'mask_head': {
            'enabled': True,
            'type': 'yolact',
            'num_classes': 1,  # Single class for simplicity
            'prototype_channels': 32,
            'mask_size': 28
        }
    }
    # Temporarily set global CONFIG for testing if it's not already set
    if 'CONFIG' not in globals():
        global CONFIG
        CONFIG = dummy_config_for_test
    else: # If CONFIG exists, update it for the test
        CONFIG.update(dummy_config_for_test)

    # Dummy model to attach mask head to
    class MockYOLO(nn.Module):
        def __init__(self):
            super().__init__()
            # Simulate a neck output feature map layer
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 256, 3, padding=1) # Simulate neck output channels
            )
            # This is where the mask head will be attached
            self.mask_head = None 
        
        # A simplified forward pass that would be modified to include mask head
        def forward(self, x):
            feats = self.model(x)
            # In a real scenario, you'd pass feats to self.mask_head
            # and return its outputs along with detection outputs.
            return {'neck_feat': feats} # Return a dict to match yolo_fine_tuning stub

    mock_yolo_model = MockYOLO()
    
    print("\n--- Testing attach_mask_head ---")
    attach_mask_head(mock_yolo_model)
    print(f"Mask head attached: {mock_yolo_model.mask_head is not None}")
    if mock_yolo_model.mask_head:
        print(f"Mask head module: {mock_yolo_model.mask_head}")

    print("\n--- Testing mask_loss ---")
    # Simulate neck features
    dummy_feats = torch.randn(2, 256, 20, 20) # Batch, Channels, H_feat, W_feat
    
    # Get outputs from the attached mask head
    if mock_yolo_model.mask_head:
        dummy_proto, dummy_masks_coeffs = mock_yolo_model.mask_head(dummy_feats)
    else:
        dummy_proto = torch.randn(2, 28, 20, 20) # Dummy prototypes
        dummy_masks_coeffs = torch.randn(2, 1, 20, 20) # Dummy coefficients (1 class)

    # Simulate ground truth masks (e.g., 2 instances, 28x28 resolution)
    # For simplicity, let's assume a single ground truth mask per image (B, 1, H, W)
    dummy_gt_masks = (torch.rand(2, 1, 28, 28) > 0.5).float() # Binary masks

    loss = mask_loss(dummy_proto, dummy_masks_coeffs, dummy_gt_masks)
    print(f"Calculated mask_loss: {loss.item():.4f}")

    # Test with mask head disabled
    print("\n--- Testing attach_mask_head (Disabled in CONFIG) ---")
    CONFIG['mask_head']['enabled'] = False
    mock_yolo_model_disabled = MockYOLO() # New model instance
    attach_mask_head(mock_yolo_model_disabled)
    print(f"Mask head attached (should be False): {mock_yolo_model_disabled.mask_head is not None}")
