"""
Segmentation model for stroke detection
"""

import os
import numpy as np
import torch
from monai.transforms import Compose, Resize, ScaleIntensity, ToTensor, Activations, AsDiscrete, FillHoles, RemoveSmallObjects
from monai.networks.nets import AttentionUnet
import cv2

# Import brain segmentation function from classification module
from StrokeModels.classification import segment_brain_inside_skull


class StrokeSegmentationModel:
    """
    Class for handling stroke segmentation using UNet
    """
    def __init__(self, model_path):
        """
        Initialize the stroke segmentation model
        
        Parameters:
        -----------
        model_path : str
            Path to the model weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
        ).to(self.device)

        
        # Load model weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Transform for processing images
        self.transform = Compose([
            Resize(spatial_size=(512, 512)),
            ScaleIntensity(),
            ToTensor(),
        ])
        
        # Post-processing transforms
        self.post_transform = Compose([Activations(sigmoid=True),
                      AsDiscrete(threshold=0.5),
                      FillHoles(),
                      RemoveSmallObjects(min_size=10),
                      ])
    
    def preprocess_image(self, image_data):
        """
        Preprocess the input image data with brain extraction
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            Raw image data
            
        Returns:
        --------
        torch.Tensor
            Processed image tensor
        """
        # Extract brain from the image
        _, _, _, brain_only, _, _ = segment_brain_inside_skull(image_data)
        
        # Add channel dimension if needed
        if len(brain_only.shape) == 2:
            brain_only = np.expand_dims(brain_only, axis=0)
        else:
            # If it has multiple channels, convert to single channel
            brain_only = np.expand_dims(np.mean(brain_only, axis=2) if brain_only.shape[2] > 1 else brain_only[:, :, 0], axis=0)
        
        # Apply transforms (resize and scale intensity)
        image_tensor = self.transform(brain_only.astype(np.float32))
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict_mask(self, image_data, threshold=0.5):
        """
        Generate segmentation mask for the input image
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            Input image data
        threshold : float, optional
            Threshold for binary segmentation, defaults to 0.5
            
        Returns:
        --------
        numpy.ndarray
            Binary segmentation mask
        """
        image_tensor = self.preprocess_image(image_data)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            
            # Apply post-processing
            predicted_mask = self.post_transform(output[0])
            
        # Remove batch and channel dimensions to return a 2D mask
        return np.squeeze(predicted_mask.cpu().numpy())
    
    def get_overlay(self, image_data, predicted_mask=None):
        """
        Create an overlay image showing segmentation results on the original image
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            Original image data
        predicted_mask : numpy.ndarray, optional
            Predicted segmentation mask. If None, prediction will be performed
            
        Returns:
        --------
        numpy.ndarray
            RGB overlay image
        """
        # Ensure we have a mask
        if predicted_mask is None:
            predicted_mask = self.predict_mask(image_data)
        
        # Ensure image is grayscale
        if len(image_data.shape) > 2:
            original_image = np.mean(image_data, axis=2).astype(np.uint8) if image_data.shape[2] > 1 else image_data[:, :, 0].astype(np.uint8)
        else:
            original_image = image_data.astype(np.uint8)
        
        # Resize original image to match mask dimensions if needed
        if original_image.shape != predicted_mask.shape:
            resized_original = cv2.resize(original_image, (predicted_mask.shape[1], predicted_mask.shape[0]))
        else:
            resized_original = original_image
        
        # Convert to RGB for overlay
        overlay_img = cv2.cvtColor(resized_original, cv2.COLOR_GRAY2BGR)
        
        # Create red mask for segmentation overlay
        red_mask = np.zeros_like(overlay_img)
        red_mask[predicted_mask > 0.5] = [0, 0, 255]  # BGR format (red)
        
        # Create overlay (70% original, 30% mask)
        overlay = cv2.addWeighted(overlay_img, 0.7, red_mask, 0.3, 0)
        
        return overlay