#!/usr/bin/env python

"""
Feature filtering utilities for selective training on LeRobot datasets.

This module provides functionality to selectively use subsets of dataset features
during training without recreating the entire dataset.
"""

import logging
from typing import Dict, List, Optional, Any
import torch
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.train import FeatureSelectionConfig


logger = logging.getLogger(__name__)


class FeatureFilter:
    """Filters dataset features based on configuration."""
    
    def __init__(self, dataset: LeRobotDataset, config: FeatureSelectionConfig):
        """
        Initialize feature filter.
        
        Args:
            dataset: The LeRobot dataset to filter
            config: Feature selection configuration
        """
        self.dataset = dataset
        self.config = config
        self.meta = dataset.meta
        
        # Cache filtered feature information
        self._filtered_cameras = None
        self._state_mask = None
        self._state_feature_names = None
        self._filtered_meta = None
        
        self._compute_filters()
        
    def _compute_filters(self):
        """Compute filtering masks and metadata."""
        self._compute_camera_filter()
        self._compute_state_filter()
        self._create_filtered_meta()
        
    def _compute_camera_filter(self):
        """Determine which cameras to use."""
        available_cameras = list(self.meta.camera_keys)
        
        if self.config.cameras is not None:
            # Use only specified cameras
            self._filtered_cameras = [cam for cam in self.config.cameras if cam in available_cameras]
            if len(self._filtered_cameras) != len(self.config.cameras):
                missing = set(self.config.cameras) - set(available_cameras)
                logger.warning(f"Cameras not found in dataset: {missing}")
                
        elif self.config.exclude_cameras is not None:
            # Use all cameras except excluded ones
            self._filtered_cameras = [cam for cam in available_cameras if cam not in self.config.exclude_cameras]
            
        else:
            # Use all cameras
            self._filtered_cameras = available_cameras
            
        logger.info(f"Using cameras: {self._filtered_cameras}")
        
    def _compute_state_filter(self):
        """Compute state dimension filtering mask."""
        # This is a simplified version - you might need to adapt based on your exact state structure
        state_names = self.meta.features["observation.state"]["names"]
        total_dims = len(state_names)
        
        if self.config.custom_state_indices is not None:
            # Use custom indices
            self._state_mask = np.zeros(total_dims, dtype=bool)
            self._state_mask[self.config.custom_state_indices] = True
            self._state_feature_names = [state_names[i] for i in self.config.custom_state_indices]
            
        else:
            # Build mask based on feature types
            mask = np.ones(total_dims, dtype=bool)
            filtered_names = []
            
            for i, name in enumerate(state_names):
                include = True
                
                # Check joint group filtering
                if self.config.joint_groups is not None:
                    include = any(group in name for group in self.config.joint_groups)
                elif self.config.exclude_joint_groups is not None:
                    include = not any(group in name for group in self.config.exclude_joint_groups)
                
                # Check feature type filtering
                if include:
                    if "_pos" in name and not self.config.use_joint_positions:
                        include = False
                    elif "_vel" in name and not self.config.use_joint_velocities:
                        include = False
                    elif "_effort" in name and not self.config.use_joint_torques:
                        include = False
                    elif "pressure" in name and not self.config.use_pressure_sensors:
                        include = False
                
                mask[i] = include
                if include:
                    filtered_names.append(name)
                    
            self._state_mask = mask
            self._state_feature_names = filtered_names
            
        logger.info(f"Using {np.sum(self._state_mask)}/{total_dims} state dimensions")
        logger.info(f"Filtered state features: {self._state_feature_names[:10]}...")  # Show first 10
        
    def _create_filtered_meta(self):
        """Create filtered metadata."""
        # Create a copy of the original metadata
        from copy import deepcopy
        self._filtered_meta = deepcopy(self.meta)
        
        # Update camera keys
        self._filtered_meta.camera_keys = self._filtered_cameras
        
        # Update state feature metadata
        state_dim = np.sum(self._state_mask)
        self._filtered_meta.features["observation.state"]["shape"] = (state_dim,)
        self._filtered_meta.features["observation.state"]["names"] = self._state_feature_names
        
        # Remove unused camera features
        features_to_remove = []
        for key in self._filtered_meta.features.keys():
            if key.startswith("observation.images."):
                camera_name = key.replace("observation.images.", "")
                if camera_name not in self._filtered_cameras:
                    features_to_remove.append(key)
                    
        for key in features_to_remove:
            del self._filtered_meta.features[key]
            
        logger.info(f"Filtered dataset will have {len(self._filtered_cameras)} cameras and {state_dim} state dimensions")
        
    @property
    def filtered_meta(self):
        """Get filtered metadata."""
        return self._filtered_meta
        
    def filter_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter a training batch to only include selected features.
        
        Args:
            batch: Original batch dictionary
            
        Returns:
            Filtered batch dictionary
        """
        filtered_batch = {}
        
        # Filter state observations
        if "observation.state" in batch:
            original_state = batch["observation.state"]
            filtered_state = original_state[:, self._state_mask]
            filtered_batch["observation.state"] = filtered_state
            
        # Filter camera observations
        for key, value in batch.items():
            if key.startswith("observation.images."):
                camera_name = key.replace("observation.images.", "")
                if camera_name in self._filtered_cameras:
                    filtered_batch[key] = value
            elif key == "observation.state":
                # Already handled above
                pass
            else:
                # Keep other keys (action, task, etc.)
                filtered_batch[key] = value
                
        return filtered_batch


def create_filtered_dataset_wrapper(dataset: LeRobotDataset, config: FeatureSelectionConfig):
    """
    Create a wrapper around a dataset that applies feature filtering.
    
    Args:
        dataset: Original LeRobot dataset
        config: Feature selection configuration
        
    Returns:
        Wrapped dataset with filtering applied
    """
    
    class FilteredDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, original_dataset, feature_filter):
            self.original_dataset = original_dataset
            self.feature_filter = feature_filter
            
        def __len__(self):
            return len(self.original_dataset)
            
        def __getitem__(self, idx):
            # Get original item
            item = self.original_dataset[idx]
            
            # Apply filtering
            filtered_item = self.feature_filter.filter_batch({k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in item.items()})
            
            # Remove batch dimension that we added
            for k, v in filtered_item.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    filtered_item[k] = v.squeeze(0)
                    
            return filtered_item
            
        @property
        def meta(self):
            return self.feature_filter.filtered_meta
            
        @property
        def num_frames(self):
            return self.original_dataset.num_frames
            
        @property
        def num_episodes(self):
            return self.original_dataset.num_episodes
            
        @property
        def episode_data_index(self):
            return self.original_dataset.episode_data_index
    
    feature_filter = FeatureFilter(dataset, config)
    return FilteredDatasetWrapper(dataset, feature_filter)
