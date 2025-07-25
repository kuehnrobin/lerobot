#!/usr/bin/env python

"""
Copyright Robin Kühn 2024
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


def make_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj


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
            # Convert camera names to full keys (e.g., cam_left_head -> observation.images.cam_left_head)
            requested_cameras = [f"observation.images.{cam}" if not cam.startswith("observation.images.") else cam 
                               for cam in self.config.cameras]
            self._filtered_cameras = [cam for cam in requested_cameras if cam in available_cameras]
            if len(self._filtered_cameras) != len(requested_cameras):
                missing = set(requested_cameras) - set(available_cameras)
                logger.warning(f"Cameras not found in dataset: {missing}")
                
        elif self.config.exclude_cameras is not None:
            # Use all cameras except excluded ones
            # Convert camera names to full keys
            excluded_cameras = [f"observation.images.{cam}" if not cam.startswith("observation.images.") else cam 
                              for cam in self.config.exclude_cameras]
            self._filtered_cameras = [cam for cam in available_cameras if cam not in excluded_cameras]
            
        else:
            # Use all cameras
            self._filtered_cameras = available_cameras
            
        logger.info(f"Using cameras: {self._filtered_cameras}")
        
    def _compute_state_filter(self):
        """Compute state dimension filtering mask."""
        # Define the exact state vector structure based on the dataset conversion
        # Total 82D state vector:
        # 0-6:    left_arm qpos (7D)
        # 7-13:   left_arm qvel (7D)  
        # 14-20:  left_arm torque (7D)
        # 21-27:  right_arm qpos (7D)
        # 28-34:  right_arm qvel (7D)
        # 35-41:  right_arm torque (7D)
        # 42-48:  left_hand qpos (7D)
        # 49-60:  left_hand pressures (12D)
        # 61-67:  right_hand qpos (7D)
        # 68-79:  right_hand pressures (12D)
        # 80-81:  camera qpos (2D)
        
        state_names = self.meta.features["observation.state"]["names"]
        total_dims = len(state_names)
        
        # Get the actual state shape from the dataset to ensure compatibility
        actual_state_shape = self.meta.features["observation.state"]["shape"]
        actual_dims = actual_state_shape[0] if actual_state_shape else total_dims
        
        logger.info(f"State names suggest {total_dims} dimensions, actual state shape: {actual_dims}")
        
        if self.config.custom_state_indices is not None:
            # Use custom indices
            self._state_mask = np.zeros(actual_dims, dtype=bool)
            valid_indices = [i for i in self.config.custom_state_indices if i < actual_dims]
            self._state_mask[valid_indices] = True
            self._state_feature_names = [state_names[i] for i in valid_indices if i < len(state_names)]
            
        else:
            # Build mask based on known state structure and feature configuration
            mask = np.zeros(actual_dims, dtype=bool)
            filtered_names = []
            
            # Define state structure mapping
            state_structure = {
                # Arms (positions, velocities, torques)
                'left_arm_qpos': (0, 7),      # 0-6
                'left_arm_qvel': (7, 14),     # 7-13
                'left_arm_torque': (14, 21),  # 14-20
                'right_arm_qpos': (21, 28),   # 21-27
                'right_arm_qvel': (28, 35),   # 28-34
                'right_arm_torque': (35, 42), # 35-41
                
                # Hands (positions and pressures)
                'left_hand_qpos': (42, 49),   # 42-48
                'left_hand_pressure': (49, 61), # 49-60
                'right_hand_qpos': (61, 68),  # 61-67
                'right_hand_pressure': (68, 80), # 68-79
                
                # Camera
                'camera_qpos': (80, 82),      # 80-81
            }
            
            # Apply filtering based on configuration
            for feature_name, (start, end) in state_structure.items():
                include_feature = True
                
                # Check feature type filtering
                if 'qpos' in feature_name and not self.config.use_joint_positions:
                    include_feature = False
                    logger.debug(f"Excluding {feature_name} - joint positions disabled")
                elif 'qvel' in feature_name and not self.config.use_joint_velocities:
                    include_feature = False
                    logger.debug(f"Excluding {feature_name} - joint velocities disabled")
                elif 'torque' in feature_name and not self.config.use_joint_torques:
                    include_feature = False
                    logger.debug(f"Excluding {feature_name} - joint torques disabled")
                elif 'pressure' in feature_name and not self.config.use_pressure_sensors:
                    include_feature = False
                    logger.debug(f"Excluding {feature_name} - pressure sensors disabled")
                elif 'camera' in feature_name:
                    # Special handling for camera positions
                    include_feature = self._should_include_camera_positions()
                    logger.info(f"Camera position feature '{feature_name}' - include: {include_feature}")
                
                # Apply the mask for this feature range
                if include_feature and start < actual_dims:
                    end_idx = min(end, actual_dims)
                    mask[start:end_idx] = True
                    for i in range(start, end_idx):
                        if i < len(state_names):
                            filtered_names.append(state_names[i])
                        else:
                            filtered_names.append(f"{feature_name}_{i-start}")
                    logger.debug(f"Including {feature_name}: indices {start}:{end_idx}")
                else:
                    logger.debug(f"Excluding {feature_name}: indices {start}:{end}")
                    
            self._state_mask = mask
            self._state_feature_names = filtered_names
            
        logger.info(f"Using {np.sum(self._state_mask)}/{actual_dims} state dimensions")
        logger.info(f"Filtered state features: {self._state_feature_names[:10]}...")  # Show first 10
        logger.info(f"State structure breakdown:")
        logger.info(f"  - Arm positions: {'✓' if self.config.use_joint_positions else '✗'}")
        logger.info(f"  - Arm velocities: {'✓' if self.config.use_joint_velocities else '✗'}")
        logger.info(f"  - Arm torques: {'✓' if self.config.use_joint_torques else '✗'}")
        logger.info(f"  - Pressure sensors: {'✓' if self.config.use_pressure_sensors else '✗'}")
        logger.info(f"  - Camera positions: {'✓' if self._should_include_camera_positions() else '✗'}")
        
        
    def _create_filtered_meta(self):
        """Create filtered metadata."""
        # Create a copy of the original metadata
        from copy import deepcopy
        self._filtered_meta = deepcopy(self.meta)
        
        # Update state feature metadata
        state_dim = int(np.sum(self._state_mask))  # Convert to regular Python int
        self._filtered_meta.features["observation.state"]["shape"] = (state_dim,)
        # Ensure feature names are JSON-serializable
        self._filtered_meta.features["observation.state"]["names"] = [
            str(name) if hasattr(name, 'item') else name for name in self._state_feature_names
        ]
        
        # Remove unused camera features from the features dict
        # This will automatically update the camera_keys property
        features_to_remove = []
        for key in self._filtered_meta.features.keys():
            if key.startswith("observation.images."):
                if key not in self._filtered_cameras:
                    features_to_remove.append(key)
                    
        for key in features_to_remove:
            del self._filtered_meta.features[key]
        
        # Update stats if they exist - filter the state statistics to match filtered dimensions
        if hasattr(self._filtered_meta, 'stats') and self._filtered_meta.stats is not None:
            original_stats = self._filtered_meta.stats
            if "observation.state" in original_stats:
                original_state_stats = original_stats["observation.state"]
                # Filter the statistics to match the filtered state dimensions
                filtered_state_stats = {}
                for stat_key, stat_value in original_state_stats.items():
                    if hasattr(stat_value, '__len__') and len(stat_value) == len(self._state_mask):
                        # This is a per-dimension statistic, filter it
                        filtered_value = stat_value[self._state_mask]
                        # Keep as numpy array/tensor for runtime use
                        filtered_state_stats[stat_key] = filtered_value
                    else:
                        # This is a scalar statistic, keep as-is
                        filtered_state_stats[stat_key] = stat_value
                original_stats["observation.state"] = filtered_state_stats
            
            # Remove camera statistics for excluded cameras
            for key in list(original_stats.keys()):
                if key.startswith("observation.images.") and key not in self._filtered_cameras:
                    del original_stats[key]
            
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
            # Safety check: ensure mask dimensions match the actual state tensor
            if original_state.shape[-1] != len(self._state_mask):
                logger.warning(f"State dimension mismatch: tensor has {original_state.shape[-1]} dims, mask has {len(self._state_mask)} dims")
                # Adjust mask to match actual tensor size
                actual_dims = original_state.shape[-1]
                if actual_dims < len(self._state_mask):
                    # Truncate mask
                    adjusted_mask = self._state_mask[:actual_dims]
                else:
                    # Extend mask with True values
                    adjusted_mask = np.concatenate([self._state_mask, np.ones(actual_dims - len(self._state_mask), dtype=bool)])
                filtered_state = original_state[:, adjusted_mask]
            else:
                filtered_state = original_state[:, self._state_mask]
            filtered_batch["observation.state"] = filtered_state
            
        # Filter camera observations
        for key, value in batch.items():
            if key.startswith("observation.images."):
                if key in self._filtered_cameras:
                    filtered_batch[key] = value
            elif key == "observation.state":
                # Already handled above
                pass
            else:
                # Keep other keys (action, task, etc.)
                filtered_batch[key] = value
                
        return filtered_batch
    
    def _should_include_camera_positions(self):
        """
        Determine if camera positions should be included in the state vector.
        Camera positions should be included if active cameras are selected.
        """
        # Check if any active cameras are included in the filtered cameras
        active_camera_names = [
            "observation.images.cam_left_active",
            "observation.images.cam_right_active"
        ]
        
        # Check if any active cameras are in the filtered cameras list
        has_active_cameras = any(cam in self._filtered_cameras for cam in active_camera_names)
        
        logger.info(f"Camera position inclusion check:")
        logger.info(f"  - Filtered cameras: {self._filtered_cameras}")
        logger.info(f"  - Active camera names: {active_camera_names}")
        logger.info(f"  - Has active cameras: {has_active_cameras}")
        
        if has_active_cameras:
            logger.info("Active cameras detected in filtered cameras - including camera positions in state")
            return True
        else:
            logger.info("No active cameras in filtered cameras - excluding camera positions from state")
            return False

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
        def stats(self):
            """Return filtered stats from the original dataset."""
            if hasattr(self.original_dataset, 'stats'):
                # Apply filtering to the original dataset's stats
                original_stats = self.original_dataset.stats
                if original_stats is None:
                    return None
                    
                from copy import deepcopy
                filtered_stats = deepcopy(original_stats)
                
                # Filter state statistics
                if "observation.state" in filtered_stats:
                    original_state_stats = filtered_stats["observation.state"]
                    filtered_state_stats = {}
                    for stat_key, stat_value in original_state_stats.items():
                        if hasattr(stat_value, '__len__') and len(stat_value) == len(self.feature_filter._state_mask):
                            # This is a per-dimension statistic, filter it
                            filtered_value = stat_value[self.feature_filter._state_mask]
                            # Keep as numpy array/tensor for runtime use
                            filtered_state_stats[stat_key] = filtered_value
                        else:
                            # This is a scalar statistic, keep as-is
                            filtered_state_stats[stat_key] = stat_value
                    filtered_stats["observation.state"] = filtered_state_stats
                
                # Remove camera statistics for excluded cameras
                for key in list(filtered_stats.keys()):
                    if key.startswith("observation.images.") and key not in self.feature_filter._filtered_cameras:
                        del filtered_stats[key]
                        
                return filtered_stats
            return None
            
        @property
        def camera_keys(self):
            return self.feature_filter.filtered_meta.camera_keys
            
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
