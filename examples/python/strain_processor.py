"""
Example: Using strain_utility with pysidecaster.py

This module shows how to integrate the strain_utility C++ extension
with your PySide6 application to display strain elastography.
"""

import numpy as np
from typing import Tuple, Optional
import sys
import time
import strain_utility # type: ignore


class StrainProcessor:
    """Wrapper for strain analysis using C++ StrainUtility library."""
    
    def __init__(self, n_lines: int, n_blocks: int, n_samples: int = 512, 
                 block_spacing: int = 4):
        """
        Initialize strain processor.
        
        Args:
            n_lines: Number of ultrasound lines
            n_blocks: Number of processing blocks per line
            n_samples: Samples per line
            block_spacing: Distance between block centers
        """
        self.n_lines = n_lines
        self.n_blocks = n_blocks
        self.n_samples = n_samples
        self.block_spacing = block_spacing
        self.block_length = n_samples // n_blocks
        
        # Initialize C++ strain utility
        self.strain_util = strain_utility.StrainUtility()
        self.strain_util.setCorrelationEstimator(1)  # Normalized correlation (0-1 range)
        self.strain_util.Reset(n_lines, n_blocks)
        
        # Configure processing parameters
        self.strain_util.threshold = 3              # Motion tracking search threshold
        self.strain_util.neighbours = 6             # Least square estimator neighbors
        self.strain_util.signPreserve = False       # Return positive strain only
        self.strain_util.SOPVar = 1                 # Use normalized correlation
        
        print(f"StrainUtility initialized: {n_lines} lines, {n_blocks} blocks, {n_samples} samples/line")
        print(f"  Block length: {self.block_length} samples")
        print(f"  Correlation estimator: {self.strain_util.getCorrelationEstimator()}")
        
        # Previous RF buffer (for frame-to-frame comparison)
        self.prev_rf_buffer: Optional[np.ndarray] = None
        self.frame_count = 0
        
    def process_rf_frame(self, rf_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Process RF ultrasound frame and estimate strain.
        
        Args:
            rf_frame: RF frame data (uint8), shape (n_lines * n_samples,) or (n_lines, n_samples)
            
        Returns:
            Tuple of:
                - axial_displacement: Axial motion map (n_lines, n_blocks)
                - strain: Strain estimation map (n_lines, n_blocks)
                - correlation: Motion correlation/quality map (n_lines, n_blocks)
                - avg_ncc: Average normalized cross-correlation (0-1)
        """
        frame_start = time.time()
        self.frame_count += 1
        
        # # Flatten if needed
        if rf_frame.ndim == 2:
            rf_frame = rf_frame.flatten()
        
        # # Ensure contiguous uint8 array
        rf_frame = np.ascontiguousarray(rf_frame, dtype=np.uint8)
        
        if self.prev_rf_buffer is None:
            # First frame - initialize and return zeros
            print(f"  Frame {self.frame_count}: Initializing buffer (returning zeros)")
            self.prev_rf_buffer = rf_frame.copy()
            axial_disp = np.zeros((self.n_lines, self.n_blocks), dtype=np.float32)
            strain = np.zeros((self.n_lines, self.n_blocks), dtype=np.float32)
            correlation = np.zeros((self.n_lines, self.n_blocks), dtype=np.float32)
            return axial_disp, strain, correlation, 0.0
        
        # Prepare output arrays - MUST BE 1D for TDPE
        print(f"Preparing output arrays")
        prep_start = time.time()
        total_size = self.n_lines * self.n_blocks
        axial_disp_flat = np.zeros(total_size, dtype=np.float32, order='C')
        lateral_disp_flat = np.zeros(total_size, dtype=np.float32, order='C')
        strain_flat = np.zeros(total_size, dtype=np.float32, order='C')
        correlation_flat = np.zeros(total_size, dtype=np.float32, order='C')
        search_map_flat = np.zeros(total_size, dtype=np.float32, order='C')
        prep_time = time.time() - prep_start
        
        # Create RFImageInfo structure
        print(f"Creating RFImageInfo structure")
        img_start = time.time()
        img_info = strain_utility.RFImageInfo()
        img_info.nLineNum = self.n_lines
        img_info.nBlockNum = self.n_blocks
        img_info.nBlockLength = self.block_length
        img_info.nLineLength = self.n_samples
        img_info.nBlockSpacing = self.block_spacing
        img_info.nStartLine = 1
        img_info.nEndLine = self.n_lines - 5
        img_info.nStartBlock = 5
        img_info.nEndBlock = self.n_blocks - 5
        img_info.fAmplify = 100.0  # Amplification for strain visualization
        img_info.uStrainLSE = 1    # Enable strain estimation
        img_info.bMedian = True    # Enable median filtering
        img_info.bLateral = False  # Lateral motion disabled for now
        img_info.bExtSearch = True # Extended search
        img_info.nKernelSize = 3   # Median filter kernel size
        img_time = time.time() - img_start
        
        # # Call C++ TDPE function
        print(f"Calling C++ TDPE function with: \nprev_rf_buffer: {self.prev_rf_buffer} \nrf_frame: {rf_frame}")
        tdpe_start = time.time()
        try:
            avg_ncc = self.strain_util.TDPE(
                self.prev_rf_buffer.astype(np.uint8),
                rf_frame.astype(np.uint8),
                axial_disp_flat.astype(np.float32),
                lateral_disp_flat.astype(np.float32),
                strain_flat.astype(np.float32),
                correlation_flat.astype(np.float32),
                search_map_flat.astype(np.float32),
                img_info
            )

            print(f"Made it past C++ function")

            tdpe_time = time.time() - tdpe_start
            print(f"  Frame {self.frame_count}: TDPE processed (avg_ncc={avg_ncc:.4f}, {tdpe_time*1000:.2f}ms)")
        except Exception as e:
            tdpe_time = time.time() - tdpe_start
            print(f"  Frame {self.frame_count}: TDPE error: {e} ({tdpe_time*1000:.2f}ms)")
            avg_ncc = 0.0
        
        # Update previous buffer for next frame
        buffer_start = time.time()
        self.prev_rf_buffer = rf_frame.copy()
        buffer_time = time.time() - buffer_start
        
        # Reshape outputs back to 2D for return
        reshape_start = time.time()
        axial_disp = axial_disp_flat.reshape((self.n_lines, self.n_blocks))
        strain = strain_flat.reshape((self.n_lines, self.n_blocks))
        correlation = correlation_flat.reshape((self.n_lines, self.n_blocks))
        reshape_time = time.time() - reshape_start
        
        total_time = time.time() - frame_start
        print(f"    Timings: prep={prep_time*1000:.2f}ms img_info={img_time*1000:.2f}ms buffer={buffer_time*1000:.2f}ms reshape={reshape_time*1000:.2f}ms | TOTAL={total_time*1000:.2f}ms")
        
        return axial_disp, strain, correlation, avg_ncc
        # return np.zeros(1), np.zeros(1), np.zeros(1), 0
    
    def normalize_for_display(self, data: np.ndarray, min_val: float | None=None, max_val: float | None=None) -> np.ndarray:
        """
        Normalize data to 0-255 for grayscale image display.
        
        Args:
            data: Input array
            min_val: Minimum value for normalization (auto if None)
            max_val: Maximum value for normalization (auto if None)
            
        Returns:
            Normalized uint8 array
        """
        norm_start = time.time()
        
        if min_val is None:
            min_v = float(np.nanmin(data))
        else:
            min_v = min_val

        if max_val is None:
            max_v = float(np.nanmax(data))
        else:
            max_v = max_val

        normalized = 255.0 * (np.clip(data, min_v, max_v) - min_v) / (max_v - min_v + 1e-6)
        result = normalized.astype(np.uint8)
        
        norm_time = time.time() - norm_start
        print(f"    normalize_for_display: {norm_time*1000:.2f}ms")

        return result
    
    def apply_colormap(self, data: np.ndarray, colormap: str = 'jet') -> np.ndarray:
        """
        Apply colormap to grayscale data (requires matplotlib).
        
        Args:
            data: Normalized uint8 array
            colormap: Colormap name ('jet', 'viridis', 'hot', etc.)
            
        Returns:
            RGB image array (H, W, 3) or RGBA (H, W, 4)
        """
        cmap_start = time.time()
        
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap)
            
            # Normalize to 0-1
            data_norm = data.astype(np.float32) / 255.0
            
            # Apply colormap
            colored = cmap(data_norm)
            
            # Convert to uint8 RGB
            result = (colored[:, :, :3] * 255).astype(np.uint8)
            
            cmap_time = time.time() - cmap_start
            print(f"    apply_colormap: {cmap_time*1000:.2f}ms")
            
            return result
        except ImportError:
            print("matplotlib not installed - returning grayscale")
            cmap_time = time.time() - cmap_start
            print(f"    apply_colormap (fallback): {cmap_time*1000:.2f}ms")
            return np.stack([data, data, data], axis=-1)


# Test/Demo code
if __name__ == "__main__":
    print("=" * 60)
    print("Strain Utility Test - Dummy Data Processing")
    print("=" * 60)
    
    # Create processor with small test dimensions
    print("\n1. Creating StrainProcessor...")
    processor = StrainProcessor(n_lines=64, n_blocks=32, n_samples=256, block_spacing=4)
    
    # Generate dummy RF frames (simulated ultrasound data)
    print("\n2. Generating dummy RF frames...")
    frame_size = processor.n_lines * processor.n_samples
    
    # Frame 1: baseline random RF data
    frame1 = np.random.randint(0, 256, frame_size, dtype=np.uint8)
    print(f"  Frame 1: shape={frame1.shape}, dtype={frame1.dtype}, min={frame1.min()}, max={frame1.max()}")
    
    # Frame 2: slightly modified (simulates tissue motion)
    frame2 = np.random.randint(0, 256, frame_size, dtype=np.uint8)
    print(f"  Frame 2: shape={frame2.shape}, dtype={frame2.dtype}, min={frame2.min()}, max={frame2.max()}")
    
    # Process frame 1 (baseline)
    print("\n3. Processing Frame 1 (baseline)...")
    axial_1, strain_1, corr_1, ncc_1 = processor.process_rf_frame(frame1)
    print(f"  Axial displacement: shape={axial_1.shape}, mean={axial_1.mean():.4f}, max={axial_1.max():.4f}")
    print(f"  Strain: shape={strain_1.shape}, mean={strain_1.mean():.4f}, max={strain_1.max():.4f}")
    print(f"  Correlation: shape={corr_1.shape}, mean={corr_1.mean():.4f}, max={corr_1.max():.4f}")
    print(f"  Avg NCC: {ncc_1:.4f}")
    
    # Process frame 2
    print("\n4. Processing Frame 2...")
    axial_2, strain_2, corr_2, ncc_2 = processor.process_rf_frame(frame2)
    print(f"  Axial displacement: shape={axial_2.shape}, mean={axial_2.mean():.4f}, max={axial_2.max():.4f}")
    print(f"  Strain: shape={strain_2.shape}, mean={strain_2.mean():.4f}, max={strain_2.max():.4f}")
    print(f"  Correlation: shape={corr_2.shape}, mean={corr_2.mean():.4f}, max={corr_2.max():.4f}")
    print(f"  Avg NCC: {ncc_2:.4f}")
    
    # Demonstrate normalization and colormap
    print("\n5. Testing visualization functions...")
    strain_norm = processor.normalize_for_display(strain_2, 0, 2.0)
    print(f"  Normalized strain: dtype={strain_norm.dtype}, shape={strain_norm.shape}, min={strain_norm.min()}, max={strain_norm.max()}")
    
    try:
        strain_rgb = processor.apply_colormap(strain_norm, 'viridis')
        print(f"  Colormap applied: shape={strain_rgb.shape}, dtype={strain_rgb.dtype}")
    except Exception as e:
        print(f"  Colormap failed: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


# Example usage in PySide6 app:
"""
In your pysidecaster.py, add this to process RF data:

from strain_processor import StrainProcessor
from PIL import Image
from PySide6 import QtGui

# In MainWidget.__init__:
self.strain_processor = StrainProcessor(n_lines=256, n_blocks=128, n_samples=512)

# In the raw image callback or a separate processing thread:
def newRawImage(image: bytes, lines: int, samples: int, bps: int, 
                axial: float, lateral: float, timestamp: int, 
                jpg_size: int, is_rf: int, angle: float):
    
    # Convert raw bytes to numpy array
    rf_data = np.frombuffer(image, dtype=np.uint8)
    
    # Process strain
    axial_disp, strain, correlation, avg_ncc = self.strain_processor.process_rf_frame(rf_data)
    
    # Normalize for display
    strain_display = self.strain_processor.normalize_for_display(strain, 0, 2.0)
    
    # Apply colormap
    strain_rgb = self.strain_processor.apply_colormap(strain_display, 'jet')
    
    # Convert to QImage
    h, w = strain_rgb.shape[:2]
    img = QtGui.QImage(strain_rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    
    # Display in a new widget or update existing one
    self.strain_view.updateImage(img)
"""
