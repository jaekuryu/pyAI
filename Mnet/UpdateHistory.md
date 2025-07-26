# Update History

## 5b02744 jaekuryu 2025-07-26 02:49:42 -0400 Fixing Parameter application issue and Analysis Algorithm Issue based on docrules.md

- Fixed core analysis algorithm discrepancy between GUI and command-line versions
- Replaced simulated analysis with proper segment-based spectrum sensing approach
- Implemented real feature extraction using MobileNetV2 model with single-channel input
- Added statistical classification based on feature energy thresholding
- Fixed channel mismatch error by ensuring single-channel spectrogram processing
- Corrected MobileNet preprocessing tab to use full IQ data instead of single segment
- Aligned hop_size parameter (256) with pyMnet.py reference implementation
- Implemented dynamic parameter updates for FFT and Hop size in preprocessing visualization
- Added on_parameter_changed method to handle real-time parameter updates
- Updated Summary tab to display real analysis data including average spectrogram
- Fixed ValueError issues in summary plotting with proper array length checks
- Set default FFT size to 1024 across all GUI components for consistency
- Enhanced Results tab to show detailed segment analysis and feature statistics
- Improved preprocessing visualization with correct parameter application

## 33f8cf2 jaekuryu 2025-07-26 02:07:31 -0400 Adding Mobilenet Preprocessing Tab : pyMnet_gui.py based on docrules.md

- Added new "MobileNet Preprocessing" tab to visualize spectrogram preprocessing steps
- Implemented 1x3 grid layout showing three preprocessing stages:
  - Step 1: dB Scale Spectrogram conversion
  - Step 2: Normalized [0,1] spectrogram scaling
  - Step 3: 224x224 pixel resizing for MobileNet input
- Integrated PyTorch interpolation for accurate spectrogram resizing
- Added text annotations showing original and final spectrogram dimensions
- Positioned tab before "Results" tab in the interface order
- Optimized font sizes and layout spacing for better visualization
- Enhanced MobileNet feature extraction pipeline visualization

## 1408431 jaekuryu 2025-07-25 21:13:47 -0400 Adding a set of Analysis Plots : pyMnet_gui.py

- Added new "Summary" tab with comprehensive analysis plots
- Implemented 1x4 grid layout for four key analysis plots:
  - Traffic/Idle Bar Chart: Shows distribution of spectrum states
  - Feature Energy Over Time: Displays energy variations over time
  - Confidence vs Feature Energy: Scatter plot of confidence vs energy correlation
  - Average Spectrogram: Time-frequency representation of signal characteristics
- Optimized font sizes (8pt titles, 7pt labels) to prevent plot overlapping
- Removed main figure title to improve layout clarity
- Integrated summary updates with main analysis workflow
- Enhanced visualization for spectrum sensing evaluation

## f2e6ec3 jaekuryu 2025-07-25 20:42:15 -0400 First GUI version : pyMnet_gui.py

- Initial implementation of PyQt6-based GUI for spectrum sensing
- Multi-tab interface with Time Domain, Frequency Domain, Spectrogram, Constellation, Results, and Settings tabs
- File dialog integration for IQ data selection
- Background analysis thread to prevent GUI freezing
- Integration with existing pyMnet.py core functionality
- MobileNet v2 pretrained model integration for spectrum classification
- Comprehensive plotting capabilities with Matplotlib integration
- Parameter configuration interface
- Progress tracking and status updates 