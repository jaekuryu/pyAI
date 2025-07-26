# Update History

## 830fc13 jaekuryu 2025-07-26 10:15:57 -0400 Adding more options (advanced/automatic) for Threshold determination : pyMnet_gui.py based on docrules.md

- Added four new advanced threshold methods to the "Threshold Method" dropdown: Otsu, Percentile, K-Means, and Adaptive
- Implemented Otsu's method for automatic thresholding that minimizes intra-class variance
- Added Percentile-based thresholding using 75th percentile of feature energy distribution
- Implemented K-Means clustering (2-means) for threshold determination using simple iterative clustering
- Added Adaptive thresholding based on local statistics using rolling mean and standard deviation
- Updated AnalysisThread to accept and process the new threshold methods in classification logic
- Added helper methods _otsu_threshold, _kmeans_threshold, and _adaptive_threshold for calculations
- Enhanced threshold method selection to address 50:50 traffic/idle ratio issues
- Maintained backward compatibility with existing threshold methods (Median, Mean, Max-Mean)
- Improved spectrum sensing classification accuracy with more sophisticated thresholding algorithms
- Updated GUI to provide users with advanced options for fine-tuning classification sensitivity

## e58dca1 jaekuryu 2025-07-26 09:50:29 -0400 Adding Segment Size Combobox : pyMnet_gui.py based on docrules.md

- Added a "Segment Size" combo box to the Parameters section, allowing users to select the number of samples per segment
- Default segment size set to 50176 (224x224), with additional options: 16384, 32768, 65536, 131072
- Updated get_settings, set_settings, and reset_settings to include segment size
- AnalysisThread now accepts and uses the segment size parameter for segmentation
- MobileNetPreprocessingTab updated to use the selected segment size for segmenting IQ data
- All calls to update_preprocessing and analysis thread creation updated to pass segment size
- Ensured segment size is respected throughout the analysis and preprocessing pipeline

## 32afa0e jaekuryu 2025-07-26 09:05:45 -0400 Remove unnecessary GUI component in main window : pyMnet_gui.py based on docrules.md

- Removed redundant "Results" section from main control panel to eliminate duplication
- Eliminated Status, Classification, and Confidence labels from control panel
- Streamlined main window interface by removing duplicate information display
- Maintained comprehensive results display in dedicated "Results" and "Summary" tabs
- Improved GUI layout efficiency and reduced visual clutter
- Enhanced user experience by centralizing results in appropriate dedicated tabs
- Removed all associated label update calls in analysis workflow
- Simplified control panel to focus on essential parameter configuration
- Maintained full functionality while improving interface organization

## 3d378ea jaekuryu 2025-07-26 08:46:21 -0400 Plotting segmented spectrogram in Mobilenet Preprocess tab : pyMnet_gui.py based on docrules.md

- Modified MobileNet Preprocessing tab to display spectrograms for specific segments instead of full IQ data
- Added segment selection spin box with segment count display for user control
- Implemented segment-based spectrogram computation using selected segment data
- Added segment information display showing current segment number and total segments
- Enhanced plot titles to include segment-specific information (e.g., "Segment 2/15")
- Added segment duration and size annotations to preprocessing visualization
- Implemented parameter persistence to enable segment switching without re-analysis
- Added on_segment_changed method to handle dynamic segment selection updates
- Increased default GUI window height from 800 to 1000 pixels for better visualization
- Improved user experience with interactive segment-based spectrogram analysis
- Maintained all existing preprocessing steps (dB scale, normalization, 224x224 resizing)
- Enhanced MobileNet preprocessing pipeline visualization with segment-specific data

## 8c4d5e1 jaekuryu 2025-07-26 03:45:12 -0400 Parameterizing threshold setting method for feature energy classification based on docrules.md

- Added threshold method parameterization to the GUI control panel (Median, Mean, Max-Mean)
- Updated AnalysisThread to accept and use threshold_method parameter
- Modified threshold calculation logic to support different statistical methods
- Added threshold method display in Results tab and Summary tab
- Updated settings save/load functionality to include threshold method
- Enhanced feature energy visualization to show selected threshold method
- Maintained backward compatibility with default median threshold method
- Improved user control over spectrum sensing classification sensitivity
- Corrected Max-Mean threshold calculation to use max(energy) - mean(energy) for normalized data
- Reflected the selected threshold method in the Results and Summary tabs
- Updated settings save/load/reset to include the threshold method
- Improved user control and transparency for spectrum sensing classification

## 7a1b3c2 jaekuryu 2025-07-26 03:15:18 -0400 Consolidating Parameters and Settings into unified interface based on docrules.md

- Consolidated all parameters from Settings tab into the main Parameters panel
- Added Window Function and Model Type controls to the Parameters section
- Integrated save/load/reset functionality directly into the Parameters panel
- Removed redundant Settings tab to eliminate confusion and duplication
- Ensured all FFT and Hop parameters are consistently used across preprocessing and analysis
- Added real-time parameter synchronization for MobileNet preprocessing visualization
- Simplified interface by having all configuration in one accessible location
- Maintained all save/load functionality while improving user experience

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