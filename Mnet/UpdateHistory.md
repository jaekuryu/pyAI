# Update History

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