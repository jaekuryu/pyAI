# Spectrum Sensing Process Documentation

## Overview

This document explains the complete pipeline for spectrum sensing using MobileNet, from raw IQ data preprocessing through machine learning application to final output generation.

## 1. Data Input

### IQ Data Format
- **File Format**: Binary files containing IQ (In-phase and Quadrature) data
- **Data Structure**: Repetition of 4-byte I and 4-byte Q floating-point values
- **File Example**: `lte_20Mhz_rate23.04Mhz_dur_10ms_pci252_idle.bin`
- **Sample Rate**: 23.04 MHz (configurable)
- **Duration**: Variable (typically 10ms for LTE signals)

### Data Loading
```python
def read_iq_file(file_path, max_samples=None):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    
    # Reshape to (N, 2) where N is number of samples, 2 for I and Q
    iq_data = data.reshape(-1, 2)
    return iq_data
```

## 2. Preprocessing Pipeline

### 2.1 Complex Signal Conversion
**Purpose**: Convert separate I/Q components to complex signal representation

```python
complex_data = iq_data[:, 0] + 1j * iq_data[:, 1]
```

### 2.2 Spectrogram Generation
**Purpose**: Convert time-domain signal to time-frequency representation suitable for CNN analysis

#### STFT (Short-Time Fourier Transform) Parameters:
- **FFT Size**: 1024 samples (configurable: 256, 512, 1024, 2048, 4096)
- **Hop Size**: 256 samples (configurable: 64-2048)
- **Window Function**: Hanning window
- **Frequency Resolution**: ~22.5 kHz (23.04 MHz / 1024)
- **Time Resolution**: ~11.1 μs (256 / 23.04 MHz)

#### STFT Process:
```python
def compute_spectrogram(self, complex_data):
    window = np.hanning(self.fft_size)
    stft = []
    
    for i in range(0, len(complex_data) - self.fft_size, self.hop_size):
        segment = complex_data[i:i + self.fft_size]
        if len(segment) == self.fft_size:
            windowed_segment = segment * window
            fft_result = np.fft.fft(windowed_segment)
            shifted_fft = np.fft.fftshift(fft_result)  # Center frequencies
            stft.append(np.abs(shifted_fft))
    
    spectrogram = np.array(stft).T  # Shape: (freq_bins, time_frames)
    return spectrogram
```

**Key Features**:
- **Frequency Centering**: `np.fft.fftshift` ensures negative frequencies on left, positive on right
- **Magnitude Calculation**: `np.abs()` extracts magnitude information
- **Transpose**: Results in frequency bins on y-axis, time frames on x-axis

### 2.3 MobileNet Input Preparation

#### 2.3.1 Size Normalization
**Purpose**: Resize spectrogram to MobileNet's fixed input size (224x224)

```python
# Resize to standard MobileNet input size
spectrogram_tensor = F.interpolate(
    spectrogram_tensor.unsqueeze(0), 
    size=(224, 224), 
    mode='bilinear', 
    align_corners=False
).squeeze(0)
```

#### 2.3.2 Amplitude Scaling
**Purpose**: Convert to dB scale for better dynamic range

```python
spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
```

#### 2.3.3 Normalization
**Purpose**: Scale values to [0, 1] range

```python
spectrogram_normalized = (spectrogram_db - spectrogram_db.min()) / 
                        (spectrogram_db.max() - spectrogram_db.min())
```

#### 2.3.4 RGB Conversion
**Purpose**: Convert grayscale spectrogram to RGB format for MobileNet

```python
spectrogram_rgb = np.stack([spectrogram_normalized] * 3, axis=-1)
```

#### 2.3.5 MobileNet Preprocessing
**Purpose**: Apply standard ImageNet normalization

```python
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

## 3. Machine Learning Application

### 3.1 Model Architecture
**Model**: MobileNet v2 (pretrained on ImageNet)
**Purpose**: Feature extraction for spectrum classification

```python
# Load pretrained MobileNet
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()

# Remove classification layer for feature extraction
model.classifier = torch.nn.Identity()
```

### 3.2 Feature Extraction
**Input**: 224x224x3 RGB spectrogram image
**Output**: 1280-dimensional feature vector

```python
with torch.no_grad():
    features = model(img)  # Shape: (1, 1280)
```

### 3.3 Classification Logic
**Current Implementation**: Heuristic-based classification using feature statistics

```python
# Calculate feature statistics
feature_mean = features.mean().item()
feature_std = features.std().item()

# Heuristic classification
if feature_std > 0.1:  # High variance suggests traffic
    classification = "Traffic"
    confidence = min(0.9, feature_std * 2)
else:
    classification = "Idle"
    confidence = min(0.9, (1 - feature_std) * 2)
```

**Classification Criteria**:
- **Traffic**: High feature variance (feature_std > 0.1)
- **Idle**: Low feature variance (feature_std ≤ 0.1)
- **Confidence**: Scaled based on feature statistics

## 4. Output Generation

### 4.1 Primary Results
```python
results = {
    'spectrogram': spectrogram,           # Original spectrogram data
    'features': features.numpy(),         # 1280-dim feature vector
    'classification': classification,     # "Idle" or "Traffic"
    'confidence': confidence,             # Confidence score [0, 1]
    'feature_mean': feature_mean,         # Mean of feature vector
    'feature_std': feature_std           # Std of feature vector
}
```

### 4.2 Visualization Outputs

#### 4.2.1 Time Domain Plot
- **Content**: I and Q components over time
- **Purpose**: Visualize raw signal characteristics
- **Configurable**: Sample range (100-10000 samples)

#### 4.2.2 Frequency Domain Plot
- **Content**: Power spectral density
- **Purpose**: Show frequency content and symmetry
- **Features**: dB scale option, configurable FFT size
- **Key**: Proper frequency centering around 0 Hz

#### 4.2.3 Spectrogram Plot
- **Content**: Time-frequency representation
- **Purpose**: Visualize spectral evolution over time
- **Features**: Configurable FFT size and overlap
- **Key**: Proper frequency axis with negative/positive frequencies

#### 4.2.4 Constellation Plot
- **Content**: I vs Q scatter plot
- **Purpose**: Visualize signal modulation characteristics
- **Configurable**: Sample count for visualization

#### 4.2.5 Summary Tab (4-plot layout)
- **Traffic/Idle Bar Chart**: Distribution of spectrum states
- **Feature Energy Over Time**: Energy variations over time
- **Confidence vs Feature Energy**: Correlation scatter plot
- **Average Spectrogram**: Time-frequency representation

### 4.3 GUI Integration
- **Multi-tab Interface**: Organized visualization and control
- **Real-time Updates**: Plots update when new data is loaded
- **Parameter Configuration**: Adjustable analysis parameters
- **Progress Tracking**: Visual feedback during analysis
- **Export Capabilities**: Save results and plots

## 5. Key Design Decisions

### 5.1 Single File Processing
- **Approach**: Entire IQ file → Single spectrogram → Single classification
- **Advantages**: Simple, fast, global analysis
- **Disadvantages**: No temporal granularity

### 5.2 Frequency Centering
- **Implementation**: `np.fft.fftshift` applied to both FFT results and spectrogram
- **Purpose**: Ensure symmetrical display around 0 Hz for complex signals
- **Critical**: Essential for proper spectrum visualization

### 5.3 MobileNet Transfer Learning
- **Strategy**: Use pretrained ImageNet weights for feature extraction
- **Rationale**: Leverage learned visual patterns for spectral analysis
- **Adaptation**: RGB conversion and normalization for compatibility

### 5.4 Heuristic Classification
- **Current**: Simple threshold-based classification
- **Future**: Can be replaced with trained classifier on spectrum features
- **Flexibility**: Easy to modify classification criteria

## 6. Performance Considerations

### 6.1 Computational Complexity
- **STFT**: O(N log N) per window, where N = FFT size
- **MobileNet**: ~3.5M parameters, efficient inference
- **Memory**: Spectrogram storage scales with file size

### 6.2 Real-time Capabilities
- **Current**: Batch processing of complete files
- **Potential**: Streaming processing with sliding windows
- **Optimization**: GPU acceleration for MobileNet inference

### 6.3 Accuracy Factors
- **Signal Quality**: SNR affects spectrogram clarity
- **Parameter Tuning**: FFT size, hop size, window function
- **Classification Thresholds**: Feature variance thresholds
- **Model Adaptation**: Fine-tuning for specific signal types

## 7. Future Enhancements

### 7.1 Segmentation Analysis
- **Multi-segment Processing**: Break files into overlapping segments
- **Temporal Resolution**: Track spectrum changes over time
- **Aggregation Methods**: Voting, averaging, or sequence models

### 7.2 Advanced Classification
- **Trained Classifier**: Replace heuristic with learned classifier
- **Multi-class Support**: Distinguish between different signal types
- **Confidence Calibration**: Improve confidence estimation

### 7.3 Real-time Processing
- **Streaming Pipeline**: Process data as it arrives
- **Buffer Management**: Handle continuous data streams
- **Latency Optimization**: Minimize processing delays

### 7.4 Model Optimization
- **Quantization**: Reduce model size and inference time
- **Pruning**: Remove unnecessary model parameters
- **Custom Architectures**: Design models specifically for spectrum sensing 