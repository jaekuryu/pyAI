# MobileNet Spectrum Sensing

This project implements spectrum sensing using pretrained MobileNet for IQ data analysis. The script can analyze IQ data files and classify spectrum segments as "idle" or "traffic" using different approaches.

## Features

- **Pretrained MobileNet Integration**: Uses ImageNet-pretrained MobileNet v2 for feature extraction
- **Multiple Analysis Methods**: 
  - Feature-based analysis (recommended for immediate use)
  - Transfer learning classifier (requires training data)
- **Comprehensive Visualization**: Spectrograms, time/frequency domain plots, prediction distributions
- **IQ Data Support**: Reads binary IQ files with 4-byte I and 4-byte Q floating point values
- **Statistical Analysis**: Feature energy analysis and confidence scoring

## How Pretrained MobileNet Works for Spectrum Sensing

### 1. **Feature-Based Analysis (Default Method)**
- Uses pretrained MobileNet as a feature extractor
- Extracts high-level features from spectrograms
- Applies statistical analysis (feature energy) to classify segments
- **Advantage**: Works immediately without training
- **Best for**: Quick analysis and understanding of spectrum characteristics

### 2. **Transfer Learning Classifier**
- Uses pretrained MobileNet with a new classification head
- Requires labeled training data (idle vs traffic files)
- Fine-tunes the model for your specific spectrum sensing task
- **Advantage**: More accurate classification when trained
- **Best for**: Production use with labeled data

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Feature-Based Analysis)
```bash
python pyMnet.py
```

### Analyze Specific File
```bash
python pyMnet.py --file your_iq_file.bin
```

### Choose Analysis Method
```bash
# Use feature-based analysis (default, works immediately)
python pyMnet.py --method features

# Use transfer learning classifier (requires training)
python pyMnet.py --method classifier
```

### Training the Classifier (Advanced)
If you have labeled data, you can train the classifier for better results:

```python
# Example training code (add to your script)
idle_files = ['iq/idle_file1.bin', 'iq/idle_file2.bin']
traffic_files = ['iq/traffic_file1.bin', 'iq/traffic_file2.bin']

trained_model = train_spectrum_classifier(
    idle_files, traffic_files, transform, device, epochs=10
)
```

## File Structure

```
Mnet/
├── pyMnet.py          # Main spectrum sensing script
├── requirements.txt   # Python dependencies
├── README.md         # This file
└── iq/               # IQ data files directory
    ├── lte_20Mhz_rate23.04Mhz_dur_10ms_pci252_idle.bin
    └── ... (other IQ files)
```

## IQ Data Format

- **Format**: Binary file with 4-byte I and 4-byte Q floating point values
- **Structure**: `[I1, Q1, I2, Q2, I3, Q3, ...]`
- **Location**: Place IQ files in the `iq/` directory

## Output

The script provides:

1. **Spectrum Analysis Plots**:
   - Time domain (I/Q components)
   - Frequency domain (FFT)
   - Spectrogram
   - IQ constellation

2. **Spectrum Sensing Results**:
   - Per-segment classification (idle/traffic)
   - Confidence scores
   - Prediction distribution charts
   - Feature energy analysis
   - Average spectrogram visualization

## Understanding the Results

### Feature-Based Analysis
- **Feature Energy**: Higher energy typically indicates traffic
- **Confidence**: Distance from threshold indicates classification certainty
- **Threshold**: Automatically calculated as median of feature energies

### Classifier Method
- **Random Initialization**: Without training, results are random
- **Training Required**: For meaningful results, train on labeled data
- **Transfer Learning**: Leverages ImageNet features for better performance

## Technical Details

### MobileNet Architecture
- **Backbone**: MobileNet v2 pretrained on ImageNet
- **Input**: Single-channel spectrograms (224x224)
- **Features**: 1280-dimensional feature vectors
- **Adaptation**: First layer modified for single-channel input

### Spectrogram Processing
- **FFT Size**: 1024 points
- **Hop Size**: 256 points (50% overlap)
- **Window**: Hanning window
- **Resizing**: Bilinear interpolation to 224x224

### Feature Analysis
- **Energy Calculation**: Sum of squared feature values
- **Classification**: Threshold-based using median energy
- **Confidence**: Normalized distance from threshold

## Tips for Best Results

1. **Use Feature-Based Method**: For immediate analysis without training data
2. **Provide Training Data**: For production use, collect labeled idle/traffic files
3. **Adjust Parameters**: Modify FFT size, hop size, and segment length as needed
4. **Validate Results**: Compare with known spectrum characteristics
5. **Fine-tune Thresholds**: Adjust classification thresholds based on your data

## Troubleshooting

- **Empty Results**: Check IQ file format and path
- **Low Confidence**: Data may be ambiguous or noise-dominated
- **Poor Classification**: Consider training the classifier on labeled data
- **Memory Issues**: Reduce `max_samples` parameter for large files

## Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0
- pathlib2 >= 2.3.0

## License

This project is provided as-is for educational and research purposes. 