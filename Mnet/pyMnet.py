#!/usr/bin/env python3
"""
Spectrum Sensing using MobileNet
This script reads IQ data files and performs spectrum sensing using MobileNet.
IQ data format: repetition of 4-byte I and 4-byte Q floating point values.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from scipy import signal
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import argparse

class IQDataset(Dataset):
    """Dataset class for IQ data"""
    def __init__(self, iq_data, labels=None, transform=None):
        self.iq_data = iq_data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.iq_data)
    
    def __getitem__(self, idx):
        sample = self.iq_data[idx]
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample

class IQTransform:
    """Transform IQ data to spectrogram-like format for MobileNet"""
    def __init__(self, fft_size=1024, hop_size=256):
        self.fft_size = fft_size
        self.hop_size = hop_size
    
    def __call__(self, iq_data):
        # Convert to complex numbers
        complex_data = iq_data[:, 0] + 1j * iq_data[:, 1]
        
        # Compute spectrogram
        spectrogram = self.compute_spectrogram(complex_data)
        
        # Convert to tensor and normalize
        spectrogram_tensor = torch.from_numpy(spectrogram).float()
        spectrogram_tensor = spectrogram_tensor.unsqueeze(0)  # Add channel dimension
        
        # Resize to standard MobileNet input size (224x224)
        spectrogram_tensor = F.interpolate(spectrogram_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        # Normalize to [0, 1] range
        spectrogram_tensor = (spectrogram_tensor - spectrogram_tensor.min()) / (spectrogram_tensor.max() - spectrogram_tensor.min() + 1e-8)
        
        return spectrogram_tensor
    
    def compute_spectrogram(self, complex_data):
        """Compute spectrogram from complex IQ data"""
        # Apply window function
        window = np.hanning(self.fft_size)
        
        # Compute STFT
        stft = []
        for i in range(0, len(complex_data) - self.fft_size, self.hop_size):
            segment = complex_data[i:i + self.fft_size]
            if len(segment) == self.fft_size:
                windowed_segment = segment * window
                fft_result = np.fft.fft(windowed_segment)
                # Use fftshift to center the frequencies properly
                # This ensures negative frequencies are on the left, positive on the right
                shifted_fft = np.fft.fftshift(fft_result)
                stft.append(np.abs(shifted_fft))
        
        if not stft:
            # If no valid segments, create a dummy spectrogram
            stft = [np.zeros(self.fft_size)]
        
        spectrogram = np.array(stft).T  # Transpose to get frequency on y-axis
        
        # The spectrogram now has frequencies properly centered:
        # Negative frequencies on top, positive frequencies on bottom
        return spectrogram

class MobileNetSpectrumClassifier(nn.Module):
    """MobileNet-based spectrum classifier using transfer learning"""
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetSpectrumClassifier, self).__init__()
        # Load pretrained MobileNet
        mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)
        
        # Modify first layer for single-channel input (spectrograms)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Use features as backbone
        self.features = mobilenet.features
        
        # Add new classification head for spectrum sensing
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Freeze feature layers initially (optional - can be unfrozen for fine-tuning)
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def unfreeze_features(self):
        """Unfreeze feature layers for fine-tuning"""
        for param in self.features.parameters():
            param.requires_grad = True

class MobileNetFeatureExtractor(nn.Module):
    """MobileNet feature extractor (removes classifier head)"""
    def __init__(self, pretrained=True):
        super(MobileNetFeatureExtractor, self).__init__()
        mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

def read_iq_file(file_path, max_samples=None):
    """
    Read IQ data from binary file
    Format: repetition of 4-byte I and 4-byte Q floating point values
    """
    try:
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
        
        # Reshape to (N, 2) where N is number of samples, 2 for I and Q
        if len(data) % 2 != 0:
            print(f"Warning: Odd number of values in {file_path}, truncating last value")
            data = data[:-1]
        
        iq_data = data.reshape(-1, 2)
        
        if max_samples and len(iq_data) > max_samples:
            iq_data = iq_data[:max_samples]
        
        print(f"Loaded {len(iq_data)} IQ samples from {file_path}")
        return iq_data
    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def segment_iq_data(iq_data, segment_length=4096, overlap=0.5):
    """Segment IQ data into overlapping segments"""
    hop_size = int(segment_length * (1 - overlap))
    segments = []
    
    for i in range(0, len(iq_data) - segment_length, hop_size):
        segment = iq_data[i:i + segment_length]
        segments.append(segment)
    
    return segments

def analyze_spectrum(iq_data, sample_rate=23.04e6, title="Spectrum Analysis"):
    """Analyze and plot spectrum of IQ data"""
    # Convert to complex
    complex_data = iq_data[:, 0] + 1j * iq_data[:, 1]
    
    # Compute FFT
    fft_size = min(4096, len(complex_data))
    fft_result = np.fft.fft(complex_data[:fft_size])
    frequencies = np.fft.fftfreq(fft_size, 1/sample_rate)
    
    # Apply fftshift to center frequencies for plotting the frequency domain
    shifted_fft_result = np.fft.fftshift(fft_result)
    shifted_frequencies = np.fft.fftshift(frequencies)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Time domain
    plt.subplot(2, 2, 1)
    plt.plot(iq_data[:1000, 0], label='I')
    plt.plot(iq_data[:1000, 1], label='Q')
    plt.title('Time Domain (First 1000 samples)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # Frequency domain
    plt.subplot(2, 2, 2)
    # For IQ data, show the full frequency range from -fs/2 to +fs/2
    plt.plot(shifted_frequencies / 1e6, 20 * np.log10(np.abs(shifted_fft_result)))
    plt.title('Frequency Domain')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.grid(True)
    
    # Spectrogram
    plt.subplot(2, 2, 3)
    # For IQ data, we need to show the full frequency range from -fs/2 to +fs/2
    f, t, Sxx = signal.spectrogram(complex_data, sample_rate, nperseg=1024, noverlap=512, 
                                   return_onesided=False)
    
    # Apply fftshift to frequencies and spectrogram data (Sxx) along the frequency axis (axis 0)
    shifted_f = np.fft.fftshift(f)
    shifted_Sxx = np.fft.fftshift(Sxx, axes=0)
    
    # Convert frequencies to MHz - for complex IQ data, frequencies go from -fs/2 to +fs/2
    f_mhz = shifted_f / 1e6
    plt.pcolormesh(t, f_mhz, 10 * np.log10(shifted_Sxx))
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.colorbar(label='Power (dB)')
    
    # IQ constellation
    plt.subplot(2, 2, 4)
    plt.scatter(iq_data[:1000, 0], iq_data[:1000, 1], alpha=0.6, s=1)
    plt.title('IQ Constellation (First 1000 samples)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

def extract_features(model, iq_data, transform, device):
    """Extract features for each IQ segment using MobileNet"""
    segments = segment_iq_data(iq_data, segment_length=4096, overlap=0.5)
    features = []
    model.eval()
    with torch.no_grad():
        for segment in segments:
            spectrogram = transform(segment).unsqueeze(0).to(device)
            feat = model(spectrogram).cpu().numpy().flatten()
            features.append(feat)
    return np.array(features)

def spectrum_sensing_unsupervised(idle_data, traffic_data, transform, device):
    print("\nExtracting features for idle channel...")
    feature_model = MobileNetFeatureExtractor(pretrained=True).to(device)
    idle_features = extract_features(feature_model, idle_data, transform, device)
    print("\nExtracting features for traffic channel...")
    traffic_features = extract_features(feature_model, traffic_data, transform, device)
    all_features = np.vstack([idle_features, traffic_features])
    # Optionally reduce dimensionality for clustering
    pca = PCA(n_components=16)
    reduced_features = pca.fit_transform(all_features)
    # K-means clustering (2 clusters: idle/traffic)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(reduced_features)
    cluster_labels = kmeans.labels_
    n_idle = np.sum(cluster_labels[:len(idle_features)] == cluster_labels[0])
    n_traffic = np.sum(cluster_labels[len(idle_features):] == cluster_labels[-1])
    print(f"Idle segments assigned to cluster 0: {n_idle}/{len(idle_features)}")
    print(f"Traffic segments assigned to cluster 1: {n_traffic}/{len(traffic_features)}")
    # Visualize cluster assignments
    plt.figure(figsize=(10, 4))
    plt.plot(cluster_labels, 'o-', label='Cluster Assignment')
    plt.axvline(len(idle_features)-1, color='r', linestyle='--', label='Idle/Traffic Split')
    plt.title('Unsupervised Spectrum Sensing (Cluster Assignments)')
    plt.xlabel('Segment Index')
    plt.ylabel('Cluster')
    plt.legend()
    plt.tight_layout()
    plt.show()

def perform_spectrum_sensing_inference(model, iq_data, transform, device, title="Spectrum Sensing Results"):
    """Perform spectrum sensing inference on IQ data using pretrained model"""
    print(f"\nPerforming spectrum sensing inference on {title}...")
    segments = segment_iq_data(iq_data, segment_length=4096, overlap=0.5)
    print(f"Created {len(segments)} segments for analysis")
    model.eval()
    predictions = []
    confidences = []
    labels = []
    with torch.no_grad():
        for idx, segment in enumerate(segments):
            spectrogram = transform(segment).unsqueeze(0).to(device)
            output = model(spectrogram)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()
            predictions.append(prediction)
            confidences.append(confidence)
            label = "idle" if prediction == 0 else "traffic"
            labels.append(label)
            print(f"Segment {idx+1:03d}: {label} (confidence: {confidence:.3f})")
    # Output summary
    idle_count = predictions.count(0)
    traffic_count = predictions.count(1)
    print(f"\n{title}:")
    print(f"  Total segments analyzed: {len(predictions)}")
    print(f"  Segments classified as Idle: {idle_count} ({100*idle_count/len(predictions):.1f}%)")
    print(f"  Segments classified as Traffic: {traffic_count} ({100*traffic_count/len(predictions):.1f}%)")
    print(f"  Average confidence: {np.mean(confidences):.3f}")
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(['Idle', 'Traffic'], [idle_count, traffic_count], color=['blue', 'orange'])
    plt.title(f'{title} - Prediction Distribution')
    plt.ylabel('Number of Segments')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.hist(confidences, bins=20, alpha=0.7, color='green')
    plt.title(f'{title} - Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return predictions, confidences, labels

def train_spectrum_classifier(idle_files, traffic_files, transform, device, epochs=10, lr=0.001):
    """Train the MobileNet spectrum classifier on labeled data"""
    print("=== Training MobileNet Spectrum Classifier ===")
    
    # Load and prepare data
    idle_data = []
    traffic_data = []
    
    print("Loading idle data...")
    for file_path in idle_files:
        data = read_iq_file(file_path, max_samples=50000)
        if data is not None:
            idle_data.append(data)
    
    print("Loading traffic data...")
    for file_path in traffic_files:
        data = read_iq_file(file_path, max_samples=50000)
        if data is not None:
            traffic_data.append(data)
    
    if not idle_data or not traffic_data:
        print("Error: Need both idle and traffic data for training")
        return None
    
    # Create segments and labels
    segments = []
    labels = []
    
    for data in idle_data:
        data_segments = segment_iq_data(data, segment_length=4096, overlap=0.5)
        segments.extend(data_segments)
        labels.extend([0] * len(data_segments))  # 0 for idle
    
    for data in traffic_data:
        data_segments = segment_iq_data(data, segment_length=4096, overlap=0.5)
        segments.extend(data_segments)
        labels.extend([1] * len(data_segments))  # 1 for traffic
    
    print(f"Total segments: {len(segments)} (Idle: {labels.count(0)}, Traffic: {labels.count(1)})")
    
    # Create dataset and dataloader
    dataset = IQDataset(segments, labels, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = MobileNetSpectrumClassifier(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Unfreeze some layers for fine-tuning
    model.unfreeze_features()
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    print("Training completed!")
    return model

def perform_spectrum_sensing_with_pretrained_features(iq_data, transform, device, title="Spectrum Sensing Results"):
    """Perform spectrum sensing using pretrained MobileNet features with statistical analysis"""
    print(f"\nPerforming spectrum sensing with pretrained features on {title}...")
    segments = segment_iq_data(iq_data, segment_length=4096, overlap=0.5)
    print(f"Created {len(segments)} segments for analysis")
    
    # Use feature extractor to get meaningful features
    feature_model = MobileNetFeatureExtractor(pretrained=True).to(device)
    feature_model.eval()
    
    features = []
    spectrograms = []
    
    with torch.no_grad():
        for idx, segment in enumerate(segments):
            spectrogram = transform(segment).unsqueeze(0).to(device)
            feature = feature_model(spectrogram).cpu().numpy().flatten()
            features.append(feature)
            spectrograms.append(spectrogram.cpu().numpy().squeeze())
    
    features = np.array(features)
    spectrograms = np.array(spectrograms)
    
    # Statistical analysis of features
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_energy = np.sum(features**2, axis=1)
    
    # Simple threshold-based classification using feature statistics
    energy_threshold = np.median(feature_energy)
    predictions = (feature_energy > energy_threshold).astype(int)
    
    # Calculate confidence based on distance from threshold
    confidences = np.abs(feature_energy - energy_threshold) / (np.max(feature_energy) - np.min(feature_energy) + 1e-8)
    confidences = np.clip(confidences, 0, 1)
    
    labels = ["traffic" if p == 1 else "idle" for p in predictions]
    
    # Print results
    for idx, (label, conf) in enumerate(zip(labels, confidences)):
        print(f"Segment {idx+1:03d}: {label} (confidence: {conf:.3f})")
    
    # Output summary
    idle_count = predictions.tolist().count(0)
    traffic_count = predictions.tolist().count(1)
    print(f"\n{title} (Feature-based Analysis):")
    print(f"  Total segments analyzed: {len(predictions)}")
    print(f"  Segments classified as Idle: {idle_count} ({100*idle_count/len(predictions):.1f}%)")
    print(f"  Segments classified as Traffic: {traffic_count} ({100*traffic_count/len(predictions):.1f}%)")
    print(f"  Average confidence: {np.mean(confidences):.3f}")
    print(f"  Feature energy threshold: {energy_threshold:.3f}")
    
    # Enhanced visualization
    plt.figure(figsize=(15, 10))
    
    # Prediction distribution
    plt.subplot(2, 3, 1)
    plt.bar(['Idle', 'Traffic'], [idle_count, traffic_count], color=['blue', 'orange'])
    plt.title(f'{title} - Prediction Distribution')
    plt.ylabel('Number of Segments')
    plt.grid(True, alpha=0.3)
    
    # Confidence distribution
    plt.subplot(2, 3, 2)
    plt.hist(confidences, bins=20, alpha=0.7, color='green')
    plt.title(f'{title} - Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Feature energy over time
    plt.subplot(2, 3, 3)
    plt.plot(feature_energy, 'b-', alpha=0.7)
    plt.axhline(y=energy_threshold, color='r', linestyle='--', label=f'Threshold: {energy_threshold:.3f}')
    plt.title('Feature Energy Over Time')
    plt.xlabel('Segment Index')
    plt.ylabel('Feature Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature statistics
    plt.subplot(2, 3, 4)
    plt.plot(feature_mean[:50], 'g-', label='Mean')
    plt.plot(feature_std[:50], 'r-', label='Std')
    plt.title('Feature Statistics (First 50 dimensions)')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confidence vs Energy
    plt.subplot(2, 3, 5)
    plt.scatter(feature_energy, confidences, alpha=0.6, c=predictions, cmap='viridis')
    plt.xlabel('Feature Energy')
    plt.ylabel('Confidence')
    plt.title('Confidence vs Feature Energy')
    plt.colorbar(label='Prediction (0=Idle, 1=Traffic)')
    plt.grid(True, alpha=0.3)
    
    # Average spectrogram
    plt.subplot(2, 3, 6)
    avg_spectrogram = np.mean(spectrograms, axis=0)
    plt.imshow(avg_spectrogram, aspect='auto', cmap='viridis')
    plt.title('Average Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Magnitude')
    
    plt.tight_layout()
    plt.show()
    
    return predictions, confidences, labels, features

def main():
    print("=== MobileNet Spectrum Sensing with Pretrained Features ===")
    parser = argparse.ArgumentParser(description="Spectrum sensing on a specific IQ file using pretrained MobileNet features.")
    parser.add_argument('--file', type=str, default='lte_20Mhz_rate23.04Mhz_dur_10ms_pci252_idle.bin',
                        help='IQ file to analyze (default: lte_20Mhz_rate23.04Mhz_dur_10ms_pci252_idle.bin)')
    parser.add_argument('--method', type=str, default='features', choices=['features', 'classifier'],
                        help='Method to use: "features" for feature-based analysis, "classifier" for transfer learning classifier')
    args = parser.parse_args()
    
    iq_dir = Path("../iq")
    iq_file = iq_dir / args.file
    if not iq_file.exists():
        print(f"Error: {iq_file} not found.")
        return
    
    print(f"\nLoading IQ data from {iq_file} ...")
    iq_data = read_iq_file(iq_file, max_samples=100000)
    if iq_data is None:
        print("Error: Failed to load IQ data")
        return
    
    print(f"Loaded {len(iq_data)} samples from {iq_file}")
    print("\nAnalyzing spectrum...")
    analyze_spectrum(iq_data, title=f"Spectrum Analysis: {args.file}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = IQTransform(fft_size=1024, hop_size=256)
    
    if args.method == 'features':
        print("\nUsing pretrained MobileNet features for spectrum sensing...")
        predictions, confidences, labels, features = perform_spectrum_sensing_with_pretrained_features(
            iq_data, transform, device, title=f"Spectrum Sensing: {args.file}")
        
        print(f"\nFeature Analysis Summary:")
        print(f"  Feature dimensions: {features.shape[1]}")
        print(f"  Feature energy range: {np.min(np.sum(features**2, axis=1)):.3f} - {np.max(np.sum(features**2, axis=1)):.3f}")
        
    else:  # classifier method
        print("\nLoading pretrained MobileNet classifier...")
        model = MobileNetSpectrumClassifier(num_classes=2, pretrained=True)
        model = model.to(device)
        model.eval()
        print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        print("Note: This classifier uses random initialization for the final layers.")
        print("For meaningful results, you would need to train it on labeled spectrum data.")
        
        predictions, confidences, labels = perform_spectrum_sensing_inference(
            model, iq_data, transform, device, title=f"Spectrum Sensing: {args.file}")
    
    print("\n=== Spectrum Sensing Complete ===")
    print("\nExplanation:")
    print("- The 'features' method uses pretrained MobileNet to extract meaningful features")
    print("- It then applies statistical analysis to classify segments as idle/traffic")
    print("- This approach leverages the pretrained model's learned representations")
    print("- For best results, you can train the classifier on your labeled spectrum data")

if __name__ == "__main__":
    main() 