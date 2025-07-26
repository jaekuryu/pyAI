import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QLabel, QFileDialog,
                             QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, 
                             QGridLayout, QProgressBar, QTextEdit, QTableWidget,
                             QTableWidgetItem, QSplitter, QFrame, QMessageBox,
                             QSlider, QCheckBox, QLineEdit, QStatusBar, QToolBar,
                             QMenu, QMenuBar, QFormLayout)
from PyQt6.QtGui import QIcon, QFont, QPixmap, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
import pandas as pd
from pathlib import Path
import pickle
import json
from scipy import signal

# Import our existing spectrum sensing modules
from pyMnet import IQTransform, analyze_spectrum, read_iq_file, segment_iq_data, MobileNetFeatureExtractor

class AnalysisThread(QThread):
    """Thread for running spectrum analysis to prevent GUI freezing"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, iq_data, sample_rate, fft_size, hop_size, window_type, model_type, threshold_method, segment_size):
        super().__init__()
        self.iq_data = iq_data
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_type = window_type
        self.model_type = model_type
        self.threshold_method = threshold_method
        self.segment_size = segment_size
        
    def run(self):
        try:
            # Convert to complex data
            complex_data = self.iq_data[:, 0] + 1j * self.iq_data[:, 1]
            
            # Initialize IQ transform
            iq_transform = IQTransform(
                fft_size=self.fft_size,
                hop_size=self.hop_size
            )
            
            # Segment the IQ data (same as pyMnet.py)
            self.progress.emit(20)
            segments = segment_iq_data(self.iq_data, segment_length=self.segment_size, overlap=0.5)
            print(f"Created {len(segments)} segments for analysis")
            
            # Use feature extractor to get meaningful features (same as pyMnet.py)
            self.progress.emit(40)
            feature_model = MobileNetFeatureExtractor(pretrained=True)
            feature_model.eval()
            
            features = []
            spectrograms = []
            
            # Process each segment individually
            self.progress.emit(60)
            with torch.no_grad():
                for idx, segment in enumerate(segments):
                    # Convert segment to complex
                    complex_segment = segment[:, 0] + 1j * segment[:, 1]
                    
                    # Compute spectrogram for this segment
                    spectrogram = iq_transform.compute_spectrogram(complex_segment)
                    
                    # Prepare for MobileNet (same preprocessing as pyMnet.py)
                    spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
                    spectrogram_normalized = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
                    
                    # Keep as single channel (don't convert to RGB) since MobileNetFeatureExtractor expects 1 channel
                    # spectrogram_rgb = np.stack([spectrogram_normalized] * 3, axis=-1)  # REMOVED
                    
                    # Transform for MobileNet input (single channel)
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        # Remove normalization since we're using single channel
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # REMOVED
                    ])
                    
                    # Extract features (single channel input)
                    img = transform(spectrogram_normalized).unsqueeze(0)  # Shape: (1, 1, 224, 224)
                    feature = feature_model(img).cpu().numpy().flatten()
                    features.append(feature)
                    spectrograms.append(spectrogram)
                    
                    # Update progress
                    if idx % 5 == 0:  # Update every 5 segments
                        progress = 60 + int(30 * idx / len(segments))
                        self.progress.emit(progress)
            
            # Statistical analysis of features (same as pyMnet.py)
            self.progress.emit(90)
            features = np.array(features)
            spectrograms = np.array(spectrograms)
            
            # Calculate feature statistics
            feature_mean = np.mean(features, axis=0)
            feature_std = np.std(features, axis=0)
            feature_energy = np.sum(features**2, axis=1)
            
            # Parameterized threshold-based classification using feature statistics
            if self.threshold_method == 'Median':
                energy_threshold = np.median(feature_energy)
            elif self.threshold_method == 'Mean':
                energy_threshold = np.mean(feature_energy)
            elif self.threshold_method == 'Max-Mean':
                energy_threshold = np.max(feature_energy) - np.mean(feature_energy)  # For normalized data between 0 and 1
            else: # Default to median
                energy_threshold = np.median(feature_energy)
            
            predictions = (feature_energy > energy_threshold).astype(int)
            
            # Calculate confidence based on distance from threshold
            confidences = np.abs(feature_energy - energy_threshold) / (np.max(feature_energy) - np.min(feature_energy) + 1e-8)
            confidences = np.clip(confidences, 0, 1)
            
            labels = ["traffic" if p == 1 else "idle" for p in predictions]
            
            # Calculate summary statistics
            idle_count = predictions.tolist().count(0)
            traffic_count = predictions.tolist().count(1)
            avg_confidence = np.mean(confidences)
            
            # Overall classification based on majority
            overall_classification = "Traffic" if traffic_count > idle_count else "Idle"
            overall_confidence = max(traffic_count, idle_count) / len(predictions)
            
            self.progress.emit(100)
            
            results = {
                'spectrogram': spectrograms[0] if len(spectrograms) > 0 else None,  # Use first segment for display
                'spectrograms': spectrograms,  # All spectrograms for average calculation
                'features': features,
                'predictions': predictions,
                'confidences': confidences,
                'labels': labels,
                'classification': overall_classification,
                'confidence': overall_confidence,
                'feature_mean': feature_mean,
                'feature_std': feature_std,
                'feature_energy': feature_energy,
                'energy_threshold': energy_threshold,
                'idle_count': idle_count,
                'traffic_count': traffic_count,
                'total_segments': len(segments),
                'avg_confidence': avg_confidence,
                'threshold_method': self.threshold_method # Add threshold method to results
            }
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))

class PlotWidget(QWidget):
    """Widget for displaying matplotlib plots"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def clear(self):
        self.figure.clear()
        self.canvas.draw()
        
    def plot(self, *args, **kwargs):
        ax = self.figure.add_subplot(111)
        ax.plot(*args, **kwargs)
        self.canvas.draw()

class TimeDomainTab(QWidget):
    """Tab for time domain visualization"""
    def __init__(self):
        super().__init__()
        self.plot_widget = PlotWidget()
        self.iq_data = None  # Store IQ data for redrawing
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.sample_range_label = QLabel("Sample Range:")
        self.sample_range_spin = QSpinBox()
        self.sample_range_spin.setRange(100, 10000)
        self.sample_range_spin.setValue(1000)
        self.sample_range_spin.valueChanged.connect(self.on_sample_range_changed)
        
        controls_layout.addWidget(self.sample_range_label)
        controls_layout.addWidget(self.sample_range_spin)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        
    def update_plot(self, iq_data=None):
        if iq_data is None:
            return
            
        # Store IQ data for redrawing when sample range changes
        self.iq_data = iq_data
        
        self.plot_widget.clear()
        ax = self.plot_widget.figure.add_subplot(111)
        
        sample_range = self.sample_range_spin.value()
        samples = min(sample_range, len(iq_data))
        
        x = np.arange(samples)
        ax.plot(x, iq_data[:samples, 0], label='I', alpha=0.7)
        ax.plot(x, iq_data[:samples, 1], label='Q', alpha=0.7)
        
        ax.set_title('Time Domain Signal')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.plot_widget.canvas.draw()
        
    def on_sample_range_changed(self):
        """Handle sample range spin box value changes"""
        if self.iq_data is not None:
            self.update_plot(self.iq_data)

class FrequencyDomainTab(QWidget):
    """Tab for frequency domain visualization"""
    def __init__(self):
        super().__init__()
        self.plot_widget = PlotWidget()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.fft_size_label = QLabel("FFT Size:")
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(['256', '512', '1024', '2048', '4096'])
        self.fft_size_combo.setCurrentText('1024')
        self.fft_size_combo.currentTextChanged.connect(self.update_plot)
        
        self.db_scale_check = QCheckBox("dB Scale")
        self.db_scale_check.setChecked(True)
        self.db_scale_check.toggled.connect(self.update_plot)
        
        controls_layout.addWidget(self.fft_size_label)
        controls_layout.addWidget(self.fft_size_combo)
        controls_layout.addWidget(self.db_scale_check)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        
    def update_plot(self, iq_data=None, sample_rate=23.04e6):
        if iq_data is None:
            return
            
        self.plot_widget.clear()
        ax = self.plot_widget.figure.add_subplot(111)
        
        fft_size = int(self.fft_size_combo.currentText())
        fft_size = min(fft_size, len(iq_data))
        
        # Convert to complex
        complex_data = iq_data[:fft_size, 0] + 1j * iq_data[:fft_size, 1]
        
        # Compute FFT
        fft_result = np.fft.fft(complex_data)
        frequencies = np.fft.fftfreq(fft_size, 1/sample_rate)
        
        # Apply fftshift for proper frequency ordering
        shifted_fft_result = np.fft.fftshift(fft_result)
        shifted_frequencies = np.fft.fftshift(frequencies)
        
        if self.db_scale_check.isChecked():
            y_data = 20 * np.log10(np.abs(shifted_fft_result))
            ylabel = 'Power (dB)'
        else:
            y_data = np.abs(shifted_fft_result)
            ylabel = 'Magnitude'
        
        ax.plot(shifted_frequencies / 1e6, y_data)
        ax.set_title('Frequency Domain')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        self.plot_widget.canvas.draw()

class SpectrogramTab(QWidget):
    """Tab for spectrogram visualization"""
    def __init__(self):
        super().__init__()
        self.plot_widget = PlotWidget()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.nperseg_label = QLabel("FFT Size:")
        self.nperseg_combo = QComboBox()
        self.nperseg_combo.addItems(['256', '512', '1024', '2048'])
        self.nperseg_combo.setCurrentText('1024')
        self.nperseg_combo.currentTextChanged.connect(self.update_plot)
        
        self.noverlap_label = QLabel("Overlap:")
        self.noverlap_spin = QSpinBox()
        self.noverlap_spin.setRange(0, 1024)
        self.noverlap_spin.setValue(512)
        self.noverlap_spin.valueChanged.connect(self.update_plot)
        
        controls_layout.addWidget(self.nperseg_label)
        controls_layout.addWidget(self.nperseg_combo)
        controls_layout.addWidget(self.noverlap_label)
        controls_layout.addWidget(self.noverlap_spin)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        
    def update_plot(self, iq_data=None, sample_rate=23.04e6):
        if iq_data is None:
            return
            
        self.plot_widget.clear()
        ax = self.plot_widget.figure.add_subplot(111)
        
        # Convert to complex
        complex_data = iq_data[:, 0] + 1j * iq_data[:, 1]
        
        # Compute spectrogram
        nperseg = int(self.nperseg_combo.currentText())
        noverlap = self.noverlap_spin.value()
        
        from scipy import signal
        f, t, Sxx = signal.spectrogram(complex_data, sample_rate, nperseg=nperseg, 
                                      noverlap=noverlap, return_onesided=False)
        
        # Apply fftshift for proper frequency ordering
        shifted_f = np.fft.fftshift(f)
        shifted_Sxx = np.fft.fftshift(Sxx, axes=0)
        
        # Plot spectrogram
        im = ax.pcolormesh(t, shifted_f / 1e6, 10 * np.log10(shifted_Sxx))
        ax.set_title('Spectrogram')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (MHz)')
        
        # Add colorbar
        self.plot_widget.figure.colorbar(im, ax=ax, label='Power (dB)')
        
        self.plot_widget.canvas.draw()

class ConstellationTab(QWidget):
    """Tab for IQ constellation visualization"""
    def __init__(self):
        super().__init__()
        self.plot_widget = PlotWidget()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.sample_count_label = QLabel("Sample Count:")
        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setRange(100, 10000)
        self.sample_count_spin.setValue(1000)
        self.sample_count_spin.valueChanged.connect(self.update_plot)
        
        controls_layout.addWidget(self.sample_count_label)
        controls_layout.addWidget(self.sample_count_spin)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        
    def update_plot(self, iq_data=None):
        if iq_data is None:
            return
            
        self.plot_widget.clear()
        ax = self.plot_widget.figure.add_subplot(111)
        
        sample_count = self.sample_count_spin.value()
        samples = min(sample_count, len(iq_data))
        
        ax.scatter(iq_data[:samples, 0], iq_data[:samples, 1], alpha=0.6, s=1)
        ax.set_title('IQ Constellation')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        self.plot_widget.canvas.draw()

class ResultsTab(QWidget):
    """Tab for analysis results"""
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        # Export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        
        layout.addWidget(QLabel("Analysis Results:"))
        layout.addWidget(self.results_text)
        layout.addWidget(self.export_button)
        self.setLayout(layout)
        
    def update_results(self, results):
        if results is None:
            self.results_text.clear()
            return
            
        text = f"""
Analysis Results:
================

Overall Classification: {results.get('classification', 'N/A')}
Overall Confidence: {results.get('confidence', 0):.3f}

Segment Analysis:
----------------
Total Segments Analyzed: {results.get('total_segments', 0)}
Segments Classified as Idle: {results.get('idle_count', 0)}
Segments Classified as Traffic: {results.get('traffic_count', 0)}
Idle Percentage: {100 * results.get('idle_count', 0) / max(results.get('total_segments', 1), 1):.1f}%
Traffic Percentage: {100 * results.get('traffic_count', 0) / max(results.get('total_segments', 1), 1):.1f}%

Feature Statistics:
------------------
Average Confidence: {results.get('avg_confidence', 0):.3f}
Threshold Method: {results.get('threshold_method', 'N/A')}
Feature Energy Threshold: {results.get('energy_threshold', 0):.3f}
Feature Mean: {results.get('feature_mean', [0])[0] if len(results.get('feature_mean', [])) > 0 else 0:.6f}
Feature Std: {results.get('feature_std', [0])[0] if len(results.get('feature_std', [])) > 0 else 0:.6f}

Analysis completed successfully.
        """
        
        self.results_text.setText(text)
        
    def export_results(self):
        # TODO: Implement export functionality
        QMessageBox.information(self, "Export", "Export functionality to be implemented")

class SummaryTab(QWidget):
    """Tab for comprehensive analysis summary with multiple plots"""
    def __init__(self):
        super().__init__()
        self.plot_widget = PlotWidget()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.update_button = QPushButton("Update Summary")
        self.update_button.clicked.connect(self.update_summary)
        
        self.export_button = QPushButton("Export Summary")
        self.export_button.clicked.connect(self.export_summary)
        
        controls_layout.addWidget(self.update_button)
        controls_layout.addWidget(self.export_button)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        
    def update_summary(self, iq_data=None, analysis_results=None, sample_rate=23.04e6):
        if iq_data is None or analysis_results is None:
            return
            
        self.plot_widget.clear()
        fig = self.plot_widget.figure
        
        # Create 1x4 grid of subplots for the 4 most important plots
        gs = fig.add_gridspec(1, 4, hspace=0.4, wspace=0.3)
        
        # Create the 4 most important plots
        ax1 = fig.add_subplot(gs[0, 0])  # Bar chart - Traffic/Idle Segments
        ax2 = fig.add_subplot(gs[0, 1])  # Histogram - Confidence Distribution
        ax3 = fig.add_subplot(gs[0, 2])  # Line plot - Feature Energy Over Time
        ax4 = fig.add_subplot(gs[0, 3])  # Spectrogram - Average Spectrogram
        
        # Plot 1: Bar chart - Number of Segments (REAL DATA)
        categories = ['Idle', 'Traffic']
        idle_count = analysis_results.get('idle_count', 0)
        traffic_count = analysis_results.get('traffic_count', 0)
        counts = [idle_count, traffic_count]
        
        bars = ax1.bar(categories, counts, color=['blue', 'orange'])
        ax1.set_title('Number of Segments', fontsize=8)
        ax1.set_ylabel('Number of Segments', fontsize=8)
        ax1.set_ylim(0, max(counts) * 1.2 if counts else 25)
        ax1.tick_params(axis='both', labelsize=7)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontsize=7)
        
        # Plot 2: Histogram - Confidence Distribution (REAL DATA)
        confidences = analysis_results.get('confidences', [])
        if len(confidences) > 0:
            ax2.hist(confidences, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax2.set_title('Confidence Distribution', fontsize=8)
            ax2.set_xlabel('Confidence', fontsize=8)
            ax2.set_ylabel('Frequency', fontsize=8)
            ax2.set_xlim(0, 1)
            ax2.tick_params(axis='both', labelsize=7)
        else:
            ax2.text(0.5, 0.5, 'No confidence data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Confidence Distribution', fontsize=8)
        
        # Plot 3: Line plot - Feature Energy Over Time (REAL DATA)
        feature_energy = analysis_results.get('feature_energy', [])
        energy_threshold = analysis_results.get('energy_threshold', 0)
        threshold_method = analysis_results.get('threshold_method', 'N/A')
        
        if len(feature_energy) > 0:
            segment_indices = np.arange(len(feature_energy))
            ax3.plot(segment_indices, feature_energy, 'b-', linewidth=2, label='Feature Energy')
            ax3.axhline(y=energy_threshold, color='red', linestyle='--', label=f'Threshold ({threshold_method}): {energy_threshold:.3f}')
            ax3.set_title(f'Feature Energy Over Time (Threshold: {threshold_method})', fontsize=8)
            ax3.set_xlabel('Segment Index', fontsize=8)
            ax3.set_ylabel('Feature Energy', fontsize=8)
            ax3.legend(fontsize=7)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='both', labelsize=7)
        else:
            ax3.text(0.5, 0.5, 'No feature energy data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Feature Energy Over Time', fontsize=8)
        
        # Plot 4: Average Spectrogram (REAL DATA)
        spectrograms = analysis_results.get('spectrograms', [])
        
        if len(spectrograms) > 0:
            # Calculate average spectrogram from all segments
            avg_spectrogram = np.mean(spectrograms, axis=0)
            im4 = ax4.imshow(avg_spectrogram, aspect='auto', cmap='viridis', origin='lower')
            ax4.set_title('Average Spectrogram', fontsize=8)
            ax4.set_xlabel('Time Frame', fontsize=8)
            ax4.set_ylabel('Frequency Bin', fontsize=8)
            ax4.tick_params(axis='both', labelsize=7)
            
            # Add colorbar
            cbar4 = fig.colorbar(im4, ax=ax4, shrink=0.8)
            cbar4.set_label('Magnitude', fontsize=7)
            cbar4.ax.tick_params(labelsize=6)
        else:
            ax4.text(0.5, 0.5, 'No spectrogram data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Average Spectrogram', fontsize=8)

        self.plot_widget.canvas.draw()
        
    def export_summary(self):
        # TODO: Implement export functionality
        QMessageBox.information(self, "Export", "Export functionality to be implemented")

class MobileNetPreprocessingTab(QWidget):
    """Tab for showing MobileNet preprocessing steps"""
    def __init__(self):
        super().__init__()
        self.plot_widget = PlotWidget()
        self.segments = []  # Store segments for selection
        self.current_params = {}  # Store current parameters for redrawing
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Add segment selector
        self.segment_label = QLabel("Segment:")
        self.segment_spin = QSpinBox()
        self.segment_spin.setRange(0, 0)  # Will be updated when segments are created
        self.segment_spin.setValue(0)
        self.segment_spin.valueChanged.connect(self.on_segment_changed)
        
        self.segment_info_label = QLabel("(0 total)")
        self.segment_info_label.setStyleSheet("color: gray; font-size: 10px;")
        
        self.update_button = QPushButton("Update Preprocessing")
        self.update_button.clicked.connect(self.update_preprocessing)
        
        self.export_button = QPushButton("Export Preprocessing")
        self.export_button.clicked.connect(self.export_preprocessing)
        
        controls_layout.addWidget(self.segment_label)
        controls_layout.addWidget(self.segment_spin)
        controls_layout.addWidget(self.segment_info_label)
        controls_layout.addWidget(self.update_button)
        controls_layout.addWidget(self.export_button)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        
    def update_preprocessing(self, iq_data=None, analysis_results=None, sample_rate=23.04e6, fft_size=1024, hop_size=256, window_type='hanning', model_type='mobilenet_v2', segment_size=50176):
        # Store current parameters for redrawing when segment changes
        self.current_params = {
            'iq_data': iq_data,
            'analysis_results': analysis_results,
            'sample_rate': sample_rate,
            'fft_size': fft_size,
            'hop_size': hop_size,
            'window_type': window_type,
            'model_type': model_type,
            'segment_size': segment_size
        }
        
        if iq_data is None:
            return
            
        # Create segments if not already done or if IQ data changed
        if not hasattr(self, '_last_iq_data') or self._last_iq_data is not iq_data:
            self.segments = segment_iq_data(iq_data, segment_length=segment_size, overlap=0.5)
            self._last_iq_data = iq_data
            # Update segment spin box range
            self.segment_spin.setRange(0, max(0, len(self.segments) - 1))
            self.segment_spin.setValue(0)
            # Update segment info label
            self.segment_info_label.setText(f"({len(self.segments)} total)")
        
        # Get selected segment
        segment_idx = self.segment_spin.value()
        if segment_idx >= len(self.segments):
            return
            
        selected_segment = self.segments[segment_idx]
        
        self.plot_widget.clear()
        fig = self.plot_widget.figure
        
        # Create 1x3 grid for the three preprocessing steps
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        # Adjust figure margins to move spectrograms down
        fig.subplots_adjust(top=0.75, bottom=0.15)
        
        # Compute spectrogram from selected segment
        from pyMnet import IQTransform
        iq_transform = IQTransform(fft_size=fft_size, hop_size=hop_size)
        complex_segment = selected_segment[:, 0] + 1j * selected_segment[:, 1]
        spectrogram = iq_transform.compute_spectrogram(complex_segment)
            
        # Step 1: Original spectrogram (dB scale)
        ax1 = fig.add_subplot(gs[0, 0])
        spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
        im1 = ax1.imshow(spectrogram_db, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_title(f'Step 1: dB Scale Spectrogram\n(Segment {segment_idx}/{len(self.segments)-1})', fontsize=10)
        ax1.set_xlabel('Time Frame', fontsize=7)
        ax1.set_ylabel('Frequency Bin', fontsize=7)
        ax1.tick_params(axis='both', labelsize=6)
        cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Power (dB)', fontsize=7)
        cbar1.ax.tick_params(labelsize=6)
        
        # Step 2: Normalized spectrogram [0,1]
        ax2 = fig.add_subplot(gs[0, 1])
        spectrogram_normalized = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
        im2 = ax2.imshow(spectrogram_normalized, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_title('Step 2: Normalized [0,1]', fontsize=10)
        ax2.set_xlabel('Time Frame', fontsize=7)
        ax2.set_ylabel('Frequency Bin', fontsize=7)
        ax2.tick_params(axis='both', labelsize=6)
        cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Normalized Value', fontsize=7)
        cbar2.ax.tick_params(labelsize=6)
        
        # Step 3: 224x224 Single Channel (not RGB)
        ax3 = fig.add_subplot(gs[0, 2])
        # Resize to 224x224
        import torch.nn.functional as F
        import torch
        spectrogram_tensor = torch.from_numpy(spectrogram_normalized).float().unsqueeze(0).unsqueeze(0)
        spectrogram_224 = F.interpolate(spectrogram_tensor, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).numpy()
        
        im3 = ax3.imshow(spectrogram_224, aspect='equal', cmap='viridis', origin='lower')
        ax3.set_title('Step 3: 224x224 (Single Channel)', fontsize=10)
        ax3.set_xlabel('Pixel (224)', fontsize=7)
        ax3.set_ylabel('Pixel (224)', fontsize=7)
        ax3.tick_params(axis='both', labelsize=6)
        cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.8)
        cbar3.set_label('Normalized Value', fontsize=7)
        cbar3.ax.tick_params(labelsize=6)
        
        # Add text annotations in a better position
        segment_duration = len(selected_segment) / sample_rate * 1000  # Duration in ms
        fig.text(0.48, 0.98, f'Segment {segment_idx}: {spectrogram.shape[1]}x{spectrogram.shape[0]} ({segment_duration:.1f}ms)', 
                fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        fig.text(0.68, 0.98, f'Final Size: 224x224x1 (Single Channel)', 
                fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        self.plot_widget.canvas.draw()
        
    def on_segment_changed(self):
        """Handle segment selection change"""
        # Redraw with current parameters if we have them
        if self.current_params:
            self.update_preprocessing(**self.current_params)
        
    def export_preprocessing(self):
        # TODO: Implement export functionality
        QMessageBox.information(self, "Export", "Export functionality to be implemented")

class SpectrumSensingGUI(QMainWindow):
    """Main GUI application for spectrum sensing"""
    def __init__(self):
        super().__init__()
        self.iq_data = None
        self.analysis_results = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Spectrum Sensing GUI")
        self.setGeometry(100, 100, 1200, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create file info and controls panel
        self.create_control_panel(main_layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.create_tabs()
        main_layout.addWidget(self.tab_widget)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def create_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Load file action
        load_action = QAction("Load File", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_file)
        toolbar.addAction(load_action)
        
        # Analyze action
        analyze_action = QAction("Analyze", self)
        analyze_action.setShortcut("Ctrl+A")
        analyze_action.triggered.connect(self.start_analysis)
        toolbar.addAction(analyze_action)
        
        toolbar.addSeparator()
        
        # Export action
        export_action = QAction("Export", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_results)
        toolbar.addAction(export_action)
        
    def create_control_panel(self, main_layout):
        # Create splitter for control panel and main area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # File info group
        file_group = QGroupBox("File Information")
        file_layout = QGridLayout()
        
        self.file_name_label = QLabel("File: None")
        self.file_size_label = QLabel("Size: N/A")
        self.file_duration_label = QLabel("Duration: N/A")
        
        file_layout.addWidget(self.file_name_label, 0, 0, 1, 2)
        file_layout.addWidget(self.file_size_label, 1, 0)
        file_layout.addWidget(self.file_duration_label, 1, 1)
        
        file_group.setLayout(file_layout)
        
        # Parameters group
        param_group = QGroupBox("Parameters")
        param_layout = QGridLayout()
        
        self.sample_rate_label = QLabel("Sample Rate (MHz):")
        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(1, 1000)
        self.sample_rate_spin.setValue(23.04)
        self.sample_rate_spin.setDecimals(2)
        
        self.fft_size_label = QLabel("FFT Size:")
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(['256', '512', '1024', '2048', '4096'])
        self.fft_size_combo.setCurrentText('1024')
        self.fft_size_combo.currentTextChanged.connect(self.on_parameter_changed)
        
        self.hop_size_label = QLabel("Hop Size:")
        self.hop_size_spin = QSpinBox()
        self.hop_size_spin.setRange(64, 2048)
        self.hop_size_spin.setValue(1024)
        self.hop_size_spin.valueChanged.connect(self.on_parameter_changed)
        
        self.window_label = QLabel("Window Function:")
        self.window_combo = QComboBox()
        self.window_combo.addItems(['hanning', 'hamming', 'blackman'])
        self.window_combo.setCurrentText('hanning')
        
        self.model_label = QLabel("Model Type:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'])
        self.model_combo.setCurrentText('mobilenet_v2')

        self.threshold_method_label = QLabel("Threshold Method:")
        self.threshold_method_combo = QComboBox()
        self.threshold_method_combo.addItems(['Median', 'Mean', 'Max-Mean'])
        self.threshold_method_combo.setCurrentText('Median')
        self.threshold_method_combo.currentTextChanged.connect(self.on_parameter_changed)
        
        self.segment_size_label = QLabel("Segment Size:")
        self.segment_size_combo = QComboBox()
        self.segment_size_combo.addItems(['16384', '32768', '50176', '65536', '131072'])
        self.segment_size_combo.setCurrentText('50176')
        self.segment_size_combo.currentTextChanged.connect(self.on_parameter_changed)
        
        param_layout.addWidget(self.sample_rate_label, 0, 0)
        param_layout.addWidget(self.sample_rate_spin, 0, 1)
        param_layout.addWidget(self.fft_size_label, 1, 0)
        param_layout.addWidget(self.fft_size_combo, 1, 1)
        param_layout.addWidget(self.hop_size_label, 2, 0)
        param_layout.addWidget(self.hop_size_spin, 2, 1)
        param_layout.addWidget(self.window_label, 3, 0)
        param_layout.addWidget(self.window_combo, 3, 1)
        param_layout.addWidget(self.model_label, 4, 0)
        param_layout.addWidget(self.model_combo, 4, 1)
        param_layout.addWidget(self.threshold_method_label, 5, 0)
        param_layout.addWidget(self.threshold_method_combo, 5, 1)
        param_layout.addWidget(self.segment_size_label, 6, 0)
        param_layout.addWidget(self.segment_size_combo, 6, 1)
        
        param_group.setLayout(param_layout)
        
        # Settings buttons
        settings_layout = QHBoxLayout()
        self.save_settings_button = QPushButton("Save Settings")
        self.load_settings_button = QPushButton("Load Settings")
        self.reset_settings_button = QPushButton("Reset")
        
        self.save_settings_button.clicked.connect(self.save_settings)
        self.load_settings_button.clicked.connect(self.load_settings)
        self.reset_settings_button.clicked.connect(self.reset_settings)
        
        settings_layout.addWidget(self.save_settings_button)
        settings_layout.addWidget(self.load_settings_button)
        settings_layout.addWidget(self.reset_settings_button)
        
        # Add groups to control panel
        control_layout.addWidget(file_group)
        control_layout.addWidget(param_group)
        control_layout.addLayout(settings_layout) # Added settings_layout
        control_layout.addStretch()
        
        # Add control panel to splitter
        splitter.addWidget(control_panel)
        splitter.setSizes([300, 900])  # Control panel width
        
    def on_parameter_changed(self):
        """Handle parameter changes in the control panel"""
        if self.iq_data is not None:
            # Update MobileNet preprocessing tab with new parameters
            sample_rate = self.sample_rate_spin.value() * 1e6
            fft_size = int(self.fft_size_combo.currentText())
            hop_size = self.hop_size_spin.value()
            window_type = self.window_combo.currentText() # Get window type
            model_type = self.model_combo.currentText() # Get model type
            threshold_method = self.threshold_method_combo.currentText() # Get threshold method
            segment_size = int(self.segment_size_combo.currentText()) # Get segment size
            self.mobilenet_tab.update_preprocessing(self.iq_data, None, sample_rate, fft_size, hop_size, window_type, model_type, segment_size)
    
    def get_settings(self):
        """Get current parameter settings"""
        return {
            'sample_rate': self.sample_rate_spin.value() * 1e6,
            'fft_size': int(self.fft_size_combo.currentText()),
            'hop_size': self.hop_size_spin.value(),
            'window_type': self.window_combo.currentText(),
            'model_type': self.model_combo.currentText(),
            'threshold_method': self.threshold_method_combo.currentText(),
            'segment_size': int(self.segment_size_combo.currentText())
        }
        
    def set_settings(self, settings):
        """Set parameter settings"""
        if 'sample_rate' in settings:
            self.sample_rate_spin.setValue(settings['sample_rate'] / 1e6)
        if 'fft_size' in settings:
            self.fft_size_combo.setCurrentText(str(settings['fft_size']))
        if 'hop_size' in settings:
            self.hop_size_spin.setValue(settings['hop_size'])
        if 'window_type' in settings:
            self.window_combo.setCurrentText(settings['window_type'])
        if 'model_type' in settings:
            self.model_combo.setCurrentText(settings['model_type'])
        if 'threshold_method' in settings:
            self.threshold_method_combo.setCurrentText(settings['threshold_method'])
        if 'segment_size' in settings:
            self.segment_size_combo.setCurrentText(str(settings['segment_size']))
            
    def save_settings(self):
        """Save current settings to file"""
        settings = self.get_settings()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Settings", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=2)
            QMessageBox.information(self, "Settings", "Settings saved successfully!")
            
    def load_settings(self):
        """Load settings from file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Settings", "", "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)
                self.set_settings(settings)
                QMessageBox.information(self, "Settings", "Settings loaded successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load settings: {str(e)}")
                
    def reset_settings(self):
        """Reset settings to defaults"""
        self.sample_rate_spin.setValue(23.04)
        self.fft_size_combo.setCurrentText('1024')
        self.hop_size_spin.setValue(1024)
        self.window_combo.setCurrentText('hanning')
        self.model_combo.setCurrentText('mobilenet_v2')
        self.threshold_method_combo.setCurrentText('Median')
        self.segment_size_combo.setCurrentText('50176')

    def create_tabs(self):
        # Create tab instances
        self.time_tab = TimeDomainTab()
        self.freq_tab = FrequencyDomainTab()
        self.spec_tab = SpectrogramTab()
        self.const_tab = ConstellationTab()
        self.results_tab = ResultsTab()
        self.summary_tab = SummaryTab() # Added SummaryTab
        self.mobilenet_tab = MobileNetPreprocessingTab() # Added MobileNetPreprocessingTab
        
        # Add tabs to widget
        self.tab_widget.addTab(self.time_tab, "Time Domain")
        self.tab_widget.addTab(self.freq_tab, "Frequency Domain")
        self.tab_widget.addTab(self.spec_tab, "Spectrogram")
        self.tab_widget.addTab(self.const_tab, "Constellation")
        self.tab_widget.addTab(self.mobilenet_tab, "MobileNet Preprocessing") # Added MobileNetPreprocessingTab
        self.tab_widget.addTab(self.results_tab, "Results")
        self.tab_widget.addTab(self.summary_tab, "Summary") # Added SummaryTab

    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load IQ File", "", "Binary Files (*.bin);;All Files (*)"
        )
        
        if filename:
            try:
                self.status_bar.showMessage("Loading file...")
                self.iq_data = read_iq_file(filename)
                
                # Update file info
                file_size = os.path.getsize(filename)
                duration = len(self.iq_data) / (self.sample_rate_spin.value() * 1e6)
                
                self.file_name_label.setText(f"File: {os.path.basename(filename)}")
                self.file_size_label.setText(f"Size: {file_size / 1024 / 1024:.2f} MB")
                self.file_duration_label.setText(f"Duration: {duration * 1000:.2f} ms")
                
                # Update plots
                self.update_all_plots()
                
                self.status_bar.showMessage("File loaded successfully")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
                self.status_bar.showMessage("Error loading file")
                
    def update_all_plots(self):
        if self.iq_data is None:
            return
            
        sample_rate = self.sample_rate_spin.value() * 1e6
        
        self.time_tab.update_plot(self.iq_data)
        self.freq_tab.update_plot(self.iq_data, sample_rate)
        self.spec_tab.update_plot(self.iq_data, sample_rate)
        self.const_tab.update_plot(self.iq_data)
        
        # Update preprocessing tab with full IQ data
        # Update MobileNet preprocessing tab with current GUI parameters
        fft_size = int(self.fft_size_combo.currentText())
        hop_size = self.hop_size_spin.value()
        window_type = self.window_combo.currentText() # Get window type
        model_type = self.model_combo.currentText() # Get model type
        threshold_method = self.threshold_method_combo.currentText() # Get threshold method
        segment_size = int(self.segment_size_combo.currentText()) # Get segment size
        self.mobilenet_tab.update_preprocessing(self.iq_data, None, sample_rate, fft_size, hop_size, window_type, model_type, segment_size)
        
    def start_analysis(self):
        if self.iq_data is None:
            QMessageBox.warning(self, "Warning", "Please load a file first")
            return
            
        # Get parameters
        sample_rate = self.sample_rate_spin.value() * 1e6
        fft_size = int(self.fft_size_combo.currentText())
        hop_size = self.hop_size_spin.value()
        window_type = self.window_combo.currentText() # Get window type
        model_type = self.model_combo.currentText() # Get model type
        threshold_method = self.threshold_method_combo.currentText() # Get threshold method
        segment_size = int(self.segment_size_combo.currentText()) # Get segment size
        
        # Create and start analysis thread
        self.analysis_thread = AnalysisThread(
            self.iq_data, sample_rate, fft_size, hop_size, window_type, model_type, threshold_method, segment_size
        )
        
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.analysis_finished)
        self.analysis_thread.error.connect(self.analysis_error)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Update status
        self.status_bar.showMessage("Analysis in progress...")
        
        # Start analysis
        self.analysis_thread.start()
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def analysis_finished(self, results):
        self.analysis_results = results
        
        # Get results information for status bar
        classification = results.get('classification', 'N/A')
        confidence = results.get('confidence', 0)
        idle_count = results.get('idle_count', 0)
        traffic_count = results.get('traffic_count', 0)
        total_segments = results.get('total_segments', 0)
        
        # Update results tab
        self.results_tab.update_results(results)
        
        # Update summary tab
        self.summary_tab.update_summary(self.iq_data, results, self.sample_rate_spin.value() * 1e6)
        
        # Update MobileNet preprocessing tab with current GUI parameters
        fft_size = int(self.fft_size_combo.currentText())
        hop_size = self.hop_size_spin.value()
        window_type = self.window_combo.currentText() # Get window type
        model_type = self.model_combo.currentText() # Get model type
        threshold_method = self.threshold_method_combo.currentText() # Get threshold method
        segment_size = int(self.segment_size_combo.currentText()) # Get segment size
        self.mobilenet_tab.update_preprocessing(self.iq_data, results, self.sample_rate_spin.value() * 1e6, fft_size, hop_size, window_type, model_type, segment_size)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Update status with detailed information
        status_msg = f"Analysis complete: {classification} ({confidence:.1%} confidence, {total_segments} segments analyzed)"
        self.status_bar.showMessage(status_msg)
        
    def analysis_error(self, error_msg):
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Update status
        self.status_bar.showMessage("Analysis failed")
        
        # Show error message
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed: {error_msg}")
        
    def export_results(self):
        if self.analysis_results is None:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
            
        # TODO: Implement export functionality
        QMessageBox.information(self, "Export", "Export functionality to be implemented")

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Spectrum Sensing GUI")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = SpectrumSensingGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 