import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import torch
import torchvision.transforms as transforms
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QLabel, QFileDialog,
                             QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, 
                             QGridLayout, QProgressBar, QTextEdit, QTableWidget,
                             QTableWidgetItem, QSplitter, QFrame, QMessageBox,
                             QSlider, QCheckBox, QLineEdit, QStatusBar, QToolBar,
                             QMenu, QMenuBar)
from PyQt6.QtGui import QIcon, QFont, QPixmap, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QFont, QPixmap
import pandas as pd
from pathlib import Path
import pickle
import json

# Import our existing spectrum sensing modules
from pyMnet import IQTransform, analyze_spectrum, read_iq_file

class AnalysisThread(QThread):
    """Thread for running spectrum analysis to prevent GUI freezing"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, iq_data, sample_rate, fft_size, hop_size, window_type, model_type):
        super().__init__()
        self.iq_data = iq_data
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_type = window_type
        self.model_type = model_type
        
    def run(self):
        try:
            # Convert to complex data
            complex_data = self.iq_data[:, 0] + 1j * self.iq_data[:, 1]
            
            # Initialize IQ transform
            iq_transform = IQTransform(
                fft_size=self.fft_size,
                hop_size=self.hop_size
            )
            
            # Compute spectrogram
            self.progress.emit(30)
            spectrogram = iq_transform.compute_spectrogram(complex_data)
            
            # Prepare for MobileNet
            self.progress.emit(60)
            # Normalize and resize spectrogram for MobileNet input
            spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
            spectrogram_normalized = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
            
            # Convert to RGB image format
            spectrogram_rgb = np.stack([spectrogram_normalized] * 3, axis=-1)
            
            # Load pretrained MobileNet
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
            model.eval()
            
            # Remove classification layer for feature extraction
            model.classifier = torch.nn.Identity()
            
            # Transform for MobileNet input
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Process spectrogram
            self.progress.emit(80)
            img = transform(spectrogram_rgb).unsqueeze(0)
            
            with torch.no_grad():
                features = model(img)
            
            # Simple classification based on feature statistics
            feature_mean = features.mean().item()
            feature_std = features.std().item()
            
            # Heuristic classification (can be improved with trained classifier)
            if feature_std > 0.1:  # High variance suggests traffic
                classification = "Traffic"
                confidence = min(0.9, feature_std * 2)
            else:
                classification = "Idle"
                confidence = min(0.9, (1 - feature_std) * 2)
            
            self.progress.emit(100)
            
            results = {
                'spectrogram': spectrogram,
                'features': features.numpy(),
                'classification': classification,
                'confidence': confidence,
                'feature_mean': feature_mean,
                'feature_std': feature_std
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
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.sample_range_label = QLabel("Sample Range:")
        self.sample_range_spin = QSpinBox()
        self.sample_range_spin.setRange(100, 10000)
        self.sample_range_spin.setValue(1000)
        self.sample_range_spin.valueChanged.connect(self.update_plot)
        
        controls_layout.addWidget(self.sample_range_label)
        controls_layout.addWidget(self.sample_range_spin)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        
    def update_plot(self, iq_data=None):
        if iq_data is None:
            return
            
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
        self.fft_size_combo.setCurrentText('4096')
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

Classification: {results.get('classification', 'N/A')}
Confidence: {results.get('confidence', 0):.3f}
Feature Mean: {results.get('feature_mean', 0):.6f}
Feature Std: {results.get('feature_std', 0):.6f}

Analysis completed successfully.
        """
        
        self.results_text.setText(text)
        
    def export_results(self):
        # TODO: Implement export functionality
        QMessageBox.information(self, "Export", "Export functionality to be implemented")

class SettingsTab(QWidget):
    """Tab for application settings"""
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Analysis parameters
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QGridLayout()
        
        self.sample_rate_label = QLabel("Sample Rate (MHz):")
        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(1, 1000)
        self.sample_rate_spin.setValue(23.04)
        self.sample_rate_spin.setDecimals(2)
        
        self.fft_size_label = QLabel("FFT Size:")
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(['256', '512', '1024', '2048', '4096'])
        self.fft_size_combo.setCurrentText('4096')
        
        self.hop_size_label = QLabel("Hop Size:")
        self.hop_size_spin = QSpinBox()
        self.hop_size_spin.setRange(64, 2048)
        self.hop_size_spin.setValue(1024)
        
        self.window_label = QLabel("Window Function:")
        self.window_combo = QComboBox()
        self.window_combo.addItems(['hanning', 'hamming', 'blackman'])
        self.window_combo.setCurrentText('hanning')
        
        analysis_layout.addWidget(self.sample_rate_label, 0, 0)
        analysis_layout.addWidget(self.sample_rate_spin, 0, 1)
        analysis_layout.addWidget(self.fft_size_label, 1, 0)
        analysis_layout.addWidget(self.fft_size_combo, 1, 1)
        analysis_layout.addWidget(self.hop_size_label, 2, 0)
        analysis_layout.addWidget(self.hop_size_spin, 2, 1)
        analysis_layout.addWidget(self.window_label, 3, 0)
        analysis_layout.addWidget(self.window_combo, 3, 1)
        
        analysis_group.setLayout(analysis_layout)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QGridLayout()
        
        self.model_label = QLabel("Model Type:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'])
        self.model_combo.setCurrentText('mobilenet_v2')
        
        model_layout.addWidget(self.model_label, 0, 0)
        model_layout.addWidget(self.model_combo, 0, 1)
        
        model_group.setLayout(model_layout)
        
        # Save/Load buttons
        button_layout = QHBoxLayout()
        self.save_settings_button = QPushButton("Save Settings")
        self.load_settings_button = QPushButton("Load Settings")
        self.reset_settings_button = QPushButton("Reset to Defaults")
        
        self.save_settings_button.clicked.connect(self.save_settings)
        self.load_settings_button.clicked.connect(self.load_settings)
        self.reset_settings_button.clicked.connect(self.reset_settings)
        
        button_layout.addWidget(self.save_settings_button)
        button_layout.addWidget(self.load_settings_button)
        button_layout.addWidget(self.reset_settings_button)
        
        layout.addWidget(analysis_group)
        layout.addWidget(model_group)
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def get_settings(self):
        return {
            'sample_rate': self.sample_rate_spin.value() * 1e6,
            'fft_size': int(self.fft_size_combo.currentText()),
            'hop_size': self.hop_size_spin.value(),
            'window_type': self.window_combo.currentText(),
            'model_type': self.model_combo.currentText()
        }
        
    def set_settings(self, settings):
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
            
    def save_settings(self):
        settings = self.get_settings()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Settings", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=2)
            QMessageBox.information(self, "Settings", "Settings saved successfully!")
            
    def load_settings(self):
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
        self.sample_rate_spin.setValue(23.04)
        self.fft_size_combo.setCurrentText('4096')
        self.hop_size_spin.setValue(1024)
        self.window_combo.setCurrentText('hanning')
        self.model_combo.setCurrentText('mobilenet_v2')

class SpectrumSensingGUI(QMainWindow):
    """Main GUI application for spectrum sensing"""
    def __init__(self):
        super().__init__()
        self.iq_data = None
        self.analysis_results = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Spectrum Sensing GUI")
        self.setGeometry(100, 100, 1200, 800)
        
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
        self.fft_size_combo.setCurrentText('4096')
        
        self.hop_size_label = QLabel("Hop Size:")
        self.hop_size_spin = QSpinBox()
        self.hop_size_spin.setRange(64, 2048)
        self.hop_size_spin.setValue(1024)
        
        param_layout.addWidget(self.sample_rate_label, 0, 0)
        param_layout.addWidget(self.sample_rate_spin, 0, 1)
        param_layout.addWidget(self.fft_size_label, 1, 0)
        param_layout.addWidget(self.fft_size_combo, 1, 1)
        param_layout.addWidget(self.hop_size_label, 2, 0)
        param_layout.addWidget(self.hop_size_spin, 2, 1)
        
        param_group.setLayout(param_layout)
        
        # Results group
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.status_label = QLabel("Status: Ready")
        self.classification_label = QLabel("Classification: N/A")
        self.confidence_label = QLabel("Confidence: N/A")
        
        results_layout.addWidget(self.status_label)
        results_layout.addWidget(self.classification_label)
        results_layout.addWidget(self.confidence_label)
        
        results_group.setLayout(results_layout)
        
        # Add groups to control panel
        control_layout.addWidget(file_group)
        control_layout.addWidget(param_group)
        control_layout.addWidget(results_group)
        control_layout.addStretch()
        
        # Add control panel to splitter
        splitter.addWidget(control_panel)
        splitter.setSizes([300, 900])  # Control panel width
        
    def create_tabs(self):
        # Create tab instances
        self.time_tab = TimeDomainTab()
        self.freq_tab = FrequencyDomainTab()
        self.spec_tab = SpectrogramTab()
        self.const_tab = ConstellationTab()
        self.results_tab = ResultsTab()
        self.settings_tab = SettingsTab()
        
        # Add tabs to widget
        self.tab_widget.addTab(self.time_tab, "Time Domain")
        self.tab_widget.addTab(self.freq_tab, "Frequency Domain")
        self.tab_widget.addTab(self.spec_tab, "Spectrogram")
        self.tab_widget.addTab(self.const_tab, "Constellation")
        self.tab_widget.addTab(self.results_tab, "Results")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        
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
        
    def start_analysis(self):
        if self.iq_data is None:
            QMessageBox.warning(self, "Warning", "Please load a file first")
            return
            
        # Get parameters
        sample_rate = self.sample_rate_spin.value() * 1e6
        fft_size = int(self.fft_size_combo.currentText())
        hop_size = self.hop_size_spin.value()
        window_type = 'hanning'  # Default
        model_type = 'mobilenet_v2'  # Default
        
        # Create and start analysis thread
        self.analysis_thread = AnalysisThread(
            self.iq_data, sample_rate, fft_size, hop_size, window_type, model_type
        )
        
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.analysis_finished)
        self.analysis_thread.error.connect(self.analysis_error)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Update status
        self.status_label.setText("Status: Analyzing...")
        self.status_bar.showMessage("Analysis in progress...")
        
        # Start analysis
        self.analysis_thread.start()
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def analysis_finished(self, results):
        self.analysis_results = results
        
        # Update results display
        self.classification_label.setText(f"Classification: {results['classification']}")
        self.confidence_label.setText(f"Confidence: {results['confidence']:.3f}")
        
        # Update results tab
        self.results_tab.update_results(results)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Update status
        self.status_label.setText("Status: Complete")
        self.status_bar.showMessage("Analysis completed successfully")
        
    def analysis_error(self, error_msg):
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Update status
        self.status_label.setText("Status: Error")
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