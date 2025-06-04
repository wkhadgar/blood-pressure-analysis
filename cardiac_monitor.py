import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from scipy.signal import butter, lfilter, find_peaks, windows
import matplotlib.pyplot as plt
import csv
import sys
from collections import deque

class BloodPressureMonitor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Setup UI
        self.setWindowTitle("Blood Pressure Monitor")
        self.resize(1200, 800) # Increased size for spectrogram
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.status_label = QtWidgets.QLabel("Preparing to start...")
        self.status_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.layout.addWidget(self.status_label)

        self.bpm_label = QtWidgets.QLabel("BPM: --")
        self.bpm_label.setStyleSheet("font-size: 32px; color: red;")
        self.layout.addWidget(self.bpm_label)

        self.debug_label = QtWidgets.QLabel("Debug: Waiting for data...")
        self.debug_label.setStyleSheet("font-size: 12px; color: blue;")
        self.layout.addWidget(self.debug_label)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.plot_widget)

        self.raw_plot = self.plot_widget.addPlot(title="Cuff Pressure with Pulse Signal (mmHg)")
        self.raw_curve = self.raw_plot.plot(pen='y')
        self.raw_plot.setLabel('bottom', "Time", units='s')
        self.raw_plot.setLabel('left', "Pressure", units='mmHg')

        # Spectrogram Plot Setup
        self.spectrogram_plot = self.plot_widget.addPlot(row=0, col=1)
        self.spectrogram_plot.setAspectLocked(False) # Allow aspect ratio to change
        self.spectrogram_plot.setTitle("Live Spectrogram of Pulse Signal")
        self.spectrogram_plot.setLabel('left', "Frequency", units='Hz')
        self.spectrogram_plot.setLabel('bottom', "Time", units='s')

        self.spectrogram_img = pg.ImageItem()
        self.spectrogram_plot.addItem(self.spectrogram_img)


        # Peak scatter item, will be added/removed from raw_plot as needed
        self.peak_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))


        # System parameters
        self.fs = 100  # Sampling rate (Hz)

        self.filtered_buffer_for_live_peaks = np.zeros(int(self.fs * 5)) # A rolling buffer for live peak detection
        # This is separate from accumulated raw_data
        self.live_peaks = []  # For live peak display
        self.all_bpms_live = deque(maxlen=20)  # Store last 20 live BPM readings for smoothing display

        self.start_time = QtCore.QDateTime.currentMSecsSinceEpoch()
        self.time_data = [] # Stores all time points
        self.raw_data = []  # Stores all raw data points
        self.analysis_filtered_data = [] # Stores filtered data after complete analysis
        self.analysis_peaks = []
        self.bpm_data_final = []  # Stores (time_index, bpm_value) from final analysis

        self.STATES = {
            "IDLE": 0,
            "INFLATING": 1,
            "DEFLATING": 2,
            "COMPLETE": 3
        }
        self.state = self.STATES["IDLE"]

        self.cuff_pressure = 0
        self.max_pressure = 180  # mmHg
        self.systolic_pressure = 120  # mmHg
        self.diastolic_pressure = 50  # mmHg
        self.min_pressure = 0  # mmHg
        self.inflation_rate = 30  # mmHg/s
        self.deflation_rate = 2.5  # mmHg/s

        self.heart_rate_for_simulation = 100  # BPM
        self.pulse_amp_scale = 1.2  # Amplitude scale for the synthetic pulse
        self.pulse_sig_retf = None
        self.pulse_signal_idx = 0

        self.b_low, self.a_low = self.butter_lowpass(3, self.fs, order=4)

        # Spectrogram parameters
        self.nperseg = int(self.fs * 1.0)  # Window length for spectrogram (1 second)
        self.noverlap = int(self.nperseg * 0.75) # Overlap
        self.spectrogram_data = [] # To store ALL spectrogram slices (columns)
        self.spectrogram_time_offset = 0 # To track start time for spectrogram columns

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000 / self.fs))

        QtCore.QTimer.singleShot(1000, self.start_inflation)

    def butter_lowpass(self, cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def start_inflation(self):
        self.state = self.STATES["INFLATING"]
        self.status_label.setText("Inflating cuff...")
        self.inflation_start_time = QtCore.QDateTime.currentMSecsSinceEpoch()

    def start_deflation(self):
        self.state = self.STATES["DEFLATING"]
        self.status_label.setText("Deflating cuff and collecting data...")
        self.deflation_start_time = QtCore.QDateTime.currentMSecsSinceEpoch()
        self.spectrogram_data = [] # Clear previous spectrogram data if any
        self.spectrogram_time_offset = len(self.raw_data) / self.fs # Spectrogram starts from current time in raw_data

        # Prepare pulse signal for deflation phase
        fhr = self.heart_rate_for_simulation / 60.0
        fwhr = 2 * np.pi * fhr

        # Calculate duration where pulse is active (between systolic and diastolic)
        pulse_active_duration = (self.systolic_pressure + 10 - (self.diastolic_pressure - 10)) / self.deflation_rate
        if pulse_active_duration <= 0:
            pulse_active_duration = 15 # Default duration if calculation is problematic

        num_pulse_samples = int(self.fs * pulse_active_duration)
        if num_pulse_samples <= 0:
            self.pulse_sig_retf = None
            print("Warning: Pulse generation time is too short or zero. No pulse will be added.")
            return

        pulse_t_local = np.linspace(0, pulse_active_duration, num_pulse_samples, endpoint=False)
        std_gaussian_factor = 4.0
        std_gaussian = num_pulse_samples / std_gaussian_factor
        if std_gaussian < 1: std_gaussian = 1

        g_filt = windows.gaussian(num_pulse_samples, std_gaussian)
        pulse = np.sin(fwhr * pulse_t_local)
        pulse_sig_modulated = pulse * g_filt
        self.pulse_sig_retf = np.array([p if p > 0 else 0 for p in pulse_sig_modulated])
        self.pulse_signal_idx = 0

        self.debug_label.setText(f"Debug: Pulse signal generated ({len(self.pulse_sig_retf)} samples).")

    def update(self):
        current_time_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
        current_elapsed_s = (current_time_ms - self.start_time) / 1000.0

        noise_amplitude = 0.15
        current_cuff_pressure_sample = self.cuff_pressure

        if self.state == self.STATES["INFLATING"]:
            elapsed_inflation_s = (current_time_ms - self.inflation_start_time) / 1000.0
            self.cuff_pressure = min(self.max_pressure, self.inflation_rate * elapsed_inflation_s)
            current_cuff_pressure_sample = self.cuff_pressure + noise_amplitude * np.random.randn()

            if self.cuff_pressure >= self.max_pressure:
                self.start_deflation()

        elif self.state == self.STATES["DEFLATING"]:
            elapsed_deflation_s = (current_time_ms - self.deflation_start_time) / 1000.0
            self.cuff_pressure = max(self.min_pressure, self.max_pressure - (elapsed_deflation_s * self.deflation_rate))
            current_cuff_pressure_sample = self.cuff_pressure + noise_amplitude * np.random.randn()

            # Add pulse signal only when cuff pressure is within the relevant range
            if self.pulse_sig_retf is not None and \
                    (self.diastolic_pressure - 10) < self.cuff_pressure < (self.systolic_pressure + 10):
                if self.pulse_signal_idx < len(self.pulse_sig_retf):
                    current_cuff_pressure_sample += self.pulse_amp_scale * self.pulse_sig_retf[self.pulse_signal_idx]
                    self.pulse_signal_idx += 1

            self.debug_label.setText(
                f"Debug: P: {self.cuff_pressure:.1f} mmHg, Pulses: {self.pulse_signal_idx}/{len(self.pulse_sig_retf) if self.pulse_sig_retf is not None else 'N/A'}")

            if self.cuff_pressure <= self.min_pressure:
                self.state = self.STATES["COMPLETE"]
                self.status_label.setText("Deflation complete. Analyzing data...")
                # Ensure raw_data has enough length for final analysis
                if len(self.raw_data) > 0:
                    QtCore.QTimer.singleShot(100, self.calculate_final_bpm)
                else:
                    self.bpm_label.setText("Final BPM: No data")
                    self.status_label.setText("Error: No data collected for analysis.")
                return

        elif self.state == self.STATES["COMPLETE"]:
            self.timer.stop()
            return

        # Always append to full raw_data and time_data
        self.raw_data.append(current_cuff_pressure_sample)
        self.time_data.append(current_elapsed_s)

        # Update the rolling buffer for live peak detection
        self.filtered_buffer_for_live_peaks = np.roll(self.filtered_buffer_for_live_peaks, -1)
        # Apply filter to the last few raw data points to get the latest filtered value
        # This is not ideal as it uses the full filter length on a small window.
        # A more correct way is to use a fixed-size buffer and filter it completely.
        # For simulation, this approximation is okay.

        # To get the latest filtered value for the rolling buffer correctly:
        # Use the last 'len(self.b_low)' raw data points
        if len(self.raw_data) >= len(self.b_low):
            recent_raw_data = np.array(self.raw_data[-len(self.b_low):])
            filtered_point = lfilter(self.b_low, self.a_low, recent_raw_data)[-1]
            self.filtered_buffer_for_live_peaks[-1] = filtered_point
        else:
            self.filtered_buffer_for_live_peaks[-1] = 0 # No filtered data yet

        # Live peak detection only during deflation
        if self.state == self.STATES["DEFLATING"]:
            peak_height_live = 0.1 * self.pulse_amp_scale # Adjust based on expected filtered pulse amplitude
            prominence_live = 0.05 * self.pulse_amp_scale
            distance_live = int(self.fs * 0.3) # Minimum 0.3 seconds between peaks

            # Detect peaks on the rolling filtered buffer
            self.live_peaks, _ = find_peaks(self.filtered_buffer_for_live_peaks,
                                            height=peak_height_live,
                                            prominence=prominence_live,
                                            distance=distance_live)

            if len(self.live_peaks) > 1:
                rr_intervals_live = np.diff(self.live_peaks) / self.fs
                valid_intervals_live = rr_intervals_live[
                    (rr_intervals_live > 0.3) & (rr_intervals_live < 2.0)] # BPM range 30-200
                if len(valid_intervals_live) > 0:
                    current_bpm_live = 60 / np.mean(valid_intervals_live)
                    self.all_bpms_live.append(current_bpm_live)
                    # Smooth BPM display
                    displayed_bpm = np.mean(self.all_bpms_live)
                    self.bpm_label.setText(f"BPM: {displayed_bpm:.1f} (Live)")
                else:
                    self.bpm_label.setText(f"BPM: -- (Live)")
            else:
                self.bpm_label.setText(f"BPM: -- (Live)")

            # Update peak scatter on the raw plot, aligned with the raw data
            # Adjust live_peaks indices to match the full raw_data time series
            if len(self.live_peaks) > 0:
                # The live_peaks are indices within filtered_buffer_for_live_peaks.
                # We need to map them back to the full time_data/raw_data array.
                # The "start" index of the live buffer in the full raw_data array
                full_data_start_idx = max(0, len(self.raw_data) - len(self.filtered_buffer_for_live_peaks))
                peak_x_coords_full_data = [full_data_start_idx + p for p in self.live_peaks]

                # Ensure these indices are valid for self.raw_data
                valid_peak_x_coords_full_data = [idx for idx in peak_x_coords_full_data if idx < len(self.raw_data)]
                peak_y_coords = [self.raw_data[idx] for idx in valid_peak_x_coords_full_data]

                self.peak_scatter.setData(np.array(self.time_data)[valid_peak_x_coords_full_data], peak_y_coords)

                if self.peak_scatter not in self.raw_plot.items:
                    self.raw_plot.addItem(self.peak_scatter)
            else:
                self.peak_scatter.clear()
                if self.peak_scatter in self.raw_plot.items:
                    self.raw_plot.removeItem(self.peak_scatter)
        else: # If not in deflation, clear live peaks from raw plot
            self.peak_scatter.clear()
            if self.peak_scatter in self.raw_plot.items:
                self.raw_plot.removeItem(self.peak_scatter)

        # Update raw curve with all accumulated data
        if self.raw_data:
            self.raw_curve.setData(self.time_data, self.raw_data)
            # Auto-range the X-axis for the raw plot
            self.raw_plot.enableAutoRange(axis='x', enable=True)
            self.raw_plot.enableAutoRange(axis='y', enable=True)


        # Update spectrogram during deflation only
        if self.state == self.STATES["DEFLATING"]:
            # Need at least one segment of raw data to calculate a spectrogram column
            if len(self.raw_data) >= len(self.time_data) and len(self.raw_data) >= self.nperseg:
                # Get the most recent segment from the raw_data
                # We need to handle the overlap correctly.
                # The spectrogram update should ideally be triggered when a new segment
                # of (nperseg - noverlap) samples is available.
                # For simplicity, we will re-calculate the last window as needed,
                # or ensure enough data is available for a new "column".

                # Let's use a simpler approach: check if enough new samples arrived for a new column
                # The index of the last sample considered for the *previous* spectrogram column
                last_spectrogram_idx_processed = 0 if not self.spectrogram_data else \
                    (len(self.spectrogram_data) - 1) * (self.nperseg - self.noverlap) + self.nperseg

                # If current raw_data length has enough new samples to form a new window
                if len(self.raw_data) >= last_spectrogram_idx_processed + (self.nperseg - self.noverlap):
                    # Get the segment for the new column
                    start_idx = len(self.raw_data) - self.nperseg # Last Nperseg samples
                    if start_idx < 0: start_idx = 0 # Should not happen if len(raw_data) >= nperseg

                    current_segment = np.array(self.raw_data[start_idx:])

                    # Apply a window function to the segment
                    windowed_segment = current_segment * windows.hann(self.nperseg)

                    # Compute FFT
                    fft_result = np.fft.fft(windowed_segment)

                    # Take magnitude and only positive frequencies
                    magnitude_spectrum = np.abs(fft_result[:self.nperseg // 2])

                    # Convert to dB scale for better visualization
                    magnitude_spectrum_db = 10 * np.log10(magnitude_spectrum + 1e-10)

                    self.spectrogram_data.append(magnitude_spectrum_db)

                    # Convert list of columns to 2D numpy array for ImageItem
                    spec_array = np.array(self.spectrogram_data).T

                    self.spectrogram_img.setImage(spec_array)

                    # Calculate time scale for the X-axis of the spectrogram
                    # The X-axis represents the time since the start of the *deflation phase*
                    time_per_col = (self.nperseg - self.noverlap) / self.fs
                    x_min_spec = self.spectrogram_time_offset
                    x_max_spec = self.spectrogram_time_offset + len(self.spectrogram_data) * time_per_col

                    # Calculate frequency scale for the Y-axis (Nyquist frequency)
                    freq_max = self.fs / 2

                    # Set the rect for the spectrogram image (x, y, width, height)
                    self.spectrogram_img.setRect(QtCore.QRectF(x_min_spec, 0, x_max_spec - x_min_spec, freq_max))

                    # Set the view range of the spectrogram plot (axis labels)
                    self.spectrogram_plot.setXRange(x_min_spec, x_max_spec, padding=0)
                    self.spectrogram_plot.setYRange(0, freq_max, padding=0)

        QtWidgets.QApplication.processEvents()

    def calculate_final_bpm(self):
        self.status_label.setText("Performing final BPM analysis...")
        # Check if enough raw data was collected for meaningful filtering
        if not self.raw_data or len(self.raw_data) < len(self.b_low):
            self.bpm_label.setText("Final BPM: No/Not enough data")
            self.status_label.setText("Error: No or insufficient data collected for final analysis.")
            return

        # Perform filtering on the entire raw dataset for final analysis
        self.analysis_filtered_data = lfilter(self.b_low, self.a_low, self.raw_data)

        # Subtract mean to center the signal around zero for consistent peak detection
        signal_for_final_peaks = self.analysis_filtered_data - np.mean(self.analysis_filtered_data)

        peak_height_final = 0.2 * self.pulse_amp_scale # Adjust based on expected filtered pulse amplitude
        prominence_final = 0.1 * self.pulse_amp_scale
        distance_final = int(self.fs * 0.3) # Minimum 0.3 seconds between peaks (200 BPM max)

        self.analysis_peaks, _ = find_peaks(signal_for_final_peaks,
                                            height=peak_height_final,
                                            prominence=prominence_final,
                                            distance=distance_final)

        final_bpms_calculated = []
        self.bpm_data_final = []

        if len(self.analysis_peaks) > 1:
            rr_intervals_samples = np.diff(self.analysis_peaks)
            rr_intervals_s = rr_intervals_samples / self.fs
            valid_intervals_s = rr_intervals_s[(rr_intervals_s > 0.3) & (rr_intervals_s < 2.0)] # Filter for 30-200 BPM

            if len(valid_intervals_s) > 0:
                final_bpms_calculated = 60 / valid_intervals_s
                median_bpm_final = np.median(final_bpms_calculated)
                self.bpm_label.setText(f"Final BPM: {median_bpm_final:.1f}")
                self.status_label.setText("Analysis Complete.")
                self.debug_label.setText(
                    f"Debug: Final Peaks: {len(self.analysis_peaks)}, Median BPM: {median_bpm_final:.1f}")

                # Store BPM values associated with the time of the *second* peak of each interval
                for i in range(len(final_bpms_calculated)):
                    peak_index_of_bpm = self.analysis_peaks[i + 1] # BPM is calculated from interval ending at this peak
                    if peak_index_of_bpm < len(self.time_data):
                        self.bpm_data_final.append((peak_index_of_bpm, final_bpms_calculated[i]))
            else:
                self.bpm_label.setText("Final BPM: N/A (intervals out of range)")
                self.status_label.setText("No valid heart rate intervals in final analysis.")
        else:
            self.bpm_label.setText("Final BPM: N/A (not enough peaks)")
            self.status_label.setText("Not enough peaks for final BPM analysis.")

        try:
            csv_file = self.save_data_to_csv()
            print(f"Data saved to {csv_file}")
            self.plot_results()
        except Exception as e:
            print(f"Error during data saving/plotting: {str(e)}")
            self.status_label.setText(f"Error saving/plotting: {str(e)}")

    def save_data_to_csv(self):
        filename = "blood_pressure_data_final_analysis.csv"
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time (s)", "Raw Pressure (mmHg)", "Low-Pass Filtered (Amplitude)", "BPM (at peak time)"])

            # Create a dictionary for quick lookup of BPM at specific time indices
            bpm_dict = {idx: val for idx, val in self.bpm_data_final}

            for i in range(len(self.time_data)):
                bpm_val = bpm_dict.get(i, "") # Get BPM if available for this index, else empty string
                writer.writerow([
                    f"{self.time_data[i]:.3f}",
                    f"{self.raw_data[i]:.2f}",
                    f"{self.analysis_filtered_data[i]:.4f}" if i < len(self.analysis_filtered_data) else "",
                    f"{bpm_val:.1f}" if bpm_val else ""
                ])
        return filename

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.plot(self.time_data, self.raw_data, 'b-', label='Raw Cuff Pressure')
        ax1.set_title('Cuff Pressure (Raw Data)')
        ax1.set_ylabel('Pressure (mmHg)')
        ax1.grid(True)
        ax1.legend()

        if self.analysis_filtered_data is not None and len(self.analysis_filtered_data) > 0:
            ax2.plot(self.time_data, self.analysis_filtered_data, 'g-', label='Low-Pass Filtered Signal')
            if self.analysis_peaks is not None and len(self.analysis_peaks) > 0:
                # Ensure peak indices are within bounds of time_data and analysis_filtered_data
                valid_peaks_plot = [p for p in self.analysis_peaks if
                                    p < len(self.time_data) and p < len(self.analysis_filtered_data)]
                if valid_peaks_plot:
                    ax2.plot(np.array(self.time_data)[valid_peaks_plot],
                             np.array(self.analysis_filtered_data)[valid_peaks_plot],
                             "ro", markersize=5, label='Detected Peaks (Final)')
            ax2.set_title('Low-Pass Filtered Signal & Detected Peaks (Final Analysis)')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Filtered Amplitude')
            ax2.grid(True)
            ax2.legend()
        else:
            ax2.set_title('Low-Pass Filtered Signal (No data or error)')

        final_bpm_text = self.bpm_label.text().replace("Final BPM: ", "")
        plt.suptitle(f"Blood Pressure Analysis - Final BPM: {final_bpm_text}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    monitor = BloodPressureMonitor()
    monitor.show()
    sys.exit(app.exec_())
