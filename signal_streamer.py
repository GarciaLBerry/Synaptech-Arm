import queue
import time
import numpy as np
import pandas as pd

from pylsl import resolve_streams, StreamInlet
from model.config import core_cols, window_size

SAMPLE_RATE = 250
CHANNEL_NUM = 8
STREAM_TIMEOUT = 1.01
verbose = False
get_timeout = 0.1

class SignalStreamer:
    _signal_buffer = queue.SimpleQueue()
    _stop_signal = False
    
    def start_streaming(self):
        try:
            streams = resolve_streams()
            eeg_streams = [s for s in streams if s.type() == 'EEG']

            if not eeg_streams:
                raise RuntimeError("No EEG streams found. Make sure OpenBCI GUI or CLI is streaming!")

            inlet = StreamInlet(eeg_streams[0])
            print("Connected to LSL stream:", eeg_streams[0].name())
            while not self._stop_signal:
                samples, ts = inlet.pull_chunk(STREAM_TIMEOUT, int(window_size))
                signals = np.array(samples, dtype=np.float32)
                
                # Replace final 3 columns with default accelerometer values
                accel_defaults = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                signals[:, -3:] = accel_defaults
                # TODO: We probably want to just drop the accelerometer values in the future models instead

                timestamps = np.array(ts, dtype=np.float64).reshape(-1, 1)
                
                extended_samples = np.hstack((timestamps, signals))
                assert extended_samples.shape[0] == window_size

                df = pd.DataFrame(extended_samples, columns=core_cols)
                self._signal_buffer.put(df)
                time.sleep(1)  # Simulate delay between signals
        except KeyboardInterrupt:
            # This catches Ctrl+C/Cmd+C gracefully
            print("\n\nLoop stopped by user. Closing down...")
            
    def stop_streaming(self):
        self._stop_signal = True

    def pop_signal(self):
        try:
            return self._signal_buffer.get(block=False, timeout=get_timeout)
        except queue.Empty:
            if verbose:
                print("[SIGNAL STREAMER] No signals in buffer.")
            return None