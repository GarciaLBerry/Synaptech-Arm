import queue
import time
import numpy as np
import pandas as pd

from pylsl import resolve_streams, StreamInlet
from model.config import PACKET_SIZE

SAMPLE_RATE = 250
CHANNEL_NUM = 8

packet_duration = SAMPLE_RATE / PACKET_SIZE
stream_timeout = packet_duration + 0.01
signal_buffer_get_timeout = packet_duration * 0.1

verbose = False

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
                samples, ts = inlet.pull_chunk(stream_timeout, int(PACKET_SIZE))
                signals = np.array(samples, dtype=np.float32)
                timestamps = np.array(ts, dtype=np.float64).reshape(-1, 1)
                extended_samples = np.hstack((timestamps, signals))
                
                assert extended_samples.shape[0] == PACKET_SIZE

                self._signal_buffer.put(extended_samples.T[None, :, :])
                
        except KeyboardInterrupt:
            # This catches Ctrl+C/Cmd+C gracefully
            print("\n\nLoop stopped by user. Closing down...")
            
    def stop_streaming(self):
        self._stop_signal = True

    def pop_signal(self):
        try:
            return self._signal_buffer.get(block=False, timeout=signal_buffer_get_timeout)
        except queue.Empty:
            if verbose:
                print("[SIGNAL STREAMER] No signals in buffer.")
            return None