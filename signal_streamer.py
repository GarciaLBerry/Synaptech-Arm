import queue
import time
import numpy as np
import pandas as pd

from pylsl import resolve_streams, StreamInlet
from model.config import SAMPLE_RATE, PACKET_SIZE, PACKET_STRIDE, core_cols

MODEL_CHANNEL_NUM = len(core_cols)

packet_duration = PACKET_SIZE / SAMPLE_RATE
sample_delay = PACKET_STRIDE / SAMPLE_RATE
stream_timeout = sample_delay + 0.01
signal_buffer_get_timeout = sample_delay * 0.1

verbose = False

class SignalStreamer:
    _signal_buffer = queue.SimpleQueue()
    _packet_buffer: np.ndarray | None = None
    _stop_signal = False
    
    def start_streaming(self):
        try:
            streams = resolve_streams()
            eeg_streams = [s for s in streams if s.type() == 'EEG']

            if not eeg_streams:
                raise RuntimeError("No EEG streams found. Make sure OpenBCI GUI or CLI is streaming!")

            inlet = StreamInlet(eeg_streams[0])
            inlet.flush()
            print("Connected to LSL stream:", eeg_streams[0].name())
            while not self._stop_signal:
                samples, _ = inlet.pull_chunk(stream_timeout, PACKET_STRIDE)
                signals = np.array(samples, dtype=np.float32)
                
                if signals.shape[0] > PACKET_STRIDE:
                    if verbose:
                        print(f"[SIGNAL STREAMER] Warning: Received {signals.shape[0]} samples, expected {PACKET_STRIDE}\n")
                
                if signals.shape[0] < PACKET_STRIDE:
                    if verbose:
                        print(f"[SIGNAL STREAMER] Warning: Received {signals.shape[0]} samples, expected {PACKET_STRIDE}.\n")
                
                signals = signals[:, :MODEL_CHANNEL_NUM]
                
                if self._packet_buffer is None:
                    self._packet_buffer = signals.T
                else:
                    self._packet_buffer = np.concatenate((self._packet_buffer, signals.T), axis=1)
                    
                while self._packet_buffer.shape[1] >= PACKET_SIZE:
                    self._signal_buffer.put(self._packet_buffer[:, :PACKET_SIZE][None, :, :])
                    self._packet_buffer = self._packet_buffer[:, PACKET_STRIDE:]
                
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
