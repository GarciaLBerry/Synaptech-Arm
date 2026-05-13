import queue
import time
import numpy as np
import pandas as pd

from pylsl import resolve_streams, StreamInlet
from model.config import PACKET_SIZE, core_cols

SAMPLE_RATE = 250
CHANNEL_NUM = 8
MODEL_CHANNEL_NUM = len(core_cols)

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
            inlet.flush()
            print("Connected to LSL stream:", eeg_streams[0].name())
            while not self._stop_signal:
                samples, _ = inlet.pull_chunk(stream_timeout, int(PACKET_SIZE))
                signals = np.array(samples, dtype=np.float32)
                
                if signals.shape[0] > PACKET_SIZE:
                    if verbose:
                        print(f"[SIGNAL STREAMER] Warning: Received {signals.shape[0]} samples, expected {PACKET_SIZE}. Trimming to expected size.\n")
                    signals = signals[:PACKET_SIZE, :]
                
                if signals.shape[0] < PACKET_SIZE:
                    if verbose:
                        print(f"[SIGNAL STREAMER] Warning: Received {signals.shape[0]} samples, expected {PACKET_SIZE}. Unable to correct.\n")
                

                signals = signals[:, :5]
                self._signal_buffer.put(signals.T[None, :, :])
                
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
