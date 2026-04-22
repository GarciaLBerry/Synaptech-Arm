import queue
import time
import numpy as np
from pylsl import resolve_streams, StreamInlet

SAMPLE_RATE = 250
WINDOW_SIZE = 250
CHANNEL_NUM = 8
STREAM_TIMEOUT = 1
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
                sample, timestamp = inlet.pull_chunk(STREAM_TIMEOUT, int(WINDOW_SIZE))
                signal = np.array(sample, dtype=np.float32)
                assert signal.shape[0] == WINDOW_SIZE
                self._signal_buffer.put(signal)
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