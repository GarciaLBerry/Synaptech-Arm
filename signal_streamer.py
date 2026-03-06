import queue
import time

temp_signal = [
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 0.3],
]

verbose = False
get_timeout = 0.1

class SignalStreamer:
    signal_buffer = queue.SimpleQueue()
    stop_signal = False
    
    def start_streaming(self):
        try:
            while not self.stop_signal:
                self.signal_buffer.put(temp_signal)
                time.sleep(1)  # Simulate delay between signals
        except KeyboardInterrupt:
            # This catches Ctrl+C/Cmd+C gracefully
            print("\n\nLoop stopped by user. Closing down...")
            
    def stop_streaming(self):
        self.stop_signal = True

    def pop_signal(self):
        try:
            return self.signal_buffer.get(block=False, timeout=get_timeout)
        except queue.Empty:
            if verbose:
                print("[SIGNAL STREAMER] No signals in buffer.")
            return None