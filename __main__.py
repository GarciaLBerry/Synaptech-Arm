import threading, os, shutil, time
from pathlib import Path
from model.inference import predict
import signal_streamer as ss
import numpy as np
from model import prediction_mapping
from model.config import MAIN_LOOP_DELAY, MAIN_LOOP_TIMEOUT

def main():
    if Path.cwd() != Path(__file__).parent:
        os.chdir(Path(__file__).parent)
        
    # 1. Start the signal streamer in a separate thread
    streamer = ss.SignalStreamer()
    signal_thread = threading.Thread(target=streamer.start_streaming)
    signal_thread.start()
    
    try:
        timeout_counter = 0
        # 2. Start the main loop to process signals
        while True:
            if timeout_counter >= MAIN_LOOP_TIMEOUT:
                break
            # 3. Check if the buffer has data
            raw_signal = streamer.pop_signal()
            if raw_signal is not None:

                # 4. Pass to the separate prediction function
                results = predict(raw_signal)
                if type(results) is np.ndarray:
                    result = results[0]
                else:
                    result = int(results) 

                
                # 5. Placeholder: Translate and print output
                if result not in prediction_mapping:
                    display_text = "UNKNOWN SIGNAL"
                else:
                    display_text = prediction_mapping[result]

                status = f"Action: {display_text}; Timeout: [{timeout_counter}]"
                print(status.ljust(shutil.get_terminal_size().columns), end='\r', flush=True)
                timeout_counter = 0
                
                
            timeout_counter += MAIN_LOOP_DELAY
            time.sleep(MAIN_LOOP_DELAY)

    except KeyboardInterrupt:
        # This catches Ctrl+C/Cmd+C gracefully
        print("\n\nLoop stopped by user. Closing down...")
        streamer.stop_streaming()
        signal_thread.join()
        exit(0)
    
    streamer.stop_streaming()
    signal_thread.join()
    exit(0)

if __name__ == "__main__":
    main()
