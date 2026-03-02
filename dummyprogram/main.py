import queue
import time
from dummy import predict

def main():
    signal_buffer = queue.SimpleQueue()

    dummy_data = [-1, 0, 1]
    for val in dummy_data:
        signal_buffer.put(val)

    try:
        while True:
            # 3. Check if the buffer has data
            if not signal_buffer.empty():
                # Pop the first element
                raw_signal = signal_buffer.get_nowait()
                
                # 4. Pass to the separate prediction function
                result = predict(raw_signal)

                # 5. Translate and print output
                mapping = {
                    -1: "DOWN",
                    0:  "REST",
                    1:  "UP"
                }
                
                display_text = mapping.get(result, "UNKNOWN SIGNAL")
                print(f"Signal: {raw_signal:2} | Action: {display_text}")

    except KeyboardInterrupt:
        # This catches Ctrl+C/Cmd+C gracefully
        print("\n\nLoop stopped by user. Closing down...")

if __name__ == "__main__":
    main()