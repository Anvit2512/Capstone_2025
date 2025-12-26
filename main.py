import serial
import time
import pandas as pd
import matplotlib.pyplot as plt

def read_sweep(ser, expected_sweep_id=None, timeout_s=20):
    """
    Read one sweep worth of data from ESP32.
    Stops when it sees a line 'DONE' or timeout.
    Returns a pandas DataFrame with columns:
    ['sweep_id', 'time_us', 'freq_hz', 'adc']
    """
    start_time = time.time()
    rows = []

    while True:
        if time.time() - start_time > timeout_s:
            print("⚠ Timeout while waiting for sweep data.")
            break

        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        if line == "DONE":
            # Sweep completed
            break

        if line.startswith("#"):
            # Comment/status line from ESP32; ignore
            continue

        parts = line.split(",")
        if len(parts) != 4:
            # Not a data line; skip
            continue

        try:
            sweep_id = int(parts1 := parts[0])
            t_us     = int(parts[1])
            freq_hz  = float(parts[2])
            adc      = int(parts[3])

            if expected_sweep_id is not None and sweep_id != expected_sweep_id:
                # Different sweep ID than expected, but still record it
                pass

            rows.append((sweep_id, t_us, freq_hz, adc))
        except ValueError:
            # Parsing error; skip this line
            continue

    if not rows:
        print("⚠ No valid data received for this sweep.")
        return pd.DataFrame(columns=["sweep_id", "time_us", "freq_hz", "adc"])

    df = pd.DataFrame(rows, columns=["sweep_id", "time_us", "freq_hz", "adc"])
    return df

def main():
    port = input("Enter serial port (e.g. COM5 or /dev/ttyUSB0): ").strip()
    user = input("Enter user name (for file naming): ").strip()

    # Open serial
    ser = serial.Serial(port, baudrate=115200, timeout=1)
    time.sleep(2)  # small delay to let ESP32 reset

    # Flush any startup lines
    ser.reset_input_buffer()

    results = {}

    for eye, cmd, sweep_id in [("left", "L", 0), ("right", "R", 1)]:
        input(f"\n👉 Put the device on the {eye.upper()} eye and press Enter to start...")
        print(f"Starting {eye} eye sweep...")

        # Send command to ESP32
        ser.write(cmd.encode("ascii"))

        # Read one sweep worth of data
        df = read_sweep(ser, expected_sweep_id=sweep_id, timeout_s=25)

        # Save CSV
        if not df.empty:
            filename = f"{user}_{eye}_eye_sweep.csv"
            df.to_csv(filename, index=False)
            print(f"✅ Saved {eye} eye data to {filename}")
        else:
            print(f"⚠ No data saved for {eye} eye (empty sweep).")

        results[eye] = df

    ser.close()

    # Plot results if we have any data
    plt.figure(figsize=(10, 6))

    for eye, color in [("left", "tab:blue"), ("right", "tab:orange")]:
        df = results.get(eye)
        if df is None or df.empty:
            continue
        t_s = df["time_us"] / 1e6  # convert µs to seconds
        adc = df["adc"]
        plt.plot(t_s, adc, label=f"{eye} eye")

    plt.xlabel("Time (seconds)")
    plt.ylabel("ADC value")
    plt.title(f"IOP Sweep Response - {user}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
