"""
Wand Receiver — BLE accelerometer data capture + live visualization
Controls:
  SPACE  — start/stop recording
  1      — label: still
  2      — label: left
  3      — label: right
  q      — quit
"""

import asyncio
import struct
import json
import threading
import sys
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
WAND_UUID_FULL = "0000bbbb-0000-1000-8000-00805f9b34fb"
DATA_DIR       = Path("data")
PLOT_WINDOW    = 100

# ── GLOBALS ───────────────────────────────────────────────────────────────────
live_ax = deque([0] * PLOT_WINDOW, maxlen=PLOT_WINDOW)
live_ay = deque([0] * PLOT_WINDOW, maxlen=PLOT_WINDOW)
live_az = deque([0] * PLOT_WINDOW, maxlen=PLOT_WINDOW)

recording       = False
current_gesture = []
current_label   = "still"
gesture_count   = {cls: len(list((DATA_DIR / cls).glob("*.json")))
                   if (DATA_DIR / cls).exists() else 0
                   for cls in ["still", "left", "right"]}
last_status     = "Scanning for wand..."

LABELS = {"1": "still", "2": "left", "3": "right"}

# ── BLE PARSING ───────────────────────────────────────────────────────────────

def parse_packet(data: bytes):
    if len(data) < 7:
        return []
    samples = []
    raw = data[1:]
    for offset in range(0, len(raw) - 5, 6):
        ax, ay, az = struct.unpack_from('<3h', raw, offset)
        samples.append({"ax": ax, "ay": ay, "az": az})
    return samples

# ── SAVE GESTURE ──────────────────────────────────────────────────────────────

def save_gesture(samples, label):
    folder = DATA_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    count = gesture_count[label] + 1
    gesture_count[label] = count
    filename = folder / f"{label}_data_{count:05d}.json"
    payload = {
        "tag":       label,
        "count":     count,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000"),
        "sensors":   samples,
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    return filename

# ── KEYBOARD ──────────────────────────────────────────────────────────────────

def keyboard_listener():
    global recording, current_gesture, current_label, last_status
    import tty, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == 'q':
                os._exit(0)
            elif ch == ' ':
                if not recording:
                    recording = True
                    current_gesture = []
                    last_status = f"⏺  Recording [{current_label}]... SPACE to stop"
                else:
                    recording = False
                    if current_gesture:
                        fname = save_gesture(current_gesture, current_label)
                        last_status = (f"✅ Saved {len(current_gesture)} samples → "
                                       f"{fname.name} "
                                       f"(total {current_label}: {gesture_count[current_label]})")
                    else:
                        last_status = "⚠️  Nothing recorded"
            elif ch in LABELS:
                current_label = LABELS[ch]
                last_status = f"Label → [{current_label}]"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ── PLOT ──────────────────────────────────────────────────────────────────────

def run_plot():
    import matplotlib
    matplotlib.use("MacOSX")
    import matplotlib.pyplot as plt

    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6),
                                          sharex=True, facecolor="#0f0f0f")
    fig.suptitle("✦ Wand Live Feed", color="white", fontsize=13, fontweight="bold")
    fig.subplots_adjust(hspace=0.4, bottom=0.1)

    panels = [
        (ax1, live_ax, "#00d4ff", "ax"),
        (ax2, live_ay, "#ff6b6b", "ay"),
        (ax3, live_az, "#69ff47", "az"),
    ]
    lines = []
    for panel, _, color, label in panels:
        panel.set_facecolor("#1a1a1a")
        panel.tick_params(colors="white", labelsize=8)
        for spine in panel.spines.values():
            spine.set_edgecolor("#333")
        panel.set_xlim(0, PLOT_WINDOW)
        panel.set_ylim(-600, 600)
        panel.axhline(0, color="#333", linewidth=0.8)
        panel.set_ylabel(label, color=color, fontsize=10, fontweight="bold")
        line, = panel.plot(range(PLOT_WINDOW), [0] * PLOT_WINDOW, color=color, linewidth=1.4)
        lines.append(line)

    status = fig.text(0.02, 0.02, last_status, color="#aaaaaa",
                      fontsize=9, fontfamily="monospace")
    rec    = fig.text(0.80, 0.02, "", color="#ff4444", fontsize=10, fontweight="bold")
    fig.text(0.02, 0.965,
             "SPACE: record   1: still   2: left   3: right   q: quit",
             color="#555", fontsize=8, fontfamily="monospace")

    while plt.fignum_exists(fig.number):
        for line, buf in zip(lines, [live_ax, live_ay, live_az]):
            line.set_ydata(list(buf))
        status.set_text(last_status)
        rec.set_text("● REC" if recording else "")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)

# ── BLE ───────────────────────────────────────────────────────────────────────

async def ble_scan():
    global last_status
    from bleak import BleakScanner

    def callback(device, advertisement_data):
        global last_status
        data = advertisement_data.service_data.get(WAND_UUID_FULL)
        if not data:
            return
        samples = parse_packet(data)
        if not samples:
            return
        for s in samples:
            live_ax.append(s["ax"])
            live_ay.append(s["ay"])
            live_az.append(s["az"])
            if recording:
                current_gesture.append(s)
        if not recording:
            s = samples[-1]
            last_status = (f"📡 ax={s['ax']:5d}  ay={s['ay']:5d}  az={s['az']:5d}"
                           f"    label: [{current_label}]   SPACE to record")

    async with BleakScanner(callback):
        while True:
            await asyncio.sleep(0.1)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    for cls in ["still", "left", "right"]:
        (DATA_DIR / cls).mkdir(parents=True, exist_ok=True)

    print("Wand Receiver starting...")
    print("Controls: SPACE=record  1=still  2=left  3=right  q=quit")

    threading.Thread(target=lambda: asyncio.run(ble_scan()), daemon=True).start()
    threading.Thread(target=keyboard_listener, daemon=True).start()

    run_plot()

if __name__ == "__main__":
    main()