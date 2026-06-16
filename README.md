# ♟️ GestureChess

An interactive, computer vision-powered Chess game that allows users to play completely hands-free using real-time hand tracking and intuitive air gestures!

## 🚀 Features
- **Real-Time Hand Tracking:** Powered by Google MediaPipe for ultra-smooth hand landmark detection.
- **Pinch-to-Grab Control:** Move chess pieces by simply pinching your thumb and index finger together.
- **Smart Move Validation:** Integrated with the standard `python-chess` engine to enforce official rules, highlights, and legal moves.
- **Keyboard Shortcuts:** Quick triggers for Undo (`u`), Reset (`r`), and Quit (`q`).

## 📋 System Prerequisites

Before running the project, ensure your environment meets the following conditions:
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python Version:** Python 3.10 or 3.11 (Recommended for MediaPipe stability)
- **Hardware:** Integrated or external USB Webcam

## 🛠️ Tech Stack
- **Language:** Python 3.11+
- **Computer Vision:** OpenCV, MediaPipe
- **Game Engine:** python-chess
- **Matrix Operations:** NumPy

## 📁 Project Structure

```text
GestureChess/
│
├── main.py            # Main application execution engine containing OpenCV loops
├── .gitignore         # Prevents heavy environment folders (like venv) from uploading
└── README.md          # Comprehensive project blueprint and guide


