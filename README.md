# Action Recognition with Mobile Phone Detection

## Overview
This project is designed to detect when a person is using a mobile phone and to send notifications through a Telegram bot. It leverages the YOLO object detection model along with MediaPipe for enhanced hand tracking capabilities. This setup allows for real-time detection and notification, making it suitable for monitoring areas where phone usage is restricted or for gathering data on phone usage in specific scenarios.

## Features
- **Real-Time Detection**: Utilizes YOLO to detect persons and mobile phones in real-time.
- **Hand Tracking**: Integrates MediaPipe to track hand movements and correlate them with phone usage.
- **Notification System**: Sends automated alerts via a Telegram bot when a mobile phone is detected in use.

## Requirements
- Python 3.6+
- OpenCV
- PyTorch
- MediaPipe
- Ultralytics YOLO
- `telebot` Python package for Telegram Bot integration

## Usage
To start the detection and notification system, run:
Make sure to configure your Telegram bot token and the target chat ID in the script to receive notifications.

## How It Works
The system processes video input from a camera source, using YOLO to detect humans and mobile phones within the frame. MediaPipe is employed to precisely track hand positions, determining if a detected phone is being held. If phone usage is confirmed, a message is sent through the Telegram bot, including the time of detection and an image snapshot.

## Customization
You can adjust detection sensitivity and other parameters by modifying the configuration settings within the `ActionRecognition` class.


