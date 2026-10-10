#!/usr/bin/env bash
# Flash a model binary into the "model" partition (offset from partitions.csv).
# Usage: ./flash_model.sh [model.bin] [port]   (default ../../model/model.bin)
set -e
PORT=${2:-/dev/ttyUSB0}
esptool.py --chip esp32 --port "$PORT" --baud 921600 write_flash 0x110000 "${1:-../../model/model.bin}"
