#!/bin/bash
# Przywracamy ultrawide
xrandr --output HDMI-A-0 --mode 2560x1080 --pos 1080x500
# Przywracamy pionowy monitor (pozycja 0x0, rotacja w prawo)
xrandr --output DisplayPort-0 --mode 1920x1080 --pos 0x0 --rotate right
echo "Tryb pulpitu przywrócony: Ultrawide + Pionowy"
