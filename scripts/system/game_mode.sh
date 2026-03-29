#!/bin/bash
# Wyłączamy pionowy monitor
xrandr --output DisplayPort-0 --off
# Ustawiamy główny na 1920x1080 i centrujemy pozycję
xrandr --output HDMI-A-0 --mode 1920x1080 --pos 0x0
echo "Tryb gry aktywny: 1080p 16:9, boczny monitor OFF"
