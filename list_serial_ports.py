'''Utility to list COM ports available on this machine'''

import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

for port in ports:
    print(port)

