# Reference: https://seisman.github.io/HinetPy/tutorial.html

from HinetPy import Client
from datetime import datetime
from HinetPy import win32

# use your own account
client = Client("anna8601030", "abc661557")

client.doctor()

# select stations of Hi-net if you know the station names
client.select_stations('0101', ['N.MISH', 'N.IKTH', 'N.UWAH', 'N.KWBH', 'N.OOZH', 'N.HIYH', 'N.YNDH', 'N.TBEH', 'N.IKKH', 'N.SJOH', 'N.TBRH', 'N.GHKH'])

# JST time [All times in HinetPy and Hi-net website are in JST time (GMT+0900).]
starttime = datetime(2014, 9, 10, 0, 0)
data, ctable = client.get_continuous_waveform('0101', starttime, 1440)

data = "0101_201409100000_1440.cnt"
ctable = "0101_20140910.ch"
win32.extract_sac(data, ctable)

win32.extract_pz(ctable)
win32.extract_pz(ctable, suffix="SACPZ", outdir="PZ/")