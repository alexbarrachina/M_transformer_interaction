import fluidsynth
import time

fs = fluidsynth.Synth()
fs.start()

sfid = fs.sfload("./piano.sf2")
fs.program_select(0, sfid, 0, 0)

fs.noteon(0, 60, 120)
time.sleep(1)
fs.noteoff(0, 60)

