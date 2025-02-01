

from midiUtils import get_midifile_data, generate_midifile_from_list

"""# SETTINGS """
# Play with the settings to get different results
midi_file = './Samples/scott_chords.midi'
nameOut = './Out/scott_chords_prior.midi'


number_of_prime_notes = 128 # min:32, max:256

"""# GET PRIMER FROM MIDI FILE """

inputs = get_midifile_data(midi_file)


"""# INFERENCE  """

inp = inputs[:number_of_prime_notes*4] # 128*4 = 512
# If you want to generate from a blank state
#inp = [126, 126+128, 0+256, 0+384]

  #with ctx:

generate_midifile_from_list(inp, nameOut)