from midiUtils import get_midifile_data, generate_midifile_from_list

"""# SETTINGS """
# Play with the settings to get different results
midi_file = './Samples/clairTester.midi'
midi_file_out = './Out/save_midi_test'


"""# GET PRIMER FROM MIDI FILE """

inputs = get_midifile_data(midi_file)

generate_midifile_from_list(inputs, midi_file_out) 

inputs2 = get_midifile_data(midi_file_out+'.mid')

print('Done!')

