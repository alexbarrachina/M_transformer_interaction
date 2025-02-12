


import TMIDIX


def get_midifile_data(midi_file):
    #@title Upload your own seed MIDI
   # midi_file = './Samples/clairTester.midi'
    uploaded_MIDI = open(midi_file, 'rb').read()

    score = TMIDIX.midi2ms_score(uploaded_MIDI)

    events_matrix = []

    itrack = 1

    while itrack < len(score):
        for event in score[itrack]:
            if event[0] == 'note' and event[3] != 9: # I suspect channel=9 are drums
                events_matrix.append(event)
        itrack += 1

    # Sorting...
    events_matrix.sort(key=lambda x: x[4], reverse=True)
    events_matrix.sort(key=lambda x: x[1])

    # recalculating timings
    for e in events_matrix:
        e[1] = int(e[1] / 10) # event time / 10
        e[2] = int(e[2] / 20) # event duration / 20

    # final processing...
    melody_chords = []
    pe = events_matrix[0] # the first event

    for e in events_matrix:

        time = max(0, min(126, e[1]-pe[1])) # time difference from previous events, but trunk to maximum 126
        dur = max(1, min(126, e[2])) # maximum duration 126

        ptc = max(1, min(126, e[4])) # maximum pitch 126
        vel = max(1, min(126, e[5])) # maximum velocity 126

        melody_chords.append([time, dur+128, ptc+256, vel+384]) # 384 = 128*3 all events, melodies + chords
        pe = e

    inputs = []

    for m in melody_chords:
        inputs.extend(m) # add to the end of the input list

    return inputs

def generate_midifile_from_list(out1, nameOut):
# create a midifile from a list of data (time, dur, pitch, vel)

      song = out1
      song_f = []
      time = 0
      dur = 0
      vel = 0
      pitch = 0
      channel = 0

      son = []

      song1 = []

      for s in song:
        if s > 127:
          son.append(s)

        else:
          if len(son) == 4:
            song1.append(son)
          son = []
          son.append(s)

      for s in song1:

          time += s[0] * 10

          dur = (s[1]-128) * 20

          channel = 0

          pitch = s[2]-256

          vel = s[3]-384

          song_f.append(['note', time, dur, channel, pitch, vel ])


      detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature = 'GIGA-Piano XL',
                                                                output_file_name = nameOut,
                                                                track_name='Project Los Angeles',
                                                                list_of_MIDI_patches=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                                                )

