import sys, struct, copy, os
import pickle

Version = '6.7'
VersionDate = '20201120'

_previous_warning = ''  # 5.4
_previous_times = 0     # 5.4

#----------------------------- Event stuff --------------------------

# Some public-access tuples:
MIDI_events = tuple('''note_off note_on key_after_touch
control_change patch_change channel_after_touch
pitch_wheel_change'''.split())

Text_events = tuple('''text_event copyright_text_event
track_name instrument_name lyric marker cue_point text_event_08
text_event_09 text_event_0a text_event_0b text_event_0c
text_event_0d text_event_0e text_event_0f'''.split())

Nontext_meta_events = tuple('''end_track set_tempo
smpte_offset time_signature key_signature sequencer_specific
raw_meta_event sysex_f0 sysex_f7 song_position song_select
tune_request'''.split())
# unsupported: raw_data

Meta_events = Text_events + Nontext_meta_events
All_events  = MIDI_events + Meta_events

###################################################################################

def Tegridy_Any_Pickle_File_Writer(Data, input_file_name='TMIDI_Pickle_File'):

  '''Tegridy Pickle File Writer
     
  Input: Data to write (I.e. a list)
         Full path and file name without extention
         
  Output: Named Pickle file

  Project Los Angeles
  Tegridy Code 2021'''

  print('Tegridy Pickle File Writer')

  full_path_to_output_dataset_to = input_file_name + '.pickle'

  if os.path.exists(full_path_to_output_dataset_to):
    os.remove(full_path_to_output_dataset_to)
    print('Removing old Dataset...')
  else:
    print("Creating new Dataset file...")

  with open(full_path_to_output_dataset_to, 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(Data, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

  print('Dataset was saved as:', full_path_to_output_dataset_to)
  print('Task complete. Enjoy! :)')

###################################################################################

def Tegridy_Any_Pickle_File_Reader(input_file_name='TMIDI_Pickle_File', ext='.pickle'):

  '''Tegridy Pickle File Loader
     
  Input: Full path and file name without extention
         File extension if different from default .pickle
       
  Output: Standard Python 3 unpickled data object

  Project Los Angeles
  Tegridy Code 2021'''

  print('Tegridy Pickle File Loader')
  print('Loading the pickle file. Please wait...')

  with open(input_file_name + ext, 'rb') as pickle_file:
    content = pickle.load(pickle_file)

  return content

###################################################################################

def Tegridy_ms_SONG_to_MIDI_Converter(SONG,
                                      output_signature = 'Tegridy TMIDI Module', 
                                      track_name = 'Composition Track',
                                      list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 0, 0, 0, 0, 0, 0],
                                      output_file_name = 'TMIDI-Composition',
                                      text_encoding='ISO-8859-1',
                                      verbose=True):

    '''Tegridy milisecond SONG to MIDI Converter
     
    Input: Input ms SONG in TMIDI ms SONG/MIDI.py ms Score format
           Output MIDI Track 0 name / MIDI Signature
           Output MIDI Track 1 name / Composition track name
           List of 16 MIDI patch numbers for output MIDI. Def. is MuseNet compatible patches.
           Output file name w/o .mid extension.
           Optional text encoding if you are working with text_events/lyrics. This is especially useful for Karaoke. Please note that anything but ISO-8859-1 is a non-standard way of encoding text_events according to MIDI specs.

    Output: MIDI File
            Detailed MIDI stats

    Project Los Angeles
    Tegridy Code 2020'''                                  
    
    if verbose:
        print('Converting to MIDI. Please stand-by...')

    output_header = [1000,
                    [['set_tempo', 0, 1000000],
                     ['time_signature', 0, 4, 2, 24, 8],
                     ['track_name', 0, bytes(output_signature, text_encoding)]]]

    patch_list = [['patch_change', 0, 0, list_of_MIDI_patches[0]], 
                    ['patch_change', 0, 1, list_of_MIDI_patches[1]],
                    ['patch_change', 0, 2, list_of_MIDI_patches[2]],
                    ['patch_change', 0, 3, list_of_MIDI_patches[3]],
                    ['patch_change', 0, 4, list_of_MIDI_patches[4]],
                    ['patch_change', 0, 5, list_of_MIDI_patches[5]],
                    ['patch_change', 0, 6, list_of_MIDI_patches[6]],
                    ['patch_change', 0, 7, list_of_MIDI_patches[7]],
                    ['patch_change', 0, 8, list_of_MIDI_patches[8]],
                    ['patch_change', 0, 9, list_of_MIDI_patches[9]],
                    ['patch_change', 0, 10, list_of_MIDI_patches[10]],
                    ['patch_change', 0, 11, list_of_MIDI_patches[11]],
                    ['patch_change', 0, 12, list_of_MIDI_patches[12]],
                    ['patch_change', 0, 13, list_of_MIDI_patches[13]],
                    ['patch_change', 0, 14, list_of_MIDI_patches[14]],
                    ['patch_change', 0, 15, list_of_MIDI_patches[15]],
                    ['track_name', 0, bytes(track_name, text_encoding)]]

    output = output_header + [patch_list + SONG]

    midi_data = score2midi(output, text_encoding)
    #detailed_MIDI_stats = score2stats(output)

    with open(output_file_name + '.mid', 'wb') as midi_file:
        midi_file.write(midi_data)
        midi_file.close()
    
    if verbose:    
        print('Done! Enjoy! :)')
    
    return None


def get_midifile_data(midi_file):
    #@title Upload your own seed MIDI
   # midi_file = './Samples/clairTester.midi'
    uploaded_MIDI = open(midi_file, 'rb').read()

    score = midi2ms_score(uploaded_MIDI)

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


      detailed_stats = Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature = 'GIGA-Piano XL',
                                                                output_file_name = nameOut,
                                                                track_name='Project Los Angeles',
                                                                list_of_MIDI_patches=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                                                )


#------------------------------- Encoding stuff --------------------------

def score2midi(score=None, text_encoding='ISO-8859-1'):
    r'''
Translates a "score" into MIDI, using score2opus() then opus2midi()
'''
    return opus2midi(score2opus(score, text_encoding), text_encoding)

def opus2midi(opus=[], text_encoding='ISO-8859-1'):
    r'''The argument is a list: the first item in the list is the "ticks"
parameter, the others are the tracks. Each track is a list
of midi-events, and each event is itself a list; see above.
opus2midi() returns a bytestring of the MIDI, which can then be
written either to a file opened in binary mode (mode='wb'),
or to stdout by means of:   sys.stdout.buffer.write()

my_opus = [
    96, # ticks
    [   # track 0: # track 0
        ['patch_change', 0, 1, 8],   # and these are the events...
        ['note_on',   5, 1, 25, 96],
        ['note_off', 96, 1, 25, 0],
        ['note_on',   0, 1, 29, 96],
        ['note_off', 96, 1, 29, 0],
    ],   # end of track 0
]
my_midi = opus2midi(my_opus)
sys.stdout.buffer.write(my_midi)
'''
    if len(opus) < 2:
        opus=[1000, [],]
    tracks = copy.deepcopy(opus)
    ticks = int(tracks.pop(0))
    ntracks = len(tracks)
    if ntracks == 1:
        format = 0
    else:
        format = 1

    my_midi = b"MThd\x00\x00\x00\x06"+struct.pack('>HHH',format,ntracks,ticks)
    for track in tracks:
        events = _encode(track, text_encoding=text_encoding)
        my_midi += b'MTrk' + struct.pack('>I',len(events)) + events
    _clean_up_warnings()
    return my_midi


def score2opus(score=None, text_encoding='ISO-8859-1'):
    r'''
The argument is a list: the first item in the list is the "ticks"
parameter, the others are the tracks. Each track is a list
of score-events, and each event is itself a list.  A score-event
is similar to an opus-event (see above), except that in a score:
 1) the times are expressed as an absolute number of ticks
    from the track's start time
 2) the pairs of 'note_on' and 'note_off' events in an "opus"
    are abstracted into a single 'note' event in a "score":
    ['note', start_time, duration, channel, pitch, velocity]
score2opus() returns a list specifying the equivalent "opus".

my_score = [
    96,
    [   # track 0:
        ['patch_change', 0, 1, 8],
        ['note', 5, 96, 1, 25, 96],
        ['note', 101, 96, 1, 29, 96]
    ],   # end of track 0
]
my_opus = score2opus(my_score)
'''
    if len(score) < 2:
        score=[1000, [],]
    tracks = copy.deepcopy(score)
    ticks = int(tracks.pop(0))
    opus_tracks = []
    for scoretrack in tracks:
        time2events = dict([])
        for scoreevent in scoretrack:
            if scoreevent[0] == 'note':
                note_on_event = ['note_on',scoreevent[1],
                 scoreevent[3],scoreevent[4],scoreevent[5]]
                note_off_event = ['note_off',scoreevent[1]+scoreevent[2],
                 scoreevent[3],scoreevent[4],scoreevent[5]]
                if time2events.get(note_on_event[1]):
                   time2events[note_on_event[1]].append(note_on_event)
                else:
                   time2events[note_on_event[1]] = [note_on_event,]
                if time2events.get(note_off_event[1]):
                   time2events[note_off_event[1]].append(note_off_event)
                else:
                   time2events[note_off_event[1]] = [note_off_event,]
                continue
            if time2events.get(scoreevent[1]):
               time2events[scoreevent[1]].append(scoreevent)
            else:
               time2events[scoreevent[1]] = [scoreevent,]

        sorted_times = []  # list of keys
        for k in time2events.keys():
            sorted_times.append(k)
        sorted_times.sort()

        sorted_events = []  # once-flattened list of values sorted by key
        for time in sorted_times:
            sorted_events.extend(time2events[time])

        abs_time = 0
        for event in sorted_events:  # convert abs times => delta times
            delta_time = event[1] - abs_time
            abs_time = event[1]
            event[1] = delta_time
        opus_tracks.append(sorted_events)
    opus_tracks.insert(0,ticks)
    _clean_up_warnings()
    return opus_tracks

#--------------------------- Decoding stuff ------------------------

def midi2score(midi=b''):
    r'''
Translates MIDI into a "score", using midi2opus() then opus2score()
'''
    return opus2score(midi2opus(midi))

def midi2ms_score(midi=b''):
    r'''
Translates MIDI into a "score" with one beat per second and one
tick per millisecond, using midi2opus() then to_millisecs()
then opus2score()
'''
    return opus2score(to_millisecs(midi2opus(midi)))

def opus2score(opus=[]):
    r'''For a description of the "opus" and "score" formats,
see opus2midi() and score2opus().
'''
    if len(opus) < 2:
        _clean_up_warnings()
        return [1000,[],]
    tracks = copy.deepcopy(opus)  # couple of slices probably quicker...
    ticks = int(tracks.pop(0))
    score = [ticks,]
    for opus_track in tracks:
        ticks_so_far = 0
        score_track = []
        chapitch2note_on_events = dict([])   # 4.0
        for opus_event in opus_track:
            ticks_so_far += opus_event[1]
            if opus_event[0] == 'note_off' or (opus_event[0] == 'note_on' and opus_event[4] == 0):  # 4.8
                cha = opus_event[2]
                pitch = opus_event[3]
                key = cha*128 + pitch
                if chapitch2note_on_events.get(key):
                    new_event = chapitch2note_on_events[key].pop(0)
                    new_event[2] = ticks_so_far - new_event[1]
                    score_track.append(new_event)
                elif pitch > 127:
                    pass #_warn('opus2score: note_off with no note_on, bad pitch='+str(pitch))
                else:
                    pass #_warn('opus2score: note_off with no note_on cha='+str(cha)+' pitch='+str(pitch))
            elif opus_event[0] == 'note_on':
                cha = opus_event[2]
                pitch = opus_event[3]
                key = cha*128 + pitch
                new_event = ['note',ticks_so_far,0,cha,pitch, opus_event[4]]
                if chapitch2note_on_events.get(key):
                    chapitch2note_on_events[key].append(new_event)
                else:
                    chapitch2note_on_events[key] = [new_event,]
            else:
                opus_event[1] = ticks_so_far
                score_track.append(opus_event)
        # check for unterminated notes (Oisín) -- 5.2
        for chapitch in chapitch2note_on_events:
            note_on_events = chapitch2note_on_events[chapitch]
            for new_e in note_on_events:
                new_e[2] = ticks_so_far - new_e[1]
                score_track.append(new_e)
                pass #_warn("opus2score: note_on with no note_off cha="+str(new_e[3])+' pitch='+str(new_e[4])+'; adding note_off at end')
        score.append(score_track)
    _clean_up_warnings()
    return score


def midi2opus(midi=b''):
    r'''Translates MIDI into a "opus".  For a description of the
"opus" format, see opus2midi()
'''
    my_midi=bytearray(midi)
    if len(my_midi) < 4:
        _clean_up_warnings()
        return [1000,[],]
    id = bytes(my_midi[0:4])
    if id != b'MThd':
        _warn("midi2opus: midi starts with "+str(id)+" instead of 'MThd'")
        _clean_up_warnings()
        return [1000,[],]
    [length, format, tracks_expected, ticks] = struct.unpack(
     '>IHHH', bytes(my_midi[4:14]))
    if length != 6:
        _warn("midi2opus: midi header length was "+str(length)+" instead of 6")
        _clean_up_warnings()
        return [1000,[],]
    my_opus = [ticks,]
    my_midi = my_midi[14:]
    track_num = 1   # 5.1
    while len(my_midi) >= 8:
        track_type   = bytes(my_midi[0:4])
        if track_type != b'MTrk':
            #_warn('midi2opus: Warning: track #'+str(track_num)+' type is '+str(track_type)+" instead of b'MTrk'")
            pass
        [track_length] = struct.unpack('>I', my_midi[4:8])
        my_midi = my_midi[8:]
        if track_length > len(my_midi):
            _warn('midi2opus: track #'+str(track_num)+' length '+str(track_length)+' is too large')
            _clean_up_warnings()
            return my_opus   # 5.0
        my_midi_track = my_midi[0:track_length]
        my_track = _decode(my_midi_track)
        my_opus.append(my_track)
        my_midi = my_midi[track_length:]
        track_num += 1   # 5.1
    _clean_up_warnings()
    return my_opus


###########################################################################

def _decode(trackdata=b'', exclude=None, include=None,
 event_callback=None, exclusive_event_callback=None, no_eot_magic=False):
    r'''Decodes MIDI track data into an opus-style list of events.
The options:
  'exclude' is a list of event types which will be ignored SHOULD BE A SET
  'include' (and no exclude), makes exclude a list
       of all possible events, /minus/ what include specifies
  'event_callback' is a coderef
  'exclusive_event_callback' is a coderef
'''
    trackdata = bytearray(trackdata)
    if exclude == None:
        exclude = []
    if include == None:
        include = []
    if include and not exclude:
        exclude = All_events
    include = set(include)
    exclude = set(exclude)

    # Pointer = 0;  not used here; we eat through the bytearray instead.
    event_code = -1; # used for running status
    event_count = 0;
    events = []

    while(len(trackdata)):
        # loop while there's anything to analyze ...
        eot = False   # When True, the event registrar aborts this loop
        event_count += 1

        E = []
        # E for events - we'll feed it to the event registrar at the end.

        # Slice off the delta time code, and analyze it
        [time, remainder] = _unshift_ber_int(trackdata)

        # Now let's see what we can make of the command
        first_byte = trackdata.pop(0) & 0xFF

        if (first_byte < 0xF0):  # It's a MIDI event
            if (first_byte & 0x80):
                event_code = first_byte
            else:
                # It wants running status; use last event_code value
                trackdata.insert(0, first_byte)
                if (event_code == -1):
                    _warn("Running status not set; Aborting track.")
                    return []

            command = event_code & 0xF0
            channel = event_code & 0x0F

            if (command == 0xF6):  #  0-byte argument
                pass
            elif (command == 0xC0 or command == 0xD0):  #  1-byte argument
                parameter = trackdata.pop(0)  # could be B
            else: # 2-byte argument could be BB or 14-bit
                parameter = (trackdata.pop(0), trackdata.pop(0))

            #################################################################
            # MIDI events

            if (command      == 0x80):
                if 'note_off' in exclude:
                    continue
                E = ['note_off', time, channel, parameter[0], parameter[1]]
            elif (command == 0x90):
                if 'note_on' in exclude:
                    continue
                E = ['note_on', time, channel, parameter[0], parameter[1]]
            elif (command == 0xA0):
                if 'key_after_touch' in exclude:
                    continue
                E = ['key_after_touch',time,channel,parameter[0],parameter[1]]
            elif (command == 0xB0):
                if 'control_change' in exclude:
                    continue
                E = ['control_change',time,channel,parameter[0],parameter[1]]
            elif (command == 0xC0):
                if 'patch_change' in exclude:
                    continue
                E = ['patch_change', time, channel, parameter]
            elif (command == 0xD0):
                if 'channel_after_touch' in exclude:
                    continue
                E = ['channel_after_touch', time, channel, parameter]
            elif (command == 0xE0):
                if 'pitch_wheel_change' in exclude:
                    continue
                E = ['pitch_wheel_change', time, channel,
                 _read_14_bit(parameter)-0x2000]
            else:
                _warn("Shouldn't get here; command="+hex(command))

        elif (first_byte == 0xFF):  # It's a Meta-Event! ##################
            #[command, length, remainder] =
            #    unpack("xCwa*", substr(trackdata, $Pointer, 6));
            #Pointer += 6 - len(remainder);
            #    # Move past JUST the length-encoded.
            command = trackdata.pop(0) & 0xFF
            [length, trackdata] = _unshift_ber_int(trackdata)
            if (command      == 0x00):
                 if (length == 2):
                     E = ['set_sequence_number',time,_twobytes2int(trackdata)]
                 else:
                     _warn('set_sequence_number: length must be 2, not '+str(length))
                     E = ['set_sequence_number', time, 0]

            elif command >= 0x01 and command <= 0x0f:   # Text events
                # 6.2 take it in bytes; let the user get the right encoding.
                # text_str = trackdata[0:length].decode('ascii','ignore')
                # text_str = trackdata[0:length].decode('ISO-8859-1')
                # 6.4 take it in bytes; let the user get the right encoding.
                text_data = bytes(trackdata[0:length])   # 6.4
                # Defined text events
                if (command == 0x01):
                     E = ['text_event', time, text_data]
                elif (command == 0x02):
                     E = ['copyright_text_event', time, text_data]
                elif (command == 0x03):
                     E = ['track_name', time, text_data]
                elif (command == 0x04):
                     E = ['instrument_name', time, text_data]
                elif (command == 0x05):
                     E = ['lyric', time, text_data]
                elif (command == 0x06):
                     E = ['marker', time, text_data]
                elif (command == 0x07):
                     E = ['cue_point', time, text_data]
                # Reserved but apparently unassigned text events
                elif (command == 0x08):
                     E = ['text_event_08', time, text_data]
                elif (command == 0x09):
                     E = ['text_event_09', time, text_data]
                elif (command == 0x0a):
                     E = ['text_event_0a', time, text_data]
                elif (command == 0x0b):
                     E = ['text_event_0b', time, text_data]
                elif (command == 0x0c):
                     E = ['text_event_0c', time, text_data]
                elif (command == 0x0d):
                     E = ['text_event_0d', time, text_data]
                elif (command == 0x0e):
                     E = ['text_event_0e', time, text_data]
                elif (command == 0x0f):
                     E = ['text_event_0f', time, text_data]

            # Now the sticky events -------------------------------------
            elif (command == 0x2F):
                 E = ['end_track', time]
                     # The code for handling this, oddly, comes LATER,
                     # in the event registrar.
            elif (command == 0x51): # DTime, Microseconds/Crochet
                 if length != 3:
                     _warn('set_tempo event, but length='+str(length))
                 E = ['set_tempo', time,
                      struct.unpack(">I", b'\x00'+trackdata[0:3])[0]]
            elif (command == 0x54):
                 if length != 5:   # DTime, HR, MN, SE, FR, FF
                     _warn('smpte_offset event, but length='+str(length))
                 E = ['smpte_offset',time] + list(struct.unpack(">BBBBB",trackdata[0:5]))
            elif (command == 0x58):
                 if length != 4:   # DTime, NN, DD, CC, BB
                     _warn('time_signature event, but length='+str(length))
                 E = ['time_signature', time]+list(trackdata[0:4])
            elif (command == 0x59):
                 if length != 2:   # DTime, SF(signed), MI
                     _warn('key_signature event, but length='+str(length))
                 E = ['key_signature',time] + list(struct.unpack(">bB",trackdata[0:2]))
            elif (command == 0x7F):   # 6.4
                 E = ['sequencer_specific',time, bytes(trackdata[0:length])]
            else:
                 E = ['raw_meta_event', time, command,
                   bytes(trackdata[0:length])]   # 6.0
                 #"[uninterpretable meta-event command of length length]"
                 # DTime, Command, Binary Data
                 # It's uninterpretable; record it as raw_data.

            # Pointer += length; #  Now move Pointer
            trackdata = trackdata[length:]

        ######################################################################
        elif (first_byte == 0xF0 or first_byte == 0xF7):
            # Note that sysexes in MIDI /files/ are different than sysexes
            # in MIDI transmissions!! The vast majority of system exclusive
            # messages will just use the F0 format. For instance, the
            # transmitted message F0 43 12 00 07 F7 would be stored in a
            # MIDI file as F0 05 43 12 00 07 F7. As mentioned above, it is
            # required to include the F7 at the end so that the reader of the
            # MIDI file knows that it has read the entire message. (But the F7
            # is omitted if this is a non-final block in a multiblock sysex;
            # but the F7 (if there) is counted in the message's declared
            # length, so we don't have to think about it anyway.)
            #command = trackdata.pop(0)
            [length, trackdata] = _unshift_ber_int(trackdata)
            if first_byte == 0xF0:
                # 20091008 added ISO-8859-1 to get an 8-bit str
                # 6.4 return bytes instead
                E = ['sysex_f0', time, bytes(trackdata[0:length])]
            else:
                E = ['sysex_f7', time, bytes(trackdata[0:length])]
            trackdata = trackdata[length:]

        ######################################################################
        # Now, the MIDI file spec says:
        #  <track data> = <MTrk event>+
        #  <MTrk event> = <delta-time> <event>
        #  <event> = <MIDI event> | <sysex event> | <meta-event>
        # I know that, on the wire, <MIDI event> can include note_on,
        # note_off, and all the other 8x to Ex events, AND Fx events
        # other than F0, F7, and FF -- namely, <song position msg>,
        # <song select msg>, and <tune request>.
        #
        # Whether these can occur in MIDI files is not clear specified
        # from the MIDI file spec.  So, I'm going to assume that
        # they CAN, in practice, occur.  I don't know whether it's
        # proper for you to actually emit these into a MIDI file.
        
        elif (first_byte == 0xF2):   # DTime, Beats
            #  <song position msg> ::=     F2 <data pair>
            E = ['song_position', time, _read_14_bit(trackdata[:2])]
            trackdata = trackdata[2:]

        elif (first_byte == 0xF3):   # <song select msg> ::= F3 <data singlet>
            # E = ['song_select', time, struct.unpack('>B',trackdata.pop(0))[0]]
            E = ['song_select', time, trackdata[0]]
            trackdata = trackdata[1:]
            # DTime, Thing (what?! song number?  whatever ...)

        elif (first_byte == 0xF6):   # DTime
            E = ['tune_request', time]
            # What would a tune request be doing in a MIDI /file/?

        #########################################################
        # ADD MORE META-EVENTS HERE.  TODO:
        # f1 -- MTC Quarter Frame Message. One data byte follows
        #     the Status; it's the time code value, from 0 to 127.
        # f8 -- MIDI clock.    no data.
        # fa -- MIDI start.    no data.
        # fb -- MIDI continue. no data.
        # fc -- MIDI stop.     no data.
        # fe -- Active sense.  no data.
        # f4 f5 f9 fd -- unallocated

            r'''
        elif (first_byte > 0xF0) { # Some unknown kinda F-series event ####
            # Here we only produce a one-byte piece of raw data.
            # But the encoder for 'raw_data' accepts any length of it.
            E = [ 'raw_data',
                         time, substr(trackdata,Pointer,1) ]
            # DTime and the Data (in this case, the one Event-byte)
            ++Pointer;  # itself

'''
        elif first_byte > 0xF0:  # Some unknown F-series event
            # Here we only produce a one-byte piece of raw data.
            # E = ['raw_data', time, bytest(trackdata[0])]   # 6.4
            E = ['raw_data', time, trackdata[0]]   # 6.4 6.7
            trackdata = trackdata[1:]
        else:  # Fallthru.
            _warn("Aborting track.  Command-byte first_byte="+hex(first_byte))
            break
        # End of the big if-group


        ######################################################################
        #  THE EVENT REGISTRAR...
        if E and  (E[0] == 'end_track'):
            # This is the code for exceptional handling of the EOT event.
            eot = True
            if not no_eot_magic:
                if E[1] > 0:  # a null text-event to carry the delta-time
                    E = ['text_event', E[1], '']
                else:
                    E = []   # EOT with a delta-time of 0; ignore it.
        
        if E and not (E[0] in exclude):
            #if ( $exclusive_event_callback ):
            #    &{ $exclusive_event_callback }( @E );
            #else:
            #    &{ $event_callback }( @E ) if $event_callback;
                events.append(E)
        if eot:
            break

    # End of the big "Event" while-block

    return events


###########################################################################
def _encode(events_lol, unknown_callback=None, never_add_eot=False,
  no_eot_magic=False, no_running_status=False, text_encoding='ISO-8859-1'):
    # encode an event structure, presumably for writing to a file
    # Calling format:
    #   $data_r = MIDI::Event::encode( \@event_lol, { options } );
    # Takes a REFERENCE to an event structure (a LoL)
    # Returns an (unblessed) REFERENCE to track data.

    # If you want to use this to encode a /single/ event,
    # you still have to do it as a reference to an event structure (a LoL)
    # that just happens to have just one event.  I.e.,
    #   encode( [ $event ] ) or encode( [ [ 'note_on', 100, 5, 42, 64] ] )
    # If you're doing this, consider the never_add_eot track option, as in
    #   print MIDI ${ encode( [ $event], { 'never_add_eot' => 1} ) };

    data = [] # what I'll store the chunks of byte-data in

    # This is so my end_track magic won't corrupt the original
    events = copy.deepcopy(events_lol)

    if not never_add_eot:
        # One way or another, tack on an 'end_track'
        if events:
            last = events[-1]
            if not (last[0] == 'end_track'):  # no end_track already
                if (last[0] == 'text_event' and len(last[2]) == 0):
                    # 0-length text event at track-end.
                    if no_eot_magic:
                        # Exceptional case: don't mess with track-final
                        # 0-length text_events; just peg on an end_track
                        events.append(['end_track', 0])
                    else:
                        # NORMAL CASE: replace with an end_track, leaving DTime
                        last[0] = 'end_track'
                else:
                    # last event was neither 0-length text_event nor end_track
                    events.append(['end_track', 0])
        else:  # an eventless track!
            events = [['end_track', 0],]

    # maybe_running_status = not no_running_status # unused? 4.7
    last_status = -1

    for event_r in (events):
        E = copy.deepcopy(event_r)
        # otherwise the shifting'd corrupt the original
        if not E:
            continue

        event = E.pop(0)
        if not len(event):
            continue

        dtime = int(E.pop(0))
        # print('event='+str(event)+' dtime='+str(dtime))

        event_data = ''

        if (   # MIDI events -- eligible for running status
             event    == 'note_on'
             or event == 'note_off'
             or event == 'control_change'
             or event == 'key_after_touch'
             or event == 'patch_change'
             or event == 'channel_after_touch'
             or event == 'pitch_wheel_change'  ):

            # This block is where we spend most of the time.  Gotta be tight.
            if (event == 'note_off'):
                status = 0x80 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>BB', int(E[1])&0x7F, int(E[2])&0x7F)
            elif (event == 'note_on'):
                status = 0x90 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>BB', int(E[1])&0x7F, int(E[2])&0x7F)
            elif (event == 'key_after_touch'):
                status = 0xA0 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>BB', int(E[1])&0x7F, int(E[2])&0x7F)
            elif (event == 'control_change'):
                status = 0xB0 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>BB', int(E[1])&0xFF, int(E[2])&0xFF)
            elif (event == 'patch_change'):
                status = 0xC0 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>B', int(E[1]) & 0xFF)
            elif (event == 'channel_after_touch'):
                status = 0xD0 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>B', int(E[1]) & 0xFF)
            elif (event == 'pitch_wheel_change'):
                status = 0xE0 | (int(E[0]) & 0x0F)
                parameters =  _write_14_bit(int(E[1]) + 0x2000)
            else:
                _warn("BADASS FREAKOUT ERROR 31415!")

            # And now the encoding
            # w = BER compressed integer (not ASN.1 BER, see perlpacktut for
            # details).  Its bytes represent an unsigned integer in base 128,
            # most significant digit first, with as few digits as possible.
            # Bit eight (the high bit) is set on each byte except the last.

            data.append(_ber_compressed_int(dtime))
            if (status != last_status) or no_running_status:
                data.append(struct.pack('>B', status))
            data.append(parameters)
 
            last_status = status
            continue
        else:
            # Not a MIDI event.
            # All the code in this block could be more efficient,
            # but this is not where the code needs to be tight.
            # print "zaz $event\n";
            last_status = -1

            if event == 'raw_meta_event':
                event_data = _some_text_event(int(E[0]), E[1], text_encoding)
            elif (event == 'set_sequence_number'):  # 3.9
                event_data = b'\xFF\x00\x02'+_int2twobytes(E[0])

            # Text meta-events...
            # a case for a dict, I think (pjb) ...
            elif (event == 'text_event'):
                event_data = _some_text_event(0x01, E[0], text_encoding)
            elif (event == 'copyright_text_event'):
                event_data = _some_text_event(0x02, E[0], text_encoding)
            elif (event == 'track_name'):
                event_data = _some_text_event(0x03, E[0], text_encoding)
            elif (event == 'instrument_name'):
                event_data = _some_text_event(0x04, E[0], text_encoding)
            elif (event == 'lyric'):
                event_data = _some_text_event(0x05, E[0], text_encoding)
            elif (event == 'marker'):
                event_data = _some_text_event(0x06, E[0], text_encoding)
            elif (event == 'cue_point'):
                event_data = _some_text_event(0x07, E[0], text_encoding)
            elif (event == 'text_event_08'):
                event_data = _some_text_event(0x08, E[0], text_encoding)
            elif (event == 'text_event_09'):
                event_data = _some_text_event(0x09, E[0], text_encoding)
            elif (event == 'text_event_0a'):
                event_data = _some_text_event(0x0A, E[0], text_encoding)
            elif (event == 'text_event_0b'):
                event_data = _some_text_event(0x0B, E[0], text_encoding)
            elif (event == 'text_event_0c'):
                event_data = _some_text_event(0x0C, E[0], text_encoding)
            elif (event == 'text_event_0d'):
                event_data = _some_text_event(0x0D, E[0], text_encoding)
            elif (event == 'text_event_0e'):
                event_data = _some_text_event(0x0E, E[0], text_encoding)
            elif (event == 'text_event_0f'):
                event_data = _some_text_event(0x0F, E[0], text_encoding)
            # End of text meta-events

            elif (event == 'end_track'):
                event_data = b"\xFF\x2F\x00"

            elif (event == 'set_tempo'):
                #event_data = struct.pack(">BBwa*", 0xFF, 0x51, 3,
                #              substr( struct.pack('>I', E[0]), 1, 3))
                event_data = b'\xFF\x51\x03'+struct.pack('>I',E[0])[1:]
            elif (event == 'smpte_offset'):
                # event_data = struct.pack(">BBwBBBBB", 0xFF, 0x54, 5, E[0:5] )
                event_data = struct.pack(">BBBbBBBB", 0xFF,0x54,0x05,E[0],E[1],E[2],E[3],E[4])
            elif (event == 'time_signature'):
                # event_data = struct.pack(">BBwBBBB",  0xFF, 0x58, 4, E[0:4] )
                event_data = struct.pack(">BBBbBBB", 0xFF, 0x58, 0x04, E[0],E[1],E[2],E[3])
            elif (event == 'key_signature'):
                event_data = struct.pack(">BBBbB", 0xFF, 0x59, 0x02, E[0],E[1])
            elif (event == 'sequencer_specific'):
                # event_data = struct.pack(">BBwa*", 0xFF,0x7F, len(E[0]), E[0])
                event_data = _some_text_event(0x7F, E[0], text_encoding)
            # End of Meta-events

            # Other Things...
            elif (event == 'sysex_f0'):
                 #event_data = struct.pack(">Bwa*", 0xF0, len(E[0]), E[0])
                 #B=bitstring w=BER-compressed-integer a=null-padded-ascii-str
                 event_data = bytearray(b'\xF0')+_ber_compressed_int(len(E[0]))+bytearray(E[0])
            elif (event == 'sysex_f7'):
                 #event_data = struct.pack(">Bwa*", 0xF7, len(E[0]), E[0])
                 event_data = bytearray(b'\xF7')+_ber_compressed_int(len(E[0]))+bytearray(E[0])

            elif (event == 'song_position'):
                 event_data = b"\xF2" + _write_14_bit( E[0] )
            elif (event == 'song_select'):
                 event_data = struct.pack('>BB', 0xF3, E[0] )
            elif (event == 'tune_request'):
                 event_data = b"\xF6"
            elif (event == 'raw_data'):
                _warn("_encode: raw_data event not supported")
                # event_data = E[0]
                continue
            # End of Other Stuff

            else:
                # The Big Fallthru
                if unknown_callback:
                    # push(@data, &{ $unknown_callback }( @$event_r ))
                    pass
                else:
                    _warn("Unknown event: "+str(event))
                    # To surpress complaint here, just set
                    #  'unknown_callback' => sub { return () }
                continue

            #print "Event $event encoded part 2\n"
            if str(type(event_data)).find("'str'") >= 0:
                event_data = bytearray(event_data.encode('Latin1', 'ignore'))
            if len(event_data): # how could $event_data be empty
                # data.append(struct.pack('>wa*', dtime, event_data))
                # print(' event_data='+str(event_data))
                data.append(_ber_compressed_int(dtime)+event_data)

    return b''.join(data)


################################################################
# The code below this line is full of frightening things, all to
# do with the actual encoding and decoding of binary MIDI data.

def _twobytes2int(byte_a):
    r'''decode a 16 bit quantity from two bytes,'''
    return (byte_a[1] | (byte_a[0] << 8))

def _int2twobytes(int_16bit):
    r'''encode a 16 bit quantity into two bytes,'''
    return bytes([(int_16bit>>8) & 0xFF, int_16bit & 0xFF])

def _read_14_bit(byte_a):
    r'''decode a 14 bit quantity from two bytes,'''
    return (byte_a[0] | (byte_a[1] << 7))

def _write_14_bit(int_14bit):
    r'''encode a 14 bit quantity into two bytes,'''
    return bytes([int_14bit & 0x7F, (int_14bit>>7) & 0x7F])

def _ber_compressed_int(integer):
    r'''BER compressed integer (not an ASN.1 BER, see perlpacktut for
details).  Its bytes represent an unsigned integer in base 128,
most significant digit first, with as few digits as possible.
Bit eight (the high bit) is set on each byte except the last.
'''
    ber = bytearray(b'')
    seven_bits = 0x7F & integer
    ber.insert(0, seven_bits)  # XXX surely should convert to a char ?
    integer >>= 7
    while integer > 0:
        seven_bits = 0x7F & integer
        ber.insert(0, 0x80|seven_bits)  # XXX surely should convert to a char ?
        integer >>= 7
    return ber

def _unshift_ber_int(ba):
    r'''Given a bytearray, returns a tuple of (the ber-integer at the
start, and the remainder of the bytearray).
'''
    if not len(ba):   # 6.7
        _warn('_unshift_ber_int: no integer found')
        return ((0, b""))
    byte = ba.pop(0)
    integer = 0
    while True:
        integer += (byte & 0x7F)
        if not (byte & 0x80):
            return ((integer, ba))
        if not len(ba):
            _warn('_unshift_ber_int: no end-of-integer found')
            return ((0, ba))
        byte = ba.pop(0)
        integer <<= 7

def _clean_up_warnings():  # 5.4
    # Call this before returning from any publicly callable function
    # whenever there's a possibility that a warning might have been printed
    # by the function, or by any private functions it might have called.
    global _previous_times
    global _previous_warning
    if _previous_times > 1:
        # E:1176, 0: invalid syntax (<string>, line 1176) (syntax-error) ???
        # print('  previous message repeated '+str(_previous_times)+' times', file=sys.stderr)
        # 6.7
        sys.stderr.write('  previous message repeated {0} times\n'.format(_previous_times))
    elif _previous_times > 0:
        sys.stderr.write('  previous message repeated\n')
    _previous_times = 0
    _previous_warning = ''

def _warn(s=''):
    global _previous_times
    global _previous_warning
    if s == _previous_warning:  # 5.4
        _previous_times = _previous_times + 1
    else:
        _clean_up_warnings()
        sys.stderr.write(str(s)+"\n")
        _previous_warning = s

def _some_text_event(which_kind=0x01, text=b'some_text', text_encoding='ISO-8859-1'):
    if str(type(text)).find("'str'") >= 0:   # 6.4 test for back-compatibility
        data = bytes(text, encoding=text_encoding)
    else:
        data = bytes(text)
    return b'\xFF'+bytes((which_kind,))+_ber_compressed_int(len(data))+data


#------------------------ Other Transformations ---------------------

def to_millisecs(old_opus=None, desired_time_in_ms=1):
    r'''Recallibrates all the times in an "opus" to use one beat
per second and one tick per millisecond.  This makes it
hard to retrieve any information about beats or barlines,
but it does make it easy to mix different scores together.
'''
    if old_opus == None:
        return [1000 * desired_time_in_ms,[],]
    try:
        old_tpq  = int(old_opus[0])
    except IndexError:   # 5.0
        _warn('to_millisecs: the opus '+str(type(old_opus))+' has no elements')
        return [1000 * desired_time_in_ms,[],]
    new_opus = [1000 * desired_time_in_ms,]
    # 6.7 first go through building a table of set_tempos by absolute-tick
    ticks2tempo = {}
    itrack = 1
    while itrack < len(old_opus):
        ticks_so_far = 0
        for old_event in old_opus[itrack]:
            if old_event[0] == 'note':
                raise TypeError('to_millisecs needs an opus, not a score')
            ticks_so_far += old_event[1]
            if old_event[0] == 'set_tempo':
                ticks2tempo[ticks_so_far] = old_event[2]
        itrack += 1
    # then get the sorted-array of their keys
    tempo_ticks = []  # list of keys
    for k in ticks2tempo.keys():
        tempo_ticks.append(k)
    tempo_ticks.sort()
    # then go through converting to millisec, testing if the next
    # set_tempo lies before the next track-event, and using it if so.
    itrack = 1
    while itrack < len(old_opus):
        ms_per_old_tick = 400 / old_tpq  # float: will round later 6.3
        i_tempo_ticks = 0
        ticks_so_far = 0
        ms_so_far = 0.0
        previous_ms_so_far = 0.0
        new_track = [['set_tempo',0,1000000 * desired_time_in_ms],]  # new "crochet" is 1 sec
        for old_event in old_opus[itrack]:
            # detect if ticks2tempo has something before this event
            # 20160702 if ticks2tempo is at the same time, leave it
            event_delta_ticks = old_event[1] * desired_time_in_ms
            if (i_tempo_ticks < len(tempo_ticks) and
              tempo_ticks[i_tempo_ticks] < (ticks_so_far + old_event[1]) * desired_time_in_ms):
                delta_ticks = tempo_ticks[i_tempo_ticks] - ticks_so_far
                ms_so_far += (ms_per_old_tick * delta_ticks * desired_time_in_ms)
                ticks_so_far = tempo_ticks[i_tempo_ticks]
                ms_per_old_tick = ticks2tempo[ticks_so_far] / (1000.0*old_tpq * desired_time_in_ms)
                i_tempo_ticks += 1
                event_delta_ticks -= delta_ticks
            new_event = copy.deepcopy(old_event)  # now handle the new event
            ms_so_far += (ms_per_old_tick * old_event[1] * desired_time_in_ms)
            new_event[1] = round(ms_so_far - previous_ms_so_far)
            if old_event[0] != 'set_tempo':
                previous_ms_so_far = ms_so_far
                new_track.append(new_event)
            ticks_so_far += event_delta_ticks
        new_opus.append(new_track)
        itrack += 1
    _clean_up_warnings()
    return new_opus


def _clean_up_warnings():  # 5.4
    # Call this before returning from any publicly callable function
    # whenever there's a possibility that a warning might have been printed
    # by the function, or by any private functions it might have called.
    global _previous_times
    global _previous_warning
    if _previous_times > 1:
        # E:1176, 0: invalid syntax (<string>, line 1176) (syntax-error) ???
        # print('  previous message repeated '+str(_previous_times)+' times', file=sys.stderr)
        # 6.7
        sys.stderr.write('  previous message repeated {0} times\n'.format(_previous_times))
    elif _previous_times > 0:
        sys.stderr.write('  previous message repeated\n')
    _previous_times = 0
    _previous_warning = ''

def _warn(s=''):
    global _previous_times
    global _previous_warning
    if s == _previous_warning:  # 5.4
        _previous_times = _previous_times + 1
    else:
        _clean_up_warnings()
        sys.stderr.write(str(s)+"\n")
        _previous_warning = s
