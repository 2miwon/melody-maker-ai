import pretty_midi
from pychord import Chord
import subprocess

def create_midi_with_beat(chords_str, file_path, length = 4, instrument = 'Electric Guitar (jazz)'):
    chords = [Chord(key) for key in chords_str[:8]]
    midi_data = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program(instrument)
    piano = pretty_midi.Instrument(program=piano_program)
    idx = 0
    for n, chord in enumerate(chords):
        for note_name in chord.components_with_pitch(root_pitch=4):
            note_number = pretty_midi.note_name_to_number(note_name)
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=n * length, end=(n + 1) * length)
            piano.notes.append(note)
        idx += 1
    midi_data.instruments.append(piano)
    midi_data.write(f'{file_path}/melody.mid')
    command = f"fluidsynth -ni font.sf2 {file_path}/melody.mid -F {file_path}/melody.wav -r 32000"
    subprocess.run(command, shell=True, check=True)

