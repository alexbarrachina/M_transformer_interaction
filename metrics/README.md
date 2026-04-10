
# To calculate Frechet Music distance scores 

Environment
```bash
conda activate tgenie
```

# 1. Build the giantMIDI reference stats once
```bash
python compute_fmd.py --build-reference-stats
```

# 2. Score one MIDI against that saved distribution
```bash
python compute_fmd.py --sample-file ./samples/jazz.midi
```

