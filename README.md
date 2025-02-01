## inference_generate_single 
non-interactive functions to test sequences with 
- simple accumulative buffer
- fixed-length circular buffer
- circular buffer that accumulates til full, then removes the first 4 tokens every step
- circular buffer that preserve the primer. Accumulates til full, then circulates, removing the 4 first tokens after the primer.

## inference_dtime 
interactive, the performer fixes dtime (via midi in), while dur, vel and pitch are inferenced

## inference_dtime_dur_vel 
interactive, the performer fixes dtime, dur, and vel (via midi in) and pitch is inferenced.
the performer can fix everything (inject in the context)



# start tensorboard:
   tensorboard --logdir=./rpr

### Original Version


