## Transformer version of the piano-genie

We introduce Transformer Genie, an intelligent controller that maps 12-button input to a full 88-key piano in real time.

How does it work?
We restrict ourselves to 1-to-1 mappings between button presses and notes, giving the user precise control over timing and degree of polyphony but not which notes are played. Even given that restriction, there are many possible mappings; for example, the 12 buttons could map to a fixed scale over a single octave. Instead of using such a fixed mapping, we learn a time-varying mapping using a discrete autoencoder architecture trained on a set of existing piano performances.
A transformer encoder maps a sequence of piano notes to a sequence of controller buttons. 
A transformer decoder then decodes these controller sequences back into piano performances. 
After training, the encoder is discarded and controller sequences are provided by user input.

Pitch contours in the piano performance closely mimic the contours of the button sequence. 
Such behavior is encouraged by a training loss term that penalizes the encoder for violating relative pitch ordering, e.g. if an ascending piano interval maps to a descending button interval. 

We wish to learn a mapping from sequences y ∈ [0, 12)n,
i.e. amateur performances of n presses on 12 buttons, to sequences x ∈ [0, 88)n
, i.e. professional performances on an 88-key piano. To preserve a one-to-one mapping between
buttons pressed and notes played, we assume that both y and x are monophonic sequences.
Given that we lack examples of y, we propose using the autoencoder framework on examples x. 
Specifically, we learn a deterministic mapping enc(x) : [0, 88)n → [0, 12)n, and a stochastic 
inverse mapping Pdec(x |enc(x)).
We use transformers for both the encoder and the decoder. Given a sequence of input piano notes,
the encoder outputs a real-valued scalar, forming a sequence encs(x) ∈ Rn (with valuesbetween -1 and 1). 
To discretize this into enc(x) we quantize it to k = 8 buckets equally spaced between -1 and 1, 
and use the straight-through estimator to bypass this non-differentiable operation in the backwards pass. 
We refer to this contribution as the integer-quantized autoencoder (IQAE); it is inspired by two papers from the image compression literature that also use autoencoders with discrete bottlenecks. 
Given the sequence of input piano notes and the sequence of button predicted by the encoder, except the last note and last button, and the information of the current dtime, velocity and the last button, the decoder predicts the current pitch.

We train this system end-to-end to
minimize:
L = Lrecons + Lmargin + Lcontour 
Lrecons = −Σ log Pdec(x |enc(x))
Lmargin = Σ max(|encs (x)| − 1, 0)^2
Lcontour = Σ max(1 − ∆x∆encs (x), 0)^2
Together, Lrecons and Lmargin constitute our proposed IQAE
objective. The Lrecons term minimizes reconstruction loss of the decoder (as is typical of autoencoders). To agree with our discretization strategy, the Lmargin term discourages the encoder from producing values outside of [−1, 1]. We also contribute a musically motivated regularization strategy which gives the model an awareness of melodic contour (Lcontour). By comparing the finite differences (musical intervals in semitones) of the input ∆x to the finite differences of the real-valued encoder output ∆encs (x), the Lcontour term encourages the encoder to produce “button contours” that match the shape of the input melodic contours.


