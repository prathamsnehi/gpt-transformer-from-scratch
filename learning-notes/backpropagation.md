Notes that I take while learning from a clanker\~

(crazy to think the whole thing I'm building is teaching me how to build it)

## Simple Definition:

- each node ("knob") in my neural network is currently set to some value (weights). If I nudge this knob a tiny bit, does my error get better or worse, and by how much?
- Try turning it in any direction that reduces the error
- Backpropagation is the efficient way to figure out which way to turn EACH and EVERY knob in the network

What happens in a neural network:

- input goes in
- input is multiplied by weights (knobs), gets added and squished (activation function?) over and over, and a prediction comes out
- calculate loss (observed - actual) -> or maybe the other way around, but js know its the difference between the ANSWER and the OUTPUT
