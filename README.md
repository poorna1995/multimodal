#### why contrasive vision encoder

#### what is contransive leanirng - clip

# Layer Normalisation

Layer Normalisation helps keep the training procedure stable by preventing **vanishing** and **exploding gradients**.

When the output of a layer becomes too large or too small, the gradient magnitudes during backpropagation can also become very large or very small.

- If gradients become too **large**, they may **explode** as they propagate backward through earlier layers.
- If gradients become too **small**, they may **vanish**, preventing parameters from updating effectively.

Layer Normalisation keeps gradients stable by ensuring the outputs of each layer stay within a controlled range. It normalises the activations so they maintain consistent statistical properties (mean and variance), preventing their magnitudes from drifting.

As training proceeds, the inputs to each layer can shift — a phenomenon called **internal covariate shift** — which slows down convergence. Layer Normalisation mitigates this problem, improving both the **stability** and **efficiency** of neural network training.

## Main Idea

Layer Normalisation adjusts the output of a neural network layer so that it has  
**mean = 0** and **variance = 1** (before applying learned scaling and shifting).  
This results in faster and more stable convergence.
