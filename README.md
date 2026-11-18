# Loss of Plasticity in Deep RL: Diagnostics and Feature Decorrelation

A minimal experimental project studying representation collapse in deep reinforcement learning, and testing continuous feature decorrelation (`CovReg`) as a non-destructive alternative to discrete neuron resets.

---

## What is this project about?

When deep neural networks are trained with reinforcement learning over long horizons, they often lose the ability to learn new things. This is called **loss of plasticity**.

Even though the network has capacity (e.g. 128 hidden units), two things happen under continual gradient updates:
1. **Dormant neurons:** Some units stop activating and receive zero gradients.
2. **Feature collapse:** Active units become collinear (they all point in the same direction). In practice, effective feature rank drops from over 40 down to under 15, leaving the network functionally rigid.

Recent papers from DeepMind and collaborators tackle this using discrete interventions:
- **ReDo (Sokar et al., NeurIPS 2023):** Finds dead units and reinitializes them.
- **Continual Backprop (Dohare et al., Nature 2024):** Tracks running utility per weight and replaces low-utility weights.
- **Layer Resets (Nikishin et al., ICML 2022):** Periodically resets entire network layers.

These methods work, but they are discrete, complex, and cause sudden policy drops when weights get wiped.

---

## The idea: Continuous Feature Decorrelation (`CovReg`)

Instead of periodically resetting weights and wiping optimizer momentum buffers, we add a continuous penalty directly to the TD loss:

$$\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{TD}} + \\lambda \\cdot \\mathcal{L}_{\\text{decorr}}$$

Where $\\mathcal{L}_{\\text{decorr}}$ is the mean squared off-diagonal value of the feature correlation matrix across the batch:

$$\\mathcal{L}_{\\text{decorr}} = \\frac{1}{D(D-1)} \\sum_{i \\ne j} C_{ij}^2$$

If two neurons fire together with high correlation, they are penalized. This forces features to stay spread out across orthogonal directions during training.
