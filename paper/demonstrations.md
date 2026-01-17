## Alignment and Sensitivity

To satisfy a "first-class" reviewer, do not rely on one plot. Use a Triangulation argument:

- Alignment (Structural): "The latent space is geometrically organized around pedagogical axes..."
    - Defense: "...validated on stratified subsets to rule out frequency confounders."
- Sensitivity (Causal): "...and crucially, intervening along these axes produces the theoretically predicted behavioral changes..."
- Defense: "...proving the relationship is causal, not just correlational."
- Rank (Parsimony): "...using the minimum necessary degrees of freedom."

Conclusion: Use Intervention (Sensitivity) as your hammer. No reviewer can argue that a "coincidence" causes a mathematically precise, monotonic response to a specific vector perturbation.

## The Role of PCA 


### PCA-related Plots 


### PCA-related Metrics 


## The Role of Probing

Probing is the "Compass" that makes both Alignment and Sensitivity possible.

You cannot perform either the Structural or Causal tests without Probing. It is the fundamental tool that bridges the gap between the "Black Box" ($z$) and the "Theory" ($L_0$).

Here is exactly how Probing fits into the workflow:

1. Probing in Structural Alignment
You cannot measure Alignment without a Probe.

The Problem: PCA gives you the "Data Axis" (PC1), but you need something to compare it against. Where is "North" (Mastery) in this high-dimensional space?
The Probe's Role: The Probe finds "North". By training the linear probe, you derive the weight vector $\vec{W}_{probe}$ which points exactly in the direction of increasing Mastery.
The Calculation: $S_{align} = \text{Cosine}(PC1, \vec{W}_{probe})$.
Without Probing: You have PC1, but you don't know if it points to "Mastery," "Time," or "Luck." The Probe validates the semantic meaning of the axis.
2. Probing in Causal Sensitivity
You cannot perform a Causal Intervention without a Probe.

The Problem: To test sensitivity, you want to "artificially increase a student's Initial Mastery" inside the vector $z$. But which way do you push the vector?
The Probe's Role: The Probe defines the Direction of Intervention.
To simulate "Getting Smarter," you calculate: $z_{new} = z_{old} + \delta \cdot \vec{W}_{probe}$.
The Calculation: Causal Effect = $Output(z_{new}) - Output(z_{old})$.
Without Probing: You wouldn't know which dimension to perturb. You would be poking the model blindly.
Summary: The Logical Hierarchy
For your paper, structure the argument like this:

Method (Active Grounding): We train the model with Probing Loss to force the structure to exist.
Tool (Diagnostic Probing): We use post-hoc Probes to extract the "Theoretical Axes" ($\vec{W}{L0}, \vec{W}{T}$) from the latent space.
Proof 1 (Alignment): We show that these Theoretical Axes align perfectly with the Data's Principal Components.
Proof 2 (Sensitivity): We show that moving along these Theoretical Axes causes the mathematically predicted change in behavior.

## Steps

