# Entropy

## Training Algorithm

- We start by training the system with an L2 loss that penalizes the deviation of {Mi} (skill mastery levels) from the values given by the theoretical model (Rasch).
- With this, the system is trained to reach the configuration that produces, as a projection, the {Mi} configuration closest to the theoretical values.
- We take this configuration as the initial value. And the {Mi} values from this configuration as reference {Mi} values {Mi0}.
- Then we proceed to train the model with a Total Loss (LT) LT = f(L1, L3, lambda) where L3 harshly penalizes those configurations that lead to forbidden {Mi}, that is, those that deviate excessively from the reference values {Mi0}. These excessive deviations will be defined by a variable epsilon that measures the maximum allowed difference and is applicable to all Mi values.
- Penalizing harshly means attempting to prevent the system from continuing in the direction that has led to the deviation from {Mi0}. Therefore, we must find a way to define this "deterrent penalty".
- We need to define a function f that combines L1, which seeks to optimize the model's global AUC, with the deterrent penalty. The objective is to try to lead the system toward the configuration that produces the best predictions while always discarding those that lead to forbidden {Mi} states.
- Ultimately, what we seek is that model configuration that optimizes the global AUC while providing {Mi} values compatible with the theoretical values, for a deviation margin given by epsilon.
- Questions of interest: 1) given a specific AUC loss value that we consider acceptable, how much can we vary epsilon to keep the loss within that given range? or how does the AUC loss vary with epsilon values (AUC loss curve)? 2) how do the results vary with lambda (Pareto curve)?
## Loss Functions

### L1 - Predictive Loss

L1 is the loss function traditionally used to train the model so that it learns the configuration that produces the best AUC values. 

### L2 - Rasch Loss

We call L2 the Rasch Loss because it is related to the deviation of the {Mi} state with respect to the theoretical values obtained through the application of the Rasch model. 


### L3 - Deterrent Penalty

LT = lambda * L1 + (1 - lambda) * L3

lambda = 0 -> L3 dominates over L1, but L3 is defined such that it only penalizes when the deviation of {Mi} exceeds a given value. In many cases this will not occur and therefore L3 will have no effect. By giving a value of 0 to lambda, we achieve the harsh penalty we are looking for. The higher the value of lambda, the less harsh the deterrent penalty will be. 

### Interpretability Losses

L2 and L3 are loss functions related to interpretability. They work jointly: one (L2) guides the system toward configurations compatible with semantically consistent states, the other (L3) allows the system to move toward solutions that improve AUC but without deviating too much from those consistent initial states. 

---

## Why Entropy?

Training a model means decreasing its entropy in the sense that, as we explore possible configurations -and discard them- the degrees of freedom diminish; that is, fewer valid or acceptable configurations remain. There is a decrease in the system's potential.

Total Energy = Useful Energy (transformable into work, or into structure?) + Dispersion Energy (which cannot be transformed into work, will it produce noise or deviations from structure, order, what is predicted by theoretical models?).

Potential Energy is Useful Energy, before transformation into structure or value through work.

The greater the Entropy of a System, the greater its Potential Energy or Useful Energy.

The concept of Entropy is closely related to Potential Energy and is useful when we are talking about information systems, as is the case with a deep learning model, rather than physical systems.

A living system is one that transforms potential energy into structure. When that system ceases to be alive, that energy becomes potential energy again. In a non-living physical system, if there are no mechanisms that maintain structure, the system will tend to equalize with its environment. Gravity maintains structure; in that sense it can be considered similar to the systems that maintain structure in living systems.

In an informational system we speak of:
Structure + Noise = Potential Energy + Dispersion Energy = Entropy (possibilities of becoming structure) + Noise

Entropy = Potential Energy
Order = Entropy transformed into structure = Potential Energy transformed into Kinetic Energy

Therefore, Entropy is not the opposite of Order, but rather measures the number of possible Order configurations, before the system consolidates into one of them. Relation to the uncertainty principle and quantum mechanics.

The fact that when there is Order there is no Entropy is related to the fact of Consolidation of one of the possibilities more than to the idea that they are contrary/opposite concepts (analogy with the Collapse of the wave function in quantum mechanics).

### How does this apply to our case?

The {Mi}, at the beginning, have high entropy (compatible possible configurations). As we train the system, the degrees of freedom diminish.

In what sense?
- At some point during the training process, we will move in an increasingly smaller space of possible configurations, seeking convergence toward a minimum (or maximum). This is optimal if it is a global minimum.
- Converging toward a local minimum has the disadvantage that we are losing the opportunity to find a global minimum.
- The training process achieves convergence toward a local minimum but has the disadvantage that we stop the search that could result in finding some global minimum.
- This type of training process entails a decrease in degrees of freedom.
- Another option could be not to stop training when a local minimum is found. To do this, we could save that configuration and continue exploring in search of a better local minimum.
- To achieve this, we need to get the system out of the "valley" to which the convergence process has led it. We could, for example, use a very large learning rate to make a kind of jump.
- When do we stop the exploration?
- Relationship between training/convergence and entropy: convergence = decrease in degrees of freedom.
- The higher the entropy of {Mi}, the better; this means there are more configurations among which to search for the optimal one.
- How can we use the concept of entropy? ... Entropy cannot be injected ... Can it be measured? ... Can it be used to guide training?
## Definition

Entropy is to information systems what Potential Energy is to physical systems. 

