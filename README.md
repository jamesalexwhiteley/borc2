# borc2.0
Bayesian Optimization with Reliability Constraints

# BORC 2.0: Bayesian Optimization with Reliability Constraints

BORC 2.0 is a framework for optimization under uncertainty that combines Bayesian optimization with reliability constraints. Here's the key structure:

## Problem Formulation
- We have a stochastic objective function `f(x, ξ)` where:
  - `x` represents design variables or parameters we can control
  - `ξ` represents random variables that introduce uncertainty

## Methodology
1. **Gaussian Process Surrogate**
   - We build a GP surrogate model to approximate `f(x, ξ)`
   - This provides both predictions and uncertainty estimates
   - The surrogate enables efficient exploration of the design space

2. **Key Metrics**
   - Expected value `E[f(x, ξ)]`: Measures average performance
   - Constraint probability `P[f(x, ξ)<0]`: Ensures reliability requirements

3. **Acquisition Function (α)**
   - Guides the optimization process
   - Balances exploitation (using known good regions) with exploration (investigating uncertain areas)
   - Helps select next points for evaluation

## Visual Components
The optimization process can be visualized through four key plots:

1. **Analytic Function** (`analytic_f.png`)
   - Shows the true underlying function surface
   - Represents the actual optimization landscape

2. **Surrogate Approximation** (`analytic_fhat.png`)
   - Displays the GP surrogate's prediction of the function
   - Demonstrates how well the model approximates the true function

3. **Posterior Distribution** (`analytic_posterior.png`)
   - Shows the uncertainty in the GP predictions
   - Highlights areas of high and low confidence

4. **Acquisition Surface** (`analytic_acquisition.png`)
   - Visualizes the acquisition function values
   - Indicates promising regions for next evaluations
   - Helps understand the exploration-exploitation trade-off

This framework enables efficient optimization while maintaining reliability constraints, making it particularly useful for engineering design and other applications where uncertainty plays a crucial role.

<div align="center">
  <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; max-width: 800px;">
    <img src="figs/analytic_f.png" alt="BayesOpt 1" width="40%"/>
    <img src="figs/analytic_fhat.png" alt="BayesOpt 2" width="40%"/>
    <img src="figs/analytic_posterior.png" alt="BayesOpt 3" width="40%"/>
    <img src="figs/analytic_acquisition.png" alt="BayesOpt 4" width="40%"/>
  </div>
</div>