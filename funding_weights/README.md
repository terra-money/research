# Funding Weight Simulation

DApps on the Terra Platform are allocated Treasury funding based on their economic activity, which is reflected in their *funding weight*. We simulate the behavior of the funding weight equation under various scenarios with three objectives:
1. Gain better intuition about its behavior
2. Determine a sound value for λ at genesis
3. Serve as a basis for the Columbus implementation

## Funding Equation
We define the **spending multiplier** for dApp i at time t as follows:

<img src="https://github.com/terra-project/research/blob/master/assets/spending_multiplier.gif" width="350" height="75">

The spending multiplier computes the growth in transaction volume per dollar of recent funding. Note that we are enforcing a non-negative constraint: we don't want to explicitly penalize dApps that are in a recession.

Then the **funding weight** of dApp i at time t will be:

<img src="https://github.com/terra-project/research/blob/master/assets/funding_weight.gif" width="600" height="100">

It is simply a weighted average (weight λ) between the normalized transaction volume and spending multiplier of dApp i at time t. Note that, by construction, all weights sum to 1.

An important implementation detail: if at any period no spending multipliers are defined the second term is dropped, effectively making λ=1.

## Simulation Framework
We simulate funding weights in a multi-firm economy:

### Inputs
* Transaction Volume timeseries for each firm

### Parameters
* Lambda

### Outputs
* Spending Multiplier timeseries for each firm
* Funding Weight timeseries for each firm

