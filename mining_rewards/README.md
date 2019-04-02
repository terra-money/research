# Mining Rewards Mechanism
We implement, simulate and stress-test Terra's mining rewards mechanism.

## Mechanism Summary

**The main objective of the mechanism is to ensure that Mining Rewards per Luna (MRL) experiences stable growth over time.** MRL is important because it tells us how much I will be compensated over time if I stake 1 Luna today, which is the main determinant of Luna's price. We want Luna's price to be relatively stable, and so we target smoothly increasing MRL to achieve this.

### What do we mean by stable growth in MRL?
The simplest and most robust way of defining stability is relative to the recent past (a moving average). We also express growth relative to the recent past. So we define stable growth as follows: *MRL today should be its 1 year moving average x some growth factor.*

### How do we achieve stable growth in MRL?
MRL depends on two factors: total mining rewards, and the number of staked Luna. The mechanism attempts to control both to keep MRL on target:

* **Transaction fee (f)**: by increasing/decreasing f we can increase/decrease mining rewards to keep MRL on target.
* **Buyback weight (w)**: by increasing/decreasing w we can increase/decrease the rate of buybacks to keep the contribution of seigniorage towards stability in line with our target (Seigniorage Burden Target). This means that when fees increase to support MRL stability, buybacks step up accordingly and vice versa.

## Control Rules
The control rule is the logic that adjusts f and w in response to economic conditions. It is the core
building block of the mechanism. We have implemented three control rules to understand and compare their behavior:
- **null**: no control at all -- f and w remain fixed at their genesis values
- **debt**: control based on the amount of "Luna debt" accumulated -- the higher Luna Supply above its genesis value 
the higher f and w
- **smooth**: control that targets smooth MRL growth -- implemented in the Columbus mainnet

## Usage
The mining_rewards script simulates the full stability mechanism using the chosen control rule over a 10-year period. The Transaction Volume input is sampled from a cyclical Geometric Brownian Motion (GBM) with flexible parameters. It also offers the option of a partial (or full) fiat reserve to aid stability. 

See documentation in the script for detailed Inputs, Outputs, Parameters and State.

Some examples:
* To run simulation using smooth control rule:

```
python mining_rewards.py smooth
````

* To run simulation using null control rule and fiat ratio of 50%:

```
python mining_rewards.py null --fiat_ratio 0.5
```

* For detailed usage instructions:

```
python mining_rewards.py -h
```
