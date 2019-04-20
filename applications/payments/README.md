# Terra for e-commerce payments: Simulation Model

We analyze the behavior of Terra's e-commerce growth model: we return a portion of new money supply to Terra users by funding e-commerce discounts, with the goal of increasing demand and thus money supply, and so on, thus creating a virtuous growth cycle.

## Objectives
We have three core objectives:
1. To understand the sustainability of the growth model in various scenarios
2. To understand the effectiveness of the growth model in increasing Terra's adoption
3. To develop a basis for valuing Luna, the economy's reserve asset

## Model Setup
We model Terra's economy over a 10-year period at monthly granularity (120 time-steps). The variables of the system at a given point in time comprise its state. The state evolves at every time-step in response to some external input. We aim to make external dependencies as explicit as possible. To that end, we make a distinction between exogenous variables, those that are external to the model, and endogenous variables, those that are produced by it.

In more detail, the state is comprised of:

### Inputs
These are variables that we consider exogenous to the system, ie the simulation model has no control over them. We model them independently and aim to understand the behavior of the system as a function of them. They are the primary levers for stress exertion.
* **Genesis Cash from fundraise** (constant)
* **Terra Alliance GMV**: aggregate transaction volume that goes through Terra's e-commerce partners.
* **Base Volume Penetration**: What is our base (promotion-independent) volume penetration over time?
* **Discount landscape & responsiveness**: What is the competitive discount landscape? In light of this, how responsive are users to discounts?
* **Inclination to hold Terra**: 
    * How inclined are users to hold Terra relative to transacting with it? Concretely: what fraction of Terra's market cap that is held as a (temporary) store-of-value, rather than as an immediate means to transacting?
    * How long does an e-commerce company hold Terra, on average, before converting to fiat?

### Model Parameters
These are parameters that the model may adjust freely.
* **Terra transaction fee**
* **Minimum Fiat Reserve Ratio**: the minimum ratio of cash reserves to Terra's market cap.
* **E-commerce discount regime**: discounts offered to users over time

### Outputs
These are the variables that are produced by the model as a response to the inputs. For that reason we consider them endogenous.
* **Terra transaction volume**
* **Seigniorage and Money Supply**
* **Cash** (Foundation)
* **Fiat Reserve Ratio**
* **Luna transaction fee rewards**
* **Free Cash Flow**: the amount of cash generated during the period that is freely available to distribute (ie post all expenses and deposits to the Fiat Reserve) — may be negative

