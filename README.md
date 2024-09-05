# Phenomenological Model of Superconducting Optoelectronic Networks

#### **Institution**: University College London

#### **Student**: Matthew Cox

#### **MSc**: Machine Learning

### Introduction to the project

Superconducting Optoelectronic Networks (SOENs) integrate superconducting electronics with optical communication, capitalising on the unique properties of each. Superconducting circuits, utilising Josephson junctions and Superconducting Quantum Interference Devices (SQUIDs), provide low-energy, high-speed analogue computational primitives. Optical components, including waveguides and superconducting single-photon detectors (SPDs), enable high-bandwidth, low-latency signal routing with minimal crosstalk. This combination allows SOENs to achieve potential processing speeds up to 250,000 times faster than biological brains while maintaining low power consumption.

In 2023, Shainline et al. introduced the 'partial' phenomenological model of SOENs, which greatly simplified the simulation process, achieving an accuracy of 1 part in 10,000 whilst increasing simulation speed by a similar magnitude. This model was refined into the 'full' phenomenological model, which abstracts away the explicit handling of neuronal spikes, resulting in a continuous framework more amenable to gradient-based machine learning algorithms. This newer model of SOENs will be the focus of this project.

The distinction between these models lies in their treatment of neuronal spikes. The partial phenomenological model deals with spikes explicitly as separate computations, leading to discontinuities in the simulation process. In contrast, the full phenomenological model approximates this process, handling spikes implicitly and maintaining continuity. This continuity is needed for the application of many machine-learning techniques.

This thesis focuses on four primary objectives:

- Creation of a simulation environment for the full phenomenological model of SOENs using PyTorch, enabling efficient GPU-based computations and compatibility with machine learning frameworks.
- Application and comparison of two learning algorithms within the SOEN framework:
  - Backpropagation Through Time (BPTT), applied to train SOEN models on the MNIST dataset and the Two Moons problem.
  - Equilibrium Propagation (EqProp), implemented to solve a simple random value prediction task.
- Investigation of SOEN dynamics, including an exploration of the effects of noise on SOENs.
- The derivation and evaluation of a SOEN-specific energy function.


### The Repo

The code uploaded to the repository is an accompaniment to the MSc report briefly introduced above. In this repository, most of the code necessary to perform the experiments mentioned in the report can be run, and the figures re-produced. There is also an introductory notebook called `Network_Creation_and_Dynamics.ipynb` that provides a tutorial for using this PyTorch implementation of the full phenomenological model of SOENs.
