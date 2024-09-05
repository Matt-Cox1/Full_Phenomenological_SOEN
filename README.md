# Phenomenological Model of Superconducting Optoelectronic Networks

#### **Institution**: University College London

#### **Student**: Matthew Cox

#### **MSc**: Machine Learning



SOENs integrate superconducting electronics with optical communication, capitalising on the unique properties of each. Superconducting circuits, utilising Josephson junctions and Superconducting Quantum Interference Devices (SQUIDs), provide low-energy, high-speed analogue computational primitives. Optical components, including waveguides and superconducting single-photon detectors (SPDs), enable high-bandwidth, low-latency signal routing with minimal crosstalk. This combination allows SOENs to achieve potential processing speeds up to $250,000$ times faster than biological brains while maintaining low power consumption \citep{Shainline2021}.

In 2023, Shainline et al. introduced the `partial' phenomenological model of SOENs, which greatly simplified the simulation process \citep{Shainline2023}, achieving an accuracy of $1$ part in $10,000$ whilst increasing simulation speed by a similar magnitude. This model was refined into the `full' phenomenological model, which abstracts away the explicit handling of neuronal spikes, resulting in a continuous framework more amenable to gradient-based machine learning algorithms \citep{Shainline2024}. This newer model of SOENs will be the focus of this project.

The distinction between these models lies in their treatment of neuronal spikes. The partial phenomenological model deals with spikes explicitly as separate computations, leading to discontinuities in the simulation process. In contrast, the full phenomenological model approximates this process, handling spikes implicitly and maintaining continuity, a process discussed in more depth in section \ref{subsubsec:theory_p_model}. This continuity is needed for the application of many machine-learning techniques.

This study focuses on four primary objectives:

\begin{enumerate}
    \item Creation of a simulation environment for the full phenomenological model of SOENs using PyTorch, enabling efficient GPU-based computations and compatibility with machine learning frameworks.
    \item Application and comparison of two learning algorithms within the SOEN framework:
    \begin{enumerate}
        \item Backpropagation Through Time (BPTT), applied to train SOEN models on the MNIST dataset and the Two Moons problem.
        \item Equilibrium Propagation (EqProp), implemented to solve a simple random value prediction task.
    \end{enumerate}
    \item Investigation of SOEN dynamics, including an exploration of the effects of noise on SOENs.
    \item The derivation and evaluation of a SOEN-specific energy function.
\end{enumerate}

The thesis is organised into several chapters. Following this introduction, the theoretical foundations of SOENs and the learning algorithms applied to them are presented. In the methodology chapter, the implementation of the simulation environment, and the experimental procedures for applying BPTT and EqProp learning algorithms are detailed. Lastly, the results of this project are presented and analysed followed by a discussion of their implications and limitations.
