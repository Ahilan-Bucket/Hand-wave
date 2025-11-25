#!/usr/bin/env python3
"""
Manually rebuild the Handwave_Presentation notebook by extracting cells
from the corrupted file and constructing valid JSON.
"""

# Just hard-code the line ranges for each cell from what we saw in view_file
# This is the most reliable way to recover the data

import json

# Build the notebook structure manually
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.7"
        },
        "rise": {
            "autolaunch": False,
            "start_slideshow_at": "selected",
            "transition": "fade",
            "width": 960,
            "height": 720
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Cell 1: Title
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Hand-Wave: A Finite-Difference Quantum Solver with Interactive Potentials\n",
        "**Author:** Ahilan  \n",
        "**Date:** Today\n",
        "\n",
        "---\n",
        "\n",
        "## 1. Introduction: What is Quantum Mechanics?\n",
        "\n",
        "**Slide 1: What is QM?**\n",
        "\n",
        "- The branch of physics where we describe the behaviour of small particles: electrons, atoms, molecules.\n",
        "- At this scale, energy is **quantized**.\n",
        "- Particles exhibit wave-like properties: **particle--wave duality**.\n",
        "- Heisenberg uncertainty: $\\Delta x\\,\\Delta p \\gtrsim \\hbar/2$, we cannot know position and momentum simultaneously.\n",
        "- We talk in terms of **probabilities**, not definite trajectories.\n",
        "\n",
        "**Classically:** an object has definite position and velocity at all times.  \n",
        "**QM:** a particle is described by a **wave function** that spreads out in space; measurement picks out one specific outcome."
    ]
})

# Cell 2: Imports
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Import necessary libraries\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import sys\n",
        "import os\n",
        "\n",
        "# Ensure we can import from the local directory\n",
        "sys.path.append(os.getcwd())\n",
        "\n",
        "# Import our custom physics library\n",
        "from functions import *\n",
        "\n",
        "print(\"Libraries imported successfully!\")"
    ]
})

# Cell 3: Wave Functions
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Wave Functions and Probabilities\n",
        "\n",
        "**Slide 2:**\n",
        "\n",
        "- A wave function $\\Psi(x)$ encodes everything we can know about a 1D system.\n",
        "- $|\\Psi(x)|^2\\,dx$ is the probability of finding the particle between $x$ and $x+dx$.\n",
        "\n",
        "**Core conditions:**\n",
        "- $\\Psi(x)$ must be continuous (and usually differentiable).\n",
        "- **Normalization:**\n",
        "  $$\n",
        "  \\int_{-\\infty}^{\\infty} |\\Psi(x)|^2\\,dx = 1.\n",
        "  $$"
    ]
})

print("Building notebook structure...")

# Save
with open('Handwave_Presentation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("✅ Partial notebook created - YOU NEED TO ADD THE REST OF THE CELLS MANUALLY")
print("   This creates the first 3 cells. The original has many more.")
print("   BETTER APPROACH: Let me try a different recovery method...")
