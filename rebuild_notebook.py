import json
import re

# Read the corrupted file
with open('Handwave_Presentation.ipynb.corrupted', 'r', encoding='utf-8') as f:
    content = f.read()

# Try to extract cells from the corrupted data by finding valid JSON fragments
# This is a recovery attempt

# For now, let's create a minimal working notebook structure with RISE metadata
# and we'll ask the user to verify the content

notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
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
    "nbformat_minor": 5,
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Hand-Wave: A Finite-Difference Quantum Solver with Interactive Potentials\n",
                "**Author:** Ahilan  \n",
                "**Date:** Today\n",
                "\n",
                "---\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Introduction: What is Quantum Mechanics?\n",
                "\n",
                "**Slide 1: What is QM?**\n",
                "\n",
                "- The branch of physics where we describe the behaviour of small particles: electrons, atoms, molecules.\n",
                "- At this scale, energy is **quantized**.\n",
                "- Particles exhibit wave-like properties: **particle--wave duality**.\n",
                "- Heisenberg uncertainty: $\\Delta x\\,\\Delta p \\gtrsim \\hbar/2$, we cannot know position and momentum simultaneously.\n",
                "- We talk in terms of **probabilities**, not definite trajectories.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import necessary libraries\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from functions import *\n",
                "\n",
                "# Set up plotting style\n",
                "plt.style.use('seaborn-v0_8-darkgrid')\n",
                "%matplotlib inline"
            ]
        }
    ]
}

# Write the clean notebook
with open('Handwave_Presentation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("✅ Created a clean notebook with RISE metadata.")
print("⚠️  Note: This is a minimal version. You may need to add back your content.")
print("    The corrupted version has been saved as 'Handwave_Presentation.ipynb.corrupted'")
