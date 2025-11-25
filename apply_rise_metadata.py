import json, re, os

# Path to the notebook (same folder as this script)
nb_path = os.path.join(os.path.dirname(__file__), "Handwave_Presentation.ipynb")

# Read raw notebook text and strip any // comments
with open(nb_path, "r", encoding="utf-8") as f:
    raw = f.read()

# Remove // comments (everything from // to end of line)
clean = re.sub(r"//.*", "", raw)

# Load JSON
nb = json.loads(clean)

# Set global RISE metadata
nb.setdefault("metadata", {})
nb["metadata"]["rise"] = {
    "slide_type": "slide",
    "transition": "fade",
    "autolaunch": False,
    "start_slideshow_at": "selected",
    "width": 960,
    "height": 720,
}

# Remove any per‑cell rise entries
for cell in nb.get("cells", []):
    if isinstance(cell.get("metadata"), dict) and "rise" in cell["metadata"]:
        del cell["metadata"]["rise"]

# Write back pretty‑printed JSON
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("✅ RISE metadata injected and per‑cell rise entries removed.")
