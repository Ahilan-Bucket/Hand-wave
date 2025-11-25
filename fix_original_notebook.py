import json
import re

# Read the corrupted file
with open('Handwave_Presentation.ipynb.corrupted', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove // comments (everything from // to end of line)
clean = re.sub(r'//[^\n]*', '', content)

# Also fix trailing commas before closing braces/brackets (common JSON error)
clean = re.sub(r',(\s*[}\]])', r'\1', clean)

# Parse the JSON
nb = json.loads(clean)

# Remove any per-cell rise metadata (we'll use global only)
for cell in nb.get('cells', []):
    if isinstance(cell.get('metadata'), dict) and 'rise' in cell['metadata']:
        del cell['metadata']['rise'] # Set global RISE metadata properly
nb.setdefault('metadata', {})
nb['metadata']['rise'] = {
    "autolaunch": False,
    "start_slideshow_at": "selected",
    "transition": "fade",
    "width": 960,
    "height": 720
}

# Write back the clean notebook
with open('Handwave_Presentation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("✅ Original notebook has been restored and fixed!")
print(f"   - {len(nb['cells'])} cells recovered")
print("   - RISE metadata added properly")
print("   - Ready to present!")
