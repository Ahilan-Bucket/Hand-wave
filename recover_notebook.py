import json
import re

print("Step 1: Reading corrupted file...")
with open('Handwave_Presentation.ipynb.corrupted', 'r', encoding='utf-8') as f:
    data = f.read()

print(f"File size: {len(data)} characters")

print("Step 2: Removing // comments...")
# Remove //... comments
clean = re.sub(r'//[^\n]*', '', data)

print("Step 3: Fixing JSON structure...")
# Fix the metadata issue - there's a misplaced closing brace
# The pattern is: "rise": {...}, "source": ...
# Should be: "rise": {...}}, "source": ...
clean = re.sub(r'(\"rise\":\s*\{[^}]+\})\s*,\s*(\"source\")', r'\1}, \2', clean)

print("Step 4: Attempting to parse JSON...")
try:
    nb = json.loads(clean)
    print(f"✅ SUCCESS! Loaded {len(nb['cells'])} cells")
   
    # Remove per-cell rise metadata
    for cell in nb.get('cells', []):
        if isinstance(cell.get('metadata'), dict) and 'rise' in cell['metadata']:
            del cell['metadata']['rise']
    
    # Add global RISE metadata
    nb.setdefault('metadata', {})
    nb['metadata']['rise'] = {
        "autolaunch": False,
        "start_slideshow_at": "selected",
        "transition": "fade",
        "width": 960,
        "height": 720
    }
    
    # Write fixed notebook
    with open('Handwave_Presentation.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    
    print("✅ Fixed notebook written to Handwave_Presentation.ipynb")
    print(f"   - {len(nb['cells'])} cells recovered")
    print("   - RISE metadata added")
    
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing failed: {e}")
    print(f"   Error at line {e.lineno}, column {e.colno}")
    print(f"   Context: {e.doc[max(0, e.pos-50):e.pos+50]}")
    
    # Write the "cleaned" version for inspection
    with open('debug_clean.json', 'w', encoding='utf-8') as  f:
        f.write(clean)
    print("   Wrote debug_clean.json for inspection")
