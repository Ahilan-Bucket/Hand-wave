import nbformat
from nbformat.v4 import new_code_cell

def ensure_autoreload():
    nb_path = r"d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\Hand-wave\ModularedPresentation_Educational.ipynb"
    print(f"Adding autoreload magic to: {nb_path}")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check if autoreload exists
    has_autoreload = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and '%autoreload' in cell.source:
            has_autoreload = True
            break
    
    if not has_autoreload:
        # Insert autoreload cell at the beginning (after first markdown cell)
        insert_idx = 1  # After title
        autoreload_cell = new_code_cell("""# Auto-reload modules (so changes to functions.py take effect immediately)
%load_ext autoreload
%autoreload 2
""")
        nb.cells.insert(insert_idx, autoreload_cell)
        print("Added autoreload magic cell.")
    else:
        print("Autoreload already exists.")
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Done!")

if __name__ == "__main__":
    ensure_autoreload()
