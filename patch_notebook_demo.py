import nbformat

def patch_notebook():
    nb_path = r"d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\Hand-wave\ModularedPresentation_Educational.ipynb"
    print(f"Patching notebook: {nb_path}")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    patched = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Fix the kinetic operator size in the Harmonic demo
            if "T_harm = kinetic_operator(len(x_v), dx_v)" in cell.source:
                print("Found buggy Harmonic demo cell. Fixing T_harm size...")
                cell.source = cell.source.replace(
                    "T_harm = kinetic_operator(len(x_v), dx_v)",
                    "T_harm = kinetic_operator(len(x_int_v), dx_v)"
                )
                patched = True

    if patched:
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Notebook patched successfully.")
    else:
        print("No buggy cells found (or already patched).")

if __name__ == "__main__":
    patch_notebook()
