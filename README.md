# Sella

Sella is a utility for finding first order saddle points

An example script
```python
#!/usr/bin/env python3

from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT

from sella import MinModeAtoms, optimize

# Set up your system as an ASE atoms object
slab = fcc111('Cu', (5, 5, 6), vacuum=7.5)
add_adsorbate(slab, 'Cu', 2.0, 'bridge')

# Define any constraints. Here we fix all atoms in the bottom half
# of the slab.
fix = [atom.index for atom in slab if atom.position[2] < slab.cell[2, 2] / 2.]

# Set up your calculator
calc = EMT()

# Create a Sella MinMode object
myminmode = MinModeAtoms(slab,  # Your Atoms object
                         calc,  # Your calculator
                         constraints=dict(fix=fix),  # Your constraints
                         trajectory='test_emt.traj',  # Optional trajectory
                         )

x1 = optimize(myminmode,    # Your MinMode object
              maxiter=500,  # Maximum number of force evaluations
              ftol=1e-3,    # Norm of the force vector, convergence threshold
              r_trust=5e-4, # Initial trust radius (Angstrom per d.o.f.)
              order=1,      # Order of saddle point to find (set to 0 for minimization)
              dxL=1e-4,     # Finite difference displacement magnitude (Angstrom)
              maxres=0.1,   # Maximum residual for eigensolver convergence (should be <= 1)
              )
```

Additional documentation forthcoming. All interfaces likely to change.

If you are using Sella or you wish to use Sella, let me know!
