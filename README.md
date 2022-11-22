[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/zadorlab/sella)

# Sella

Sella is a utility for finding first order saddle points

An example script
```python
#!/usr/bin/env python3

from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT

from sella import Sella, Constraints

# Set up your system as an ASE atoms object
slab = fcc111('Cu', (5, 5, 6), vacuum=7.5)
add_adsorbate(slab, 'Cu', 2.0, 'bridge')

# Optionally, create and populate a Constraints object.
cons = Constraints(slab)
for atom in slab:
    if atom.position[2] < slab.cell[2, 2] / 2.:
        cons.fix_translation(atom.index)

# Set up your calculator
slab.calc = EMT()

# Set up a Sella Dynamics object
dyn = Sella(
    slab,
    constraints=cons,
    trajectory='test_emt.traj',
)

dyn.run(1e-3, 1000)
```

If you are using Sella or you wish to use Sella, let me know!

## Documentation

For more information on how to use Sella, please check the [wiki](https://github.com/zadorlab/sella/wiki).

## Support

If you need help using Sella, please visit our [gitter support channel](https://gitter.im/zadorlab/sella),
or open a GitHub issue.

## How to cite

If you use our code in publications, please cite the revelant work(s). (1) is recommended when Sella is used for solids or in heterogeneous catalysis, (3) is recommended for molecular systems.

1. Hermes, E., Sargsyan, K., Najm, H. N., Zádor, J.: Accelerated saddle point refinement through full exploitation of partial Hessian diagonalization. Journal of Chemical Theory and Computation, 2019 15 6536-6549. https://pubs.acs.org/doi/full/10.1021/acs.jctc.9b00869
2. Hermes, E. D., Sagsyan, K., Najm, H. N., Zádor, J.: A geodesic approach to internal coordinate optimization. The Journal of Chemical Physics, 2021 155 094105. https://aip.scitation.org/doi/10.1063/5.0060146
3. Hermes, E. D., Sagsyan, K., Najm, H. N., Zádor, J.: Sella, an open-source automation-friendly molecular saddle point optimizer. Journal of Chemical Theory and Computation, 2022 18 6974–6988. https://pubs.acs.org/doi/10.1021/acs.jctc.2c00395

## Acknowledgments

This work was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences, Chemical Sciences, Geosciences and Biosciences Division, as part of the Computational Chemistry Sciences Program.
