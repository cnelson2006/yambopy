

# `Xn_from_signal`: documentation (version 1.1)

#### Myrta Grüning, Claudio Attaccalite, Mike Pointeck, Anna Romani, Mao Yuncheng

This document describes the `shg_analysis` module, part of the `YamboPy` code, for converting the nonlinear susceptibilities extracted by `Xn_from_signal` into measurable second-harmonic generation (SHG) intensities for a 2D material in a layered structure (air / 2D material / dielectric film / substrate).
---

## 0. Minimal theoretical compendium 


The problem solved is algebraic:

$$ M_{kj} S_j = P_k,$$

where $P_k$ is the time-dependent polarization (or current) sampled on $N_t$ times $\{t_k\}$ which is output by the `yambo`/`lumen`code; the resulting $S_j$ is proportional to the susceptibility (conductivity) of nonlinear order $j$. The matrix of coefficients $M_{kj}$, of dimensions $N_t \times N_\text{nl}$ contains the time dependence to the applied electric field. So far three physical situations are implemented:
1. a single monochromatic electric field: ${\bf E}_0 \sin(\omega_0 t)$
2. two monochromatic electric fields: ${\bf E}_0 (\sin(\omega_1 t) + \sin(\omega_2 t))$
3. a pulse-shaped electric field: ${\bf E}(t) \sin(\omega_0 t)$. Here, it is assumed that the shape of the pulse ${\bf E}(t)$ varies slowly with respect to the period $2\pi/\omega_0$. So far, only a Gaussian pulse, ${\bf E}(t) = {\bf E}_0 \exp(-(t-t_0)^2/(2\sigma^2))/(\sqrt{2}\sigma)$ has been implemented. 

Four solvers are available:

1. the standard solver for full, well-determined matrix:  calls [`numpy.linalg.solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)
2. the least square solver, when $N_t \gg N_\text{nl}$ : calls  [`numpy.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq)
3. the single value decomposition, using the Moore-Penrose pseudoinverse,  when $N_t \gg N_\text{nl}$: calls [`numpy.linalg.pinv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv)
4. the least square solver with an initial guess, when $N_t \gg N_\text{nl}$ : calls  [`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)

From $S_j$ the susceptibilities (or conductivities) $\chi^{(n)}(-\omega_\sigma, \omega_1, \dots, \omega_n)$  are obtained using the following expression:

$$ S_j = C_0 K (-\omega_\sigma, \omega_1, \dots, \omega_n)\chi^{(n)}(-\omega_\sigma, \omega_1, \dots, \omega_n) $$

where $K(-\omega_\sigma; \omega_1, \dots, \omega_n)$ is a numerical factor that accounts for the intrinsic permutation symmetry depending on the nonlinear order and frequency arguments of $\chi$. $C_0$ is a further numerical factor depending on the applied electric field (field strength, normalisation factor, nonlinear order).

Details on the implementation can be found in the sources listed in the bibliography.

---

## 1. Code

### 1.1 Input

Three outputs of a `yambo_nl` run analysed with `Xn_from_sine` are required:

1. `o.YamboPy-X_probe_order_1`, `_2` — the $\chi^{(1)}$, $\chi^{(2)}$ spectra written by `output_analysis` (Gaussian units);
2. `SAVE/ns.db1` — the lattice database, read with `YamboLatticeDB`, providing the supercell height $L_z$;
3. `SAVE/ndb.Nonlinear` — the field database, providing the applied intensity $I$ (variable `Field_Intensity_1`, atomic units).

Substrate optical constants are taken from the [refractiveindex.info](https://refractiveindex.info) database via the `refractiveindex` python package, which auto-downloads the database on first use (configurable through the `REFRACTIVEINDEX_DB` environment variable).

### 1.2 Structure

The module is a single file, `yambopy/nl/shg_analysis.py`, organised in numbered sections:

1. **Unit conversions and yambo I/O** — `chi2_supercell_to_sheet_SI`, `sheet_to_bulk_chi2`, `nk_from_chi1_supercell`, `intensity_au_to_SI`, and the loaders `load_chi_order`, `supercell_height_SI`, `field_intensity_SI`;
2. **Database access** — keyword search and record loading over the refractiveindex.info catalog (`search_database`, `print_search`, `load_material`, `get_n`, `get_k`, `get_epsilon`, `database_version`);
3. **Material objects** — `Substrate` (database optical constants) and `SimulatedMaterial` (effective constants from the run's own $\chi^{(1)}$). Both expose the same interface — `n(E)`, `k(E)`, `complex_index(E)`, `epsilon(E)`, `wl_range_eV()`, `covers(E)` — so the models accept them interchangeably (duck typing);
4. **SHG intensity models** — `Stack` (structure-factor model, general case), `WoodwardModel` (single-interface strict SI), `ClarkModel` (bulk reference formula, Eq. (4) of [4]).

Energies outside a layer's data range at either $\omega$ or $2\omega$ yield `NaN` rather than extrapolated values.

### 1.3 Abstract class diagram 

Attributes, constructor, abstract and concrete methods included in `Xn_from_signal`. 


```mermaid
classDiagram
    class Xn_from_signal {
        <<abstract>>
        %% --- Attributes ---
        +time : np.ndarray
        +T_step : float
        +T_deph : float
        +efield : dict
        +pumps : list
        +efields : list
        +n_runs : int
        +polarization : np.ndarray
        +current : np.ndarray
        +l_eval_current : bool
        +l_out_current : bool
        +X_order : int
        +solver : str
        +samp_mod : str
        +nsamp : int
        +T_urange : list[float]
        +freqs : np.ndarray
        +prefix : str
        +out_dim : int
        +tol : float

        %% --- Constructor ---
        +__init__(nldb, X_order=4, T_range=[-1,-1], l_out_current=False, nsamp=-1, samp_mod='', solver='', tol=1e-10)

        %% --- Special ---
        +__str__() str

        %% --- Abstract hooks ---
        #set_defaults()
        #set_sampling(ifrq)
        #define_matrix(samp, ifrq)
        #update_time_range()
        #get_Unit_of_Measure(i_order)   
        #output_analysis(out, to_file)
        #reconstruct_signal(out, to_file)

        %% --- Concrete methods ---
        +solve_lin_system(mat, samp, init=None) np.ndarray
        +perform_analysis() np.ndarray
        +get_Unit_of_Measure(i_order) float
        +get_sampling(range, idir, ifrq)
    }

    %% Auxiliary (module-level) functions for completeness
    class AuxMath {
        +IsSquare(m) bool
        +IsWellConditioned(m) bool
        +residuals_func(x, M, S_i) float
    }
```

### 1.4 Subclasses diagram

The subclasses inherit the attributes and implement the abstract methods from `Xn_from_signal`. In addition: 

* `Xn_from_freqmix` has the `pump_freq` attribute, which is the frequency of the second electric field, and the `spike_correction` method which performs again the analysis for data points where the simple least square algebraic solution failed, using least square optimization starting from the averaged solution of the neighbouring data points.
* `Xn_from_pulse`has the `T0` attribute, centre of the Gaussian, and the `build_map` and `divide_by_factor`. The former maps the nonlinearity order $n$ and the number of negative frequencies $m$ into a single index. The latter is a modification of `Divide_by_Field` and is in this class until a proper generalisation of `Divide_by_Field` is available.        

```mermaid
classDiagram
note "Attributes and methods of Xn_from_signal in previous diagram"
Xn_from_signal <|-- Xn_from_sine
Xn_from_signal <|-- Xn_from_freqmix
Xn_from_signal <|-- Xn_from_pulse
class Xn_from_pulse{
+ T0 : float
+ build_map()
+ divide_by_factor()
}
class Xn_from_freqmix{
+ pump_freq : float
+ spike_correction()
}
class Xn_from_sine
```
---
## 2. How to use

The diagram below illustrates the general use of the code.

Once the appropriate `ndb.Nonlinear` database has been created with `yambo_nl`, the  `YamboNLDB` class is used to read the database and create the object containing all information. 

Depending on the external field used in `yambo_nl`, a different subclass is instantiated:

* `Xn_from_sine`: a single monochromatic electric field 
* `Xn_from_freqmix`: two monochromatic electric fields
* `Xn_from_pulse`: a pulse-shaped electric field

All subclasses implement the `perform_analysis()` method (setting up and solving the algebraic problem in the [Theory section](## 0. Theory)). The `output_analysis` writes (by default) the susceptibilities (conductivities) on files. For checking the goodness of the analysis, one may output the reconstructed signal (to be compared with the input signal) with `reconstruct_signal`.  


```mermaid

graph LR
    F[(ndb.Nonlinear)]--> A(Read database with YamboNLDB) -->|NLDB| B1(Xn_from_sine) -->|SIG| C(perform_analysis) -->|OUT| D(output_analysis) --> o.*@{shape: lean-l}
    C --> |OUT|E(reconstruct_signal)
    E --> o.*@{shape: lean-l}
    A -->|NLDB| B2(Xn_from_freqmix)
    A -->|NLDB| B3(Xn_from_pulse)
    B2 -->|SIG| C
    B3 -->|SIG| C

```



### Example 1: Monochromatic external field. Harmonic generation from polarization.

For one monochromatic external field, the `Xn_from_sine` class is instantiated. One can `print` the instance `SIG` to check the value of the class attributes read from `NLDB` and the defaults. Here, the default for `X_order` is overwritten and set to `5`. The output of `perform_analysis` is passed to `OUT`. This is passed as input to `output_analysis` that outputs ` o-DBs.YamboPy-X_probe_order_?`. Second harmonic is in `o-DBs.YamboPy-X_probe_order_2`.

```python
NLDB=YamboNLDB(calc='DBs')
SIG = Xn_from_sine(NLDB,X_order=5)
print(SIG)
OUT = SIG.perform_analysis()
SIG.output_analysis(OUT)
```
### Example 2: Monochromatic external field. Shift-current from current.

For one monochromatic external field, the `Xn_from_sine` class is instantiated. Here, the default for `l_out_current` is overwritten and set to `True`.  This means that the current, rather than the polarization is analysed.  Note that the `yambo_nl` run must output the current together with polarization. The output of `perform_analysis` is passed  to `output_analysis` that outputs ` o-DBs.YamboPy-Sigma_probe_order_?` (shift current is in the `o-DBs.YamboPy-X_probe_order_0` ), and to `reconstruct_signal` that outputs the files `o-DBs.YamboPy-curr_reconstructed_F*` 

```python
NLDB=YamboNLDB(calc="DBs")
SIG = Xn_from_sine(NLDB,l_out_current=True)
OUT = SIG.perform_analysis()
SIG.output_analysis(OUT)
SIG.reconstruct_signal(OUT)
```
### Example 3: Frequency mixing. Sum/difference of frequencies.

For two monochromatic external fields, the `Xn_from_freqmix` class is instantiated. Here, the default for the lower limit`T_range` is overwritten and set to `50 fs`.  The output of `perform_analysis` is passed  to `output_analysis` that in turn outputs `o-DBs.YamboPy-X_probe_order_?_?`  each containing the $\chi^{(|n|+|m|)} (-\omega_\sigma; n\omega_1, m\omega_2 )$. The sum/difference of frequencies is given for $n,m=1$ (`o-DBs.YamboPy-X_probe_order_1_1`) and $n=1, m=-1$ (`o-DBs.YamboPy-X_probe_order_1_-1`).  

```python
NLDB=YamboNLDB(calc="DBs")
SIG = Xn_from_freqmix(NLDB,T_range=[50.0*fs2aut,-1.0])
OUT = SIG.perform_analysis()
SIG.output_analysis(OUT)
```

Note: in all snippets one must add `from yambopy import *`

---

## Bibliography

1. Butcher PN, Cotter D. The constitutive relation. In: The Elements of Nonlinear Optics. Cambridge Studies in Modern Optics. Cambridge University Press; 1990:12-36.[doi.org/10.1017/CBO9781139167994](https://doi.org/10.1017/CBO9781139167994)
2. Attaccalite C and Grüning M, [Phys. Rev. B 88, 235113 (2013)](https://doi.org/10.1103/PhysRevB.88.235113)
3. Pionteck MN, Grüning M, Sanna S, Attaccalite C, [SciPost Phys. 19, 129 (2025)](https://10.21468/SciPostPhys.19.5.129) 
4. Romani A, Grüning M, 'Notes on nonlinear analysis from Gaussian pulses' (unpublished)
