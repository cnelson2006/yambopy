# From `yambo_nl` to a measurable SHG spectrum: an end-to-end tutorial for `shg_intensity`

This tutorial walks through the complete workflow of the `shg_intensity` module
(`yambopy/nl/shg_intensity.py`): starting from the raw output of a `yambo_nl`
real-time simulation and ending with a second-harmonic generation (SHG) intensity
spectrum that can be compared directly with experiment. Each step shows the code
first, then explains what it does and why it is done that way.

Throughout, all quantities in the module's public interface are **SI** — lengths
in m, intensities in W m⁻², sheet susceptibilities in m² V⁻¹ — with two
deliberate exceptions: photon energies are passed in **eV**, and the raw
`o.YamboPy-X_probe` spectra are in **Gaussian (esu) units**, converted explicitly
during the workflow.

All snippets assume:

```python
from yambopy import *
from yambopy.nl.shg_intensity import *
```

---

## 1. What problem does this module solve?

A `yambo`/`lumen` real-time run gives you a *microscopic* nonlinear
susceptibility χ⁽²⁾. An experimentalist measures a *macroscopic* SHG intensity
from a sample sitting on a substrate. Three separate gaps stand between the two
numbers, and the module closes each one:

1. **Units and normalisation.** The simulation output is in Gaussian units and is
   averaged over a supercell that is mostly vacuum, so its magnitude depends on
   an arbitrary computational choice (the vacuum padding). It must be converted
   to a vacuum-independent SI quantity.
2. **The environment.** The measured signal depends strongly on what the
   monolayer sits on. A SiO₂ film on Si acts as an interferometer at both the
   fundamental and the harmonic; constructive or destructive interference can
   change the detected SHG by orders of magnitude. This is captured by a
   *structure factor*.
3. **Trust.** Published formulas for step 2 disagree with each other (one widely
   cited paper carries a spurious factor of (4π)² ≃ 158 from an incomplete
   unit conversion). The module therefore implements *several independently
   derived models* and treats their agreement as the validation criterion.

The workflow is: **load → convert → describe the materials → run the models →
cross-check**.

---

## 2. What you need before starting

Three outputs of a `yambo_nl` run analysed with `Xn_from_sine`:

* `o.YamboPy-X_probe_order_1` and `_2` — the χ⁽¹⁾ and χ⁽²⁾ spectra
  (Gaussian units). The column layout is fixed for all files:
  `E[eV], Im(x), Re(x), Im(y), Re(y), Im(z), Re(z)`.
* `SAVE/ns.db1` — the lattice database, which provides the supercell height.
* `SAVE/ndb.Nonlinear` — the field database, which records the pump intensity
  actually applied in the run.

Substrate optical constants come from the
[refractiveindex.info](https://refractiveindex.info) database, auto-downloaded on
first use (location configurable via the `REFRACTIVEINDEX_DB` environment
variable).

---

## 3. Step 1 — load the run with `ChiLoader`

```python
CHI = ChiLoader('.', 'SAVE', h_2D=0.65e-9)
print(CHI)
```

`ChiLoader` is the front door of the module. One object represents one run: it
reads *both* spectra, keeps the photon-energy grid, opens the lattice database,
and stores the monolayer thickness you intend to use. The constructor arguments
are the folder containing the `o.YamboPy-X_probe` files, the run's `SAVE`
folder, and `h_2D`, the **effective monolayer thickness in m** (0.65 nm is the
conventional value for MoS₂ — more on why it matters in Step 4).

Two design points are worth understanding rather than just using:

**χ⁽²⁾ is a 3D object.** `CHI.order1` and `CHI.order2` are complex arrays of
shape `(N, 3)`: all three Cartesian components, columns (x, y, z), loaded
together. Nothing forces you to choose a direction at load time; the choice is
made — visibly — at the point of use:

```python
chi2_x = CHI.component(2, 'x')      # 1D complex array
chi2_y = CHI.component(2, 'y')
chi2_z = CHI.component(2, 'z')
```

**The z component is a free sanity check.** A flat monolayer (point group
D₃ₕ) cannot have an out-of-plane second-order response — it is forbidden by
symmetry. So before anything else:

```python
import numpy as np
assert np.allclose(CHI.component(2, 'z'), 0.0)
```

If this fails, something is wrong with the run (geometry, field direction, or
convergence), and no amount of downstream analysis will fix it. The x and y
columns are the two symmetry-related in-plane components; x is the default
throughout the module.

`CHI.omega_eV` is the photon-energy grid of the run, in eV, common to both
spectra — every model call below takes it as the frequency axis.

---

## 4. Step 2 — convert to physically meaningful quantities

```python
CHI.supercell_height_SI()          # -> CHI.Lz          (m)
CHI.chi2_supercell_to_sheet_SI()   # -> CHI.SHG_sheet   (N, 3), m^2/V
CHI.chi2_supercell_to_eff_SI()     # -> CHI.SHG_eff     (N, 3), m/V
CHI.nk_from_chi()                  # -> CHI.n, CHI.k
```

Each method stores its result on the object and returns it; the order above is
the natural one, and the methods raise informative errors if a prerequisite is
missing (for example, asking for the sheet susceptibility before `Lz` is set).

**Why `supercell_height_SI` reads the database instead of trusting you.** The
supercell height `Lz` enters every conversion, so it is read directly from
`SAVE/ns.db1` — the value the calculation actually used — rather than typed by
hand. This removes an entire class of transcription errors. (It assumes the
out-of-plane lattice vector is axis-aligned, which is the standard 2D-material
setup.)

**Why the *sheet* susceptibility is the primary quantity.** The real-time code
averages the induced polarisation over the whole supercell, most of which is
vacuum. Double the vacuum padding and the raw χ⁽²⁾ halves — clearly not
physics. Multiplying by `Lz`,

χ⁽²⁾ₛ = (4π / c·10⁻⁴) · Lz · χ⁽²⁾_supercell   [m²/V],

undoes the vacuum dilution (and performs the strict Gaussian→SI conversion in
the same step), giving a quantity independent of the computational box. This is
what `SHG_sheet` holds, and it is what the recommended intensity models consume.

**Why the "effective" (bulk-equivalent) value exists at all.** The literature
often quotes χ⁽²⁾ in pm/V, which presumes a 3D material. `SHG_eff = SHG_sheet /
h_2D` produces that number — but note what this means: it depends on the
*conventional choice* of monolayer thickness `h_2D`. Two papers using h = 0.65
nm and h = 0.61 nm will quote different pm/V values for identical physics. Use
`SHG_eff` to compare with the literature; use `SHG_sheet` for actual
predictions.

**Why n and k come from the run's own χ⁽¹⁾.** The interference model below
needs an effective refractive index for the monolayer. Taking it from the same
simulation that produced χ⁽²⁾ (via ε = 1 + 4π(Lz/h_2D)χ⁽¹⁾) keeps the band
gap, excitonic features, and broadening *self-consistent* between the linear and
nonlinear parts — mixing your χ⁽²⁾ with someone else's measured monolayer index
would not.

---

## 5. Step 3 — describe the materials

The models accept any object exposing `n(E)`, `k(E)`, `complex_index(E)`,
`epsilon(E)`, `wl_range_eV()`, `covers(E)`. Two classes provide this interface
from the two places optical constants come from.

**The monolayer, from the simulation:**

```python
MoS2 = SimulatedMaterial(CHI.omega_eV, CHI.component(1, 'x'),
                         CHI.Lz, CHI.h_2D, name="MoS2")
```

**The substrate layers, from experiment:**

```python
print_search("SiO2")        # list available records with energy ranges
silica = Substrate("SiO2", record_index=8)
si     = Substrate("Si",   record_index=200)
print(silica); print(si)    # provenance: source paper, validity range
```

Why a database rather than hard-coded constants: substrate optics should come
from *citable experimental measurements*, and a material name alone is
ambiguous — SiO₂ has many records from different measurement papers covering
different ranges. `print_search` shows what exists so the choice is explicit,
and the chosen `source` is provenance that belongs in your thesis methods
section. If you need long-term reproducibility, select by `source=` (stable
against database updates) rather than by index, and record
`database_version()`.

A behavioural difference to know: `Substrate` raises a hard error outside its
tabulated range (extrapolating measured data is silently wrong), while
`SimulatedMaterial` returns `NaN` outside its grid (your own grids are dense,
and NaN keeps full-spectrum plots alive).

---

## 6. Step 4 — compute the SHG intensity

```python
I0 = field_intensity_SI('SAVE/ndb.Nonlinear')     # W/m^2, the run's own pump

stack = Stack(MoS2, silica, si, film_thickness=285e-9, h_2D=CHI.h_2D)
beta  = stack.structure_factor(CHI.omega_eV)
I_shg = stack.shg_intensity(CHI.omega_eV, CHI.SHG_sheet[:, 0], I0)   # W/m^2
```

`Stack` implements the structure-factor model of Song et al. (Opt. Express 31,
19746 (2023)) for the geometry air / 2D material / film(d) / substrate:

I(2ω) = |β|² · |2π χ⁽²⁾ₛ / λ|² · I² / (2ε₀c),  β = (1+R_ω)² (1+R_{2ω}).

All environmental physics lives in β: R_ω and R_{2ω} are the reflection
coefficients of the *full* structure at the fundamental and the harmonic,
including the film interference and the monolayer's own linear response. The
factor appears squared at ω because two pump photons drive the process, and once
at 2ω for the out-coupling of the generated light. |β|² ranges from 0 (fully
destructive at both frequencies — the film can *hide* your signal) to 64;
inspecting `structure_factor` separately from the intensity tells you whether a
weak measured signal is weak physics or unlucky interference, and lets you
choose an oxide thickness *before* a measurement.

Why the pump intensity is read from `ndb.Nonlinear` rather than typed in: it is
the field the simulation actually applied, so the predicted intensity is
consistent with the extracted χ⁽²⁾ by construction.

Note the explicit `[:, 0]` selecting the x component of the sheet
susceptibility: the models are scalar in the tensor sense, so which component
(or symmetry-adapted combination) enters is a *visible physics decision* in the
analysis, not a hidden loader default.

**Practical caveat — coverage at 2ω.** For every energy, every layer must have
optical data at both ω *and* 2ω. Energies failing this test are returned as
`NaN` (with a printed warning listing the usable window) rather than silently
extrapolated. Consequence: your usable fundamental range ends at *half* the
maximum tabulated energy of your narrowest material. If the top of a spectrum
comes back NaN, check `stack.usable_omega(CHI.omega_eV)` before suspecting
anything else.

---

## 7. Step 5 — cross-check before you believe anything

The module's design philosophy: never trust one formula. Three independently
derived, independently coded models are provided, and agreement between them is
the validation.

```python
# d = 0 on a transparent substrate: two independent code paths, same physics
stack0 = Stack(MoS2, silica, silica, film_thickness=0.0, h_2D=CHI.h_2D)
I_st0  = stack0.shg_intensity(CHI.omega_eV, CHI.SHG_sheet[:, 0], I0)
I_wood = WoodwardModel(MoS2, silica, CHI.h_2D).sheet_intensity(
             CHI.omega_eV, CHI.SHG_sheet[:, 0], I0)
ratio  = I_st0 / I_wood        # should be ~1 across the usable range
```

`WoodwardModel` is the strict-SI single-interface sheet formula
(I(2ω) = 32 ω² |χ⁽²⁾ₛ|² I² / (ε₀ c³ (1+n)⁶), Woodward et al., 2D Mater. 4,
011006 (2017)). At zero film thickness on a transparent substrate it describes
the same physics as `Stack` through a completely different derivation, so their
ratio should be ≃ 1. **Run this check once per new dataset.** If it drifts from
1, a unit or convention error has crept in somewhere — this style of
cross-check is precisely how a factor-158 discrepancy in the published
literature was identified during the development of this module.

On that subject: the widely cited sheet formula of Clark et al. (Phys. Rev. B
90, 121409(R) (2014), Eq. (1)) carries a spurious (4π)² from an incomplete
Gaussian→SI conversion of Bloembergen & Pershan (1962). For a given SI χ⁽²⁾ₛ it
overestimates I(2ω) by ≃ 158, and susceptibilities extracted with it sit 4π
below strict-SI values. The module deliberately does *not* implement that
formula; it retains Clark's *bulk thin-slab* reference (their Eq. (4)) as
`ClarkModel`, which is correct and useful for thickness-scaling comparisons:

```python
I_bulk = ClarkModel(MoS2, silica, CHI.h_2D).bulk_intensity(
             CHI.omega_eV, CHI.SHG_eff[:, 0], I0)     # takes the BULK chi2, m/V
```

Note `ClarkModel` consumes `SHG_eff` (m/V), not `SHG_sheet` — it is a bulk
formula. `WoodwardModel` also warns loudly if given an absorbing substrate,
because its derivation assumes transparency; for absorbing substrates the
`Stack` model is the correct tool.

---

## 8. The whole workflow in one script

```python
import numpy as np
from yambopy import *
from yambopy.nl.shg_intensity import *

# --- 1. load the run -------------------------------------------------------
CHI = ChiLoader('.', 'SAVE', h_2D=0.65e-9)
assert np.allclose(CHI.component(2, 'z'), 0.0)      # D3h sanity check

# --- 2. convert ------------------------------------------------------------
CHI.supercell_height_SI()
CHI.chi2_supercell_to_sheet_SI()                    # m^2/V, all components
CHI.chi2_supercell_to_eff_SI()                      # m/V, for pm/V literature
CHI.nk_from_chi()

# --- 3. materials ----------------------------------------------------------
MoS2   = SimulatedMaterial(CHI.omega_eV, CHI.component(1, 'x'),
                           CHI.Lz, CHI.h_2D, name="MoS2")
silica = Substrate("SiO2", record_index=8)
si     = Substrate("Si",   record_index=200)
I0     = field_intensity_SI('SAVE/ndb.Nonlinear')

# --- 4. intensity ----------------------------------------------------------
stack = Stack(MoS2, silica, si, film_thickness=285e-9, h_2D=CHI.h_2D)
beta  = stack.structure_factor(CHI.omega_eV)
I_shg = stack.shg_intensity(CHI.omega_eV, CHI.SHG_sheet[:, 0], I0)

# --- 5. cross-check --------------------------------------------------------
stack0 = Stack(MoS2, silica, silica, film_thickness=0.0, h_2D=CHI.h_2D)
ratio  = (stack0.shg_intensity(CHI.omega_eV, CHI.SHG_sheet[:, 0], I0)
          / WoodwardModel(MoS2, silica, CHI.h_2D).sheet_intensity(
                CHI.omega_eV, CHI.SHG_sheet[:, 0], I0))
print("d=0 Stack/Woodward median ratio:", np.nanmedian(ratio))   # expect ~1
```

---

## 9. Things that commonly go wrong

**The top of the spectrum is NaN.** Coverage at 2ω (Section 6): your usable
range ends at half the narrowest material's maximum tabulated energy. Inspect
`stack.usable_omega(...)` and either restrict the grid or choose a substrate
record with wider coverage.

**pm/V values don't match a paper.** Check the paper's assumed monolayer
thickness first — bulk-equivalent χ⁽²⁾ scales as 1/h_2D — and then check
whether the paper's extraction pipeline descends from Clark Eq. (1), in which
case its susceptibilities sit a factor 4π below strict-SI values by
construction.

**Numbers change after a database update.** You selected substrate records by
`record_index`. Re-select by `source=` and record `database_version()`.

**A different run gives wildly different raw χ⁽²⁾.** Compare `Lz` first: the
raw supercell susceptibility scales with vacuum padding. The *sheet* values are
the comparable quantities.

**`chi2_supercell_to_eff_SI` or `nk_from_chi` raises.** Both need `h_2D > 0`
(and `Lz` set): the effective quantities are undefined without a thickness
convention. Construct the loader with `h_2D=` or set `CHI.h_2D` before calling.

---

## 10. Where the formulas come from

Song et al., Opt. Express 31, 19746 (2023) — structure-factor model (`Stack`);
Cheng et al., J. Phys. Photonics 1, 015002 (2019) — the sheet-optics framework
it builds on; Woodward et al., 2D Mater. 4, 011006 (2017) — strict-SI
single-interface formula (`WoodwardModel`); Clark et al., Phys. Rev. B 90,
121409(R) (2014) — bulk thin-slab reference (`ClarkModel`) and the unit-convention
cautionary tale; Bloembergen & Pershan, Phys. Rev. 128, 606 (1962) — the
original radiating-sheet theory; Butcher & Cotter, *The Elements of Nonlinear
Optics* (CUP, 1990) — the bulk formula's textbook form.
