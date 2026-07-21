import sys, types, numpy as np
def _err(): raise AssertionError("value mismatch")

# stub yambopy.units so no yambopy install is needed
u = types.ModuleType("yambopy.units")
u.AU2M = 5.29177210903e-11
u.FREE_SPACE_PERM = 8.8541878128e-12
u.speed_of_light_SI = 299792458.0
u.hbar = 1.054571817e-34
u.electron_charge_SI = 1.602176634e-19
u.AU2KWCMm2 = 6.436409e15
yp = types.ModuleType("yambopy"); yp.units = u
sys.modules["yambopy"] = yp; sys.modules["yambopy.units"] = u

try:
    import shg_analysis as S
except ImportError:
    import shg_analysis_incomplete as S
print("module imported OK\n")

# fake transparent substrate so no database is needed
N_SUB, H, CHI, I0 = 1.45, 0.65e-9, 2.0e-20, 1.0e13
E = np.linspace(0.5, 1.5, 21)

class Fake:
    def get_wl_range(self, unit='eV'): return (0.3, 8.0)
    def get_refractive_index(self, v, unit='eV'):
        return np.full(np.shape(v) or (), N_SUB)
    def get_extinction_coefficient(self, v, unit='eV'):
        raise Exception("no k")

S.search_database = lambda n, db_root=None, exact_book=False: [
    {'shelf':'main','book':n,'page':'f','material':n,'source':'Fake'}]
S.load_material = lambda r: Fake()

g = np.linspace(0.3, 6.0, 300)
MoS2 = S.SimulatedMaterial(g, np.zeros(len(g), complex), 2.1e-9, H, name="fake2D")
sub = S.Substrate("SiO2")

results = []
def check(name, fn):
    try:
        fn(); print("PASS ", name); results.append(True)
    except Exception as e:
        print("FAIL ", name, "\n       ", type(e).__name__, e); results.append(False)

check("conversions", lambda: (
    np.isclose(S.ESU_TO_SI_CHI2, 4*np.pi/(299792458.0*1e-4)) or _err()))
check("n,k transparent limit", lambda: np.allclose(
    S.nk_from_chi1_supercell(np.zeros(5, complex), 2e-9, H)[0], 1.0) or _err())
check("Substrate", lambda: np.isclose(sub.n(1.0), N_SUB) or _err())
check("Stack runs at all", lambda: S.Stack(MoS2, sub, sub, 0.0, H).structure_factor(E))
check("Stack beta value", lambda: np.allclose(
    S.Stack(MoS2, sub, sub, 0.0, H).structure_factor(E),
    (2/(1+N_SUB))**3, rtol=1e-10) or _err())
check("Stack(d=0) == Woodward", lambda: np.allclose(
    S.Stack(MoS2, sub, sub, 0.0, H).shg_intensity(E, CHI, I0),
    S.WoodwardModel(MoS2, sub, H).sheet_intensity(E, CHI, I0), rtol=1e-8) or _err())
check("Clark bulk", lambda: np.isfinite(
    S.ClarkModel(MoS2, sub, H).bulk_intensity(E, CHI/H, I0)).all() or _err())

print("\n%d/%d passed" % (sum(results), len(results)))