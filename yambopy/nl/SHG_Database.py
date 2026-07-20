
import os
import numpy as np
import yaml

try:
    from refractiveindex import RefractiveIndexMaterial
except ImportError:
    RefractiveIndexMaterial = None
    
DEFAULT_DB_ROOT = os.path.expanduser("~/.refractiveindex.info-database")

_CATALOG = None
_DB_ROOT = None


def load_catalog(db_root=None, force_reload=False):
    "Load (once) and return the refractiveindex.info catalog."
    global _CATALOG, _DB_ROOT
    root = db_root or os.environ.get("REFRACTIVEINDEX_DB", DEFAULT_DB_ROOT)
    if _CATALOG is not None and not force_reload and root == _DB_ROOT:
        return _CATALOG
    catalog_path = os.path.join(root, "catalog-nk.yml")
    if not os.path.isfile(catalog_path):
        raise FileNotFoundError(
            "refractiveindex.info database not found at '%s'.\n"
            "It is auto-downloaded on the first RefractiveIndexMaterial(...) "
            "call (internet required), e.g.\n"
            "    from refractiveindex import RefractiveIndexMaterial\n"
            "    RefractiveIndexMaterial(shelf='main', book='Ag', page='Johnson')\n"
            "or set REFRACTIVEINDEX_DB / pass db_root=... to point at an "
            "existing copy." % root)
    with open(catalog_path) as f:
        _CATALOG = yaml.safe_load(f)
    _DB_ROOT = root
    return _CATALOG

def database_version(db_root=None):
    "Database snapshot stamp from its .version file (None if absent)."
    root = db_root or _DB_ROOT or os.environ.get("REFRACTIVEINDEX_DB",
                                                 DEFAULT_DB_ROOT)
    path = os.path.join(root, ".version")
    if os.path.isfile(path):
        with open(path) as f:
            return f.read().strip()
    return None

def search_database(keyword, db_root=None, exact_book=False):
    "Find database entries whose text contains keyword"
    catalog = load_catalog(db_root)
    keyword = keyword.lower()
    results = []
    for shelf_entry in catalog:
        if 'SHELF' not in shelf_entry:      # skip top-level dividers
            continue
        shelf = shelf_entry['SHELF']
        for book_entry in shelf_entry.get('content', []):
            if 'BOOK' not in book_entry:    # skip dividers
                continue
            book = book_entry['BOOK']
            book_name = book_entry.get('name', book)
            if exact_book and book.lower() != keyword:
                continue
            for page_entry in book_entry.get('content', []):
                if 'PAGE' not in page_entry:  # skip dividers
                    continue
                page = page_entry['PAGE']
                source = page_entry.get('name', page)
                text = ("%s %s %s %s" % (book, book_name, page, source)).lower()
                if exact_book or keyword in text:
                    results.append({'shelf': shelf, 'book': book, 'page': page,
                                    'material': book_name, 'source': source})
    return results

def load_material(result):
    "Turn one search-result into a RefractiveIndexMaterial."
    if RefractiveIndexMaterial is None:
        raise ImportError("the 'refractiveindex' package is required: "
                          "pip install refractiveindex")
    return RefractiveIndexMaterial(shelf=result['shelf'], book=result['book'],
                                   page=result['page'])


def print_search(keyword, db_root=None, exact_book=False):
    "Print matches with their energy range in eV and return the list."
    results = search_database(keyword, db_root=db_root, exact_book=exact_book)
    if not results:
        print("No matches for '%s'." % keyword)
        return results
    print("%d match(es) for '%s':\n" % (len(results), keyword))
    for i, r in enumerate(results):
        try:
            mat = load_material(r)
            lo, hi = mat.get_wl_range(unit='eV')
            lo, hi = min(lo, hi), max(lo, hi)
            erange = "%.2f - %.2f eV" % (lo, hi)
        except Exception:
            erange = "range unavailable"
        print("[%d] %s" % (i, r['material']))
        print("     shelf='%s', book='%s', page='%s'"
              % (r['shelf'], r['book'], r['page']))
        print("     %s" % r['source'])
        print("     energy range: %s\n" % erange)
    return results

def get_n(mat, value, unit='eV'):
    return mat.get_refractive_index(value, unit=unit)


def get_k(mat, value, unit='eV'):
    try:
        k = mat.get_extinction_coefficient(value, unit=unit)
    except Exception:
        return 0.0
    if k is None:
        return 0.0
    k = np.asarray(k, dtype=float)
    k = np.where(np.isnan(k), 0.0, k)
    return float(k) if k.ndim == 0 else k


def get_epsilon(mat, value, unit='eV'):
    n = get_n(mat, value, unit=unit)
    k = get_k(mat, value, unit=unit)
    return (np.asarray(n) + 1j * np.asarray(k))**2
