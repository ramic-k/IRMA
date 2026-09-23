"""Generic 2-column phonon-DOS reader for the mode-0 path (irma.spectra.dos_io)."""
import numpy as np
import pytest

from irma.spectra.dos_io import read_dos_2col


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_reads_mev_with_comments_and_headers(tmp_path):
    path = _write(tmp_path, "dos.txt",
                  "# a comment\nfreq_meV  dos\n0 0\n10 1.0\n20 4.0\n30 9.0\n40 0\n")
    omega, rho = read_dos_2col(path, unit="meV")
    assert omega[0] == 0.0 and np.allclose(np.diff(omega), np.diff(omega)[0])  # uniform
    assert omega[-1] == pytest.approx(0.040)                                   # 40 meV -> eV
    assert rho[0] == 0.0 and rho.max() > 0


def test_comma_separated_and_cm1_unit(tmp_path):
    # 1 meV = 8.0655 cm-1 -> 100 cm-1 ~ 12.4 meV
    path = _write(tmp_path, "dos.csv", "0,0\n100,1\n200,2\n400,0\n")
    omega, rho = read_dos_2col(path, unit="cm-1")
    assert omega[-1] == pytest.approx(400 * 1.239841984e-4, rel=1e-6)          # 400 cm-1 in eV
    assert np.all(np.isfinite(rho)) and rho.max() > 0


def test_resamples_nonuniform_grid_to_uniform(tmp_path):
    path = _write(tmp_path, "dos.txt", "0 0\n5 1\n7 3\n31 2\n50 0\n")           # non-uniform
    omega, rho = read_dos_2col(path, unit="meV")
    d = np.diff(omega)
    assert np.allclose(d, d[0], rtol=1e-9) and omega[0] == 0.0                  # now uniform
    # interpolated value at 7 meV is preserved (~3)
    assert rho[np.argmin(np.abs(omega - 0.007))] == pytest.approx(3.0, abs=0.3)


def test_imaginary_modes_dropped_with_warning(tmp_path):
    """QA4 F33: negative-frequency rows are DROPPED (with a warning), not
    folded onto omega=0 where interpolation would hand their weight to the
    first positive interval."""
    path = _write(tmp_path, "dos.txt", "-5 2\n0 0\n10 1\n20 2\n30 0\n")
    with pytest.warns(UserWarning, match="negative-frequency"):
        omega, rho = read_dos_2col(path, unit="meV")
    assert omega.min() >= 0.0 and np.all(rho >= 0.0)
    # the dropped row's weight must NOT appear near omega=0: the clean file
    # without the imaginary row gives the identical resampled DOS
    clean = _write(tmp_path, "clean.txt", "0 0\n10 1\n20 2\n30 0\n")
    omega2, rho2 = read_dos_2col(clean, unit="meV")
    assert np.array_equal(omega, omega2) and np.array_equal(rho, rho2)


def test_fine_grid_resample_floor(tmp_path):
    """QA4 F35: a very finely gridded DOS (e.g. an MD/VACF export) resamples
    onto a floored 0.01 meV step instead of inheriting the input spacing and
    blowing up the O(npt^2) phonon-expansion kernel."""
    rows = "".join(f"{w:.4f} {w/100.0}\n" for w in np.arange(0.0, 100.0005, 0.001))
    path = _write(tmp_path, "fine.txt", rows)                # 100k rows, 0.001 meV
    omega, rho = read_dos_2col(path, unit="meV")
    assert omega.size <= 100.0 / 0.01 + 2                    # floored at 0.01 meV
    assert omega.size >= 4 and rho.max() > 0


def test_bad_inputs_rejected(tmp_path):
    with pytest.raises(ValueError):                              # too few rows
        read_dos_2col(_write(tmp_path, "x.txt", "# only a comment\n1 2\n"), unit="meV")
    with pytest.raises(ValueError):                              # unknown unit
        read_dos_2col(_write(tmp_path, "y.txt", "0 0\n1 1\n2 2\n"), unit="furlongs")
