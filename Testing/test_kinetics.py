"""Tests of the kinetic modelling library without Slicer: python -m pytest Testing.

The curves are simulated independently of the library: the compartment equations are integrated with RK4 on a
0.002 min grid from a Feng FDG-like plasma curve, then averaged over every frame."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DICOMGatedDynamicPET"))
from DICOMGatedDynamicPETLib import kinetics as kin  # noqa: E402

DT = 0.002
DURATIONS_S = [10] * 6 + [30] * 4 + [60] * 5 + [120] * 5 + [300] * 8 + [600] * 2      # 78 min


def schedule():
    ends = np.cumsum(DURATIONS_S).astype(float)
    return ends - np.array(DURATIONS_S), ends


def plasma(t):
    """Feng et al. 1993 input (t in min)."""
    a1, a2, a3, l1, l2, l3 = 851.1, 21.88, 20.81, 4.134, 0.1191, 0.01043
    t = np.maximum(t, 0.0)
    return (a1 * t - a2 - a3) * np.exp(-l1 * t) + a2 * np.exp(-l2 * t) + a3 * np.exp(-l3 * t)


def simulate(K1, k2, k3=0.0, k4=0.0, vB=0.0, source=None):
    """Fine time grid and the tissue curve of a two tissue compartment model (k3 = 0: one compartment)."""
    t = np.arange(0.0, 78.0 + DT, DT)
    cp = plasma(t) if source is None else source
    c1 = np.zeros_like(t)
    c2 = np.zeros_like(t)

    def derivative(p, a, b):
        return K1 * p - (k2 + k3) * a + k4 * b, k3 * a - k4 * b

    for i in range(t.size - 1):
        a, b = c1[i], c2[i]
        p0, p2 = cp[i], cp[i + 1]
        p1 = (p0 + p2) / 2
        d1 = derivative(p0, a, b)
        d2 = derivative(p1, a + DT / 2 * d1[0], b + DT / 2 * d1[1])
        d3 = derivative(p1, a + DT / 2 * d2[0], b + DT / 2 * d2[1])
        d4 = derivative(p2, a + DT * d3[0], b + DT * d3[1])
        c1[i + 1] = a + DT / 6 * (d1[0] + 2 * d2[0] + 2 * d3[0] + d4[0])
        c2[i + 1] = b + DT / 6 * (d1[1] + 2 * d2[1] + 2 * d3[1] + d4[1])
    return t, (1 - vB) * (c1 + c2) + vB * cp


def frameMeans(t, curve, starts, ends):
    cumulative = np.concatenate([[0.0], np.cumsum((curve[1:] + curve[:-1]) * DT / 2)])
    s, e = starts / 60.0, ends / 60.0
    return (np.interp(e, t, cumulative) - np.interp(s, t, cumulative)) / (e - s)


@pytest.fixture(scope="module")
def setting():
    starts, ends = schedule()
    frames = kin.Frames(starts, ends)
    t = np.arange(0.0, 78.0 + DT, DT)
    sampled = kin.SampledInput(frames, t[::5], plasma(t[::5]))
    return frames, starts, ends, sampled


def close(value, expected, relative):
    assert value == pytest.approx(expected, rel=relative), (value, expected)


def test_frames_and_integrals():
    frames = kin.Frames([0, 60, 180], [60, 120, 240])
    np.testing.assert_allclose(frames.mid, [0.5, 1.5, 3.5])
    # constant 2 everywhere: integral up to the middles; the gap 2 - 3 min is linear between equal values
    np.testing.assert_allclose(frames.integralAtMid(np.array([2.0, 2.0, 2.0])), [1.0, 3.0, 7.0])
    np.testing.assert_allclose(frames.weights(kin.WEIGHT_DURATION), [1.0, 1.0, 1.0])
    late = kin.Frames([-60, 0, 60], [0, 60, 120])     # the first frame ends at time zero: not used
    assert list(late.valid) == [False, True, True]
    assert late.weights()[0] == 0
    grid = frames.grid()
    np.testing.assert_allclose(frames.frameMeans(grid, grid), frames.mid, rtol=1e-6)


def test_frames_from_intervals_time_zero():
    intervals = [{"start": 100.0, "end": 160.0}, {"start": 160.0, "end": 220.0}]
    np.testing.assert_allclose(kin.Frames.fromIntervals(intervals).mid, [0.5, 1.5])
    np.testing.assert_allclose(kin.Frames.fromIntervals(intervals, 40.0).mid, [1.5, 2.5])   # injection 1 min earlier


def test_two_tissue_reversible(setting):
    frames, starts, ends, sampled = setting
    truth = {"K1": 0.12, "k2": 0.15, "k3": 0.08, "k4": 0.04, "vB": 0.05}
    t, curve = simulate(**truth)
    tac = frameMeans(t, curve, starts, ends)
    result = kin.fit("2TC", frames, tac, inputFunction=sampled)
    for name, value in truth.items():
        close(result["values"][name], value, 0.03)
    close(result["values"]["VT"], 0.12 / 0.15 * 3.0, 0.02)
    close(result["values"]["BPnd"], 2.0, 0.03)
    assert result["metrics"]["R2"] > 0.9999
    assert result["metrics"]["NRMSE"] < 0.01
    assert kin.resultColumns("2TC")[-4:] == ["R2", "RMSE", "NRMSE", "AIC"]


def test_two_tissue_irreversible_and_patlak(setting):
    frames, starts, ends, sampled = setting
    truth = {"K1": 0.1, "k2": 0.12, "k3": 0.06, "vB": 0.04}
    t, curve = simulate(**truth)
    tac = frameMeans(t, curve, starts, ends)
    ki = 0.1 * 0.06 / 0.18
    result = kin.fit("2TCi", frames, tac, inputFunction=sampled)
    close(result["values"]["Ki"], ki, 0.02)
    close(result["values"]["k3"], 0.06, 0.03)
    patlak = kin.fit("Patlak", frames, tac, inputFunction=sampled, options={"tStar": 20.0})
    close(patlak["values"]["Ki"], ki, 0.05)
    assert patlak["values"]["t*"] == 20.0
    assert patlak["graphical"]["used"].sum() == np.count_nonzero(frames.mid >= 20.0)
    assert patlak["metrics"]["R2"] > 0.999
    auto = kin.fit("Patlak", frames, tac, inputFunction=sampled, options={"autoTStar": True, "maxError": 5.0})
    close(auto["values"]["Ki"], ki, 0.05)
    assert 2.0 < auto["values"]["t*"] < 40.0


def test_one_tissue_and_logan(setting):
    frames, starts, ends, sampled = setting
    t, curve = simulate(K1=0.2, k2=0.25, vB=0.03)
    tac = frameMeans(t, curve, starts, ends)
    result = kin.fit("1TC", frames, tac, inputFunction=sampled)
    close(result["values"]["K1"], 0.2, 0.02)
    close(result["values"]["VT"], 0.8, 0.02)
    t, curve = simulate(K1=0.12, k2=0.15, k3=0.08, k4=0.04)
    tac = frameMeans(t, curve, starts, ends)
    logan = kin.fit("Logan", frames, tac, inputFunction=sampled, options={"tStar": 30.0})
    close(logan["values"]["VT"], 2.4, 0.05)


def test_delay_fit(setting):
    frames, starts, ends, sampled = setting
    t = np.arange(0.0, 78.0 + DT, DT)
    t2, curve = simulate(K1=0.2, k2=0.25, source=plasma(t - 0.3))      # the tissue sees the input 0.3 min later
    tac = frameMeans(t2, curve, starts, ends)
    params = kin.model("1TC").defaultParameters()
    for p in params:
        if p.name == "delay":
            p.fitted = True
        if p.name == "vB":
            p.fitted, p.initial = False, 0.0
    result = kin.fit("1TC", frames, tac, inputFunction=sampled, options={"parameters": params})
    close(result["values"]["delay"], 0.3, 0.1)
    close(result["values"]["K1"], 0.2, 0.03)


def test_reference_methods(setting):
    frames, starts, ends, _ = setting
    # reference: K1' 0.1, k2' 0.15; target 1TC with R1 1.2, BPnd 2: k2 = 0.18, k2a = 0.06 (SRTM holds exactly)
    t, reference = simulate(K1=0.1, k2=0.15)
    _, target = simulate(K1=0.12, k2=0.06)
    ref = kin.FrameInput(frames, frameMeans(t, reference, starts, ends))
    tac = frameMeans(t, target, starts, ends)
    srtm = kin.fit("SRTM", frames, tac, reference=ref)
    close(srtm["values"]["BPnd"], 2.0, 0.03)
    close(srtm["values"]["R1"], 1.2, 0.03)
    close(srtm["values"]["k2'"], 0.15, 0.05)
    assert srtm["metrics"]["R2"] > 0.999
    logan = kin.fit("LoganRef", frames, tac, reference=ref, options={"tStar": 30.0})
    close(logan["values"]["BPnd"], 2.0, 0.07)
    suvr = kin.fit("SUVR", frames, tac, reference=ref, options={"window": (60.0, 78.0)})
    late = (frames.mid > 60)
    assert suvr["values"]["SUVR"] > 2.0
    assert suvr["metrics"]["R2"] is None
    assert late.any()
    # voxel-wise: three voxels, the target, the reference and twice the target
    data = np.column_stack([tac, ref.frameMeans(), 2 * tac])
    maps = kin.parametricMaps("SRTM", frames, data, reference=ref)
    close(float(maps["BPnd"][0]), 2.0, 0.08)
    assert abs(float(maps["BPnd"][1])) < 0.05
    close(float(maps["R1"][2]), 2.4, 0.08)
    maps = kin.parametricMaps("LoganRef", frames, data, reference=ref, options={"tStar": 30.0})
    close(float(maps["BPnd"][0]), 2.0, 0.07)
    maps = kin.parametricMaps("SUVR", frames, data, reference=ref, options={"window": (60.0, 78.0)})
    close(float(maps["SUVR"][0]), suvr["values"]["SUVR"], 1e-6)
    with pytest.raises(ValueError):
        kin.parametricMaps("2TC", frames, data, inputFunction=ref)


def test_voxel_patlak_and_one_tissue(setting):
    frames, starts, ends, sampled = setting
    t, curve = simulate(K1=0.1, k2=0.12, k3=0.06)
    tac = frameMeans(t, curve, starts, ends)
    maps = kin.parametricMaps("Patlak", frames, np.column_stack([tac, tac * 0.5]), inputFunction=sampled,
                              options={"tStar": 20.0})
    single = kin.fit("Patlak", frames, tac, inputFunction=sampled, options={"tStar": 20.0})
    close(float(maps["Ki"][0]), single["values"]["Ki"], 1e-3)
    close(float(maps["Ki"][1]), single["values"]["Ki"] / 2, 1e-3)
    t, curve = simulate(K1=0.2, k2=0.25, vB=0.05)
    tac = frameMeans(t, curve, starts, ends)
    maps = kin.parametricMaps("1TC", frames, tac[:, None], inputFunction=sampled)
    close(float(maps["VT"][0]), 0.8, 0.08)


def test_image_derived_input_is_consistent():
    """A tissue curve made from the image-derived input as the library interpolates it is fitted back exactly."""
    starts, ends = schedule()
    frames = kin.Frames(starts, ends)
    t = np.arange(0.0, 78.0 + DT, DT)
    blood = kin.FrameInput(frames, frameMeans(t, plasma(t), starts, ends))
    _, curve = simulate(K1=0.12, k2=0.15, k3=0.08, k4=0.04, vB=0.05, source=blood.onGrid(t))
    tac = frameMeans(t, curve, starts, ends)
    result = kin.fit("2TC", frames, tac, inputFunction=blood, options={"weighting": kin.WEIGHT_UNIFORM})
    close(result["values"]["k3"], 0.08, 0.05)
    close(result["values"]["VT"], 2.4, 0.03)


def test_auto_tstar_and_metrics():
    frames = kin.Frames(np.arange(10) * 60.0, np.arange(1, 11) * 60.0)
    x = np.arange(10, dtype=float)
    y = np.where(x < 4, x ** 2, 3 * x + 4)      # straight from frame 4 on (3 x + 4 meets x^2 at x = 4)
    first = kin.autoTStar(frames, x, y, np.ones(10), maxError=1.0)
    assert first == 4
    metrics = kin.fitMetrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], np.ones(3), 2)
    assert metrics["R2"] == 1.0 and metrics["RMSE"] == 0.0 and metrics["AIC"] is None
    metrics = kin.fitMetrics([1.0, 2.0, 3.0], [1.1, 2.0, 2.9], np.ones(3), 1)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(0.02 / 3))
    assert metrics["NRMSE"] == pytest.approx(np.sqrt(0.02 / 3) / 2.0)
    assert metrics["n"] == 3


def test_levenberg_marquardt_bounds():
    x = np.linspace(0, 1, 20)
    data = 3.0 * np.exp(-2.0 * x)
    p, cost = kin.levenbergMarquardt(lambda q: q[0] * np.exp(-q[1] * x) - data, [1.0, 0.5], [0, 0], [10, 10])
    np.testing.assert_allclose(p, [3.0, 2.0], rtol=1e-5)
    p, _ = kin.levenbergMarquardt(lambda q: q[0] * np.exp(-q[1] * x) - data, [1.0, 0.5], [0, 0], [2.5, 10])
    assert p[0] == pytest.approx(2.5)


def test_csv_reading_and_writing():
    times, values = kin.readCurveCsv("Time (min),Plasma (Bq/ml)\n0,0\n0.5,100\n1,80\n")
    np.testing.assert_allclose(times, [0, 0.5, 1])
    times, values = kin.readCurveCsv("0;0\n30;100\n60;80\n", timeUnit="s")
    np.testing.assert_allclose(times, [0, 0.5, 1])
    np.testing.assert_allclose(values, [0, 100, 80])
    with pytest.raises(ValueError):
        kin.readCurveCsv("time,value\n1,2\n")
    frames = kin.Frames([0, 60], [60, 120])
    result = {"model": "SUVR", "values": {"SUVR": 1.5, "BPnd": 0.5}, "metrics": {}, "warnings": []}
    text = kin.resultsCsv([("Striatum", result, ())], "SUVR")
    assert text.splitlines() == ["Region,SUVR,BPnd,Warnings", "Striatum,1.5,0.5,"]
    tacs = kin.tacCsv(frames, {"Striatum": [1.0, 2.0]}, "Bq/ml")
    assert tacs.splitlines()[0] == "Start (min),End (min),Mid (min),Striatum (Bq/ml)"


def test_parameters_from_dicts():
    params = kin._parameterList("2TC", [{"name": "k4", "initial": 0.0, "fitted": False},
                                        {"name": "vB", "lower": 0.1, "upper": 0.0}])
    k4 = next(p for p in params if p.name == "k4")
    vB = next(p for p in params if p.name == "vB")
    assert not k4.fitted and k4.initial == 0.0
    assert (vB.lower, vB.upper) == (0.0, 0.1)
    assert kin.model("2TC").parameters[3].fitted       # the model's own defaults stay as they were
    with pytest.raises(ValueError):
        kin.model("XYZ")


def test_voxel_maps_clip_negative(setting):
    frames, starts, ends, _ = setting
    t, reference = simulate(K1=0.1, k2=0.15)
    ref = kin.FrameInput(frames, frameMeans(t, reference, starts, ends))
    data = np.column_stack([ref.frameMeans(), 0.5 * ref.frameMeans()])     # SUVR 1 and 0.5: BPnd 0 and -0.5
    clipped = kin.parametricMaps("LoganRef", frames, data, reference=ref, options={"tStar": 30.0})
    assert float(clipped["BPnd"][1]) == 0.0
    raw = kin.parametricMaps("LoganRef", frames, data, reference=ref,
                             options={"tStar": 30.0, "clipNegative": False})
    assert float(raw["DVR"][1]) == pytest.approx(0.5, abs=0.02)
    assert float(raw["BPnd"][1]) == 0.0                  # BPnd stays within BPND_RANGE


def test_bpnd_limits(setting):
    frames, starts, ends, _ = setting
    t, reference = simulate(K1=0.1, k2=0.15)
    ref = kin.FrameInput(frames, frameMeans(t, reference, starts, ends))
    result = kin.fit("SUVR", frames, 0.5 * ref.frameMeans(), reference=ref)        # BPnd -0.5
    assert result["values"]["BPnd"] == 0.0
    assert result["values"]["SUVR"] == pytest.approx(0.5, abs=0.01)
    assert any("BPnd" in w for w in result["warnings"])
    assert kin.limitBPnd(250.0) == kin.BPND_RANGE[1] and kin.limitBPnd(None) is None


def test_sort_texts():
    assert kin.sortTexts(["1.5", "", "0.2", "1.000e-05", "3"]) == [3, 2, 0, 4, 1]           # numbers, empty last
    assert kin.sortTexts(["1.5", "", "0.2", "3"], descending=True) == [3, 0, 2, 1]
    assert kin.sortTexts(["Region 10", "region 2", "Caudate L", ""]) == [2, 1, 0, 3]        # names, natural order
    assert kin.sortTexts(["b", "a", "b"]) == [1, 0, 2]                                      # stable
    assert kin.sortTexts([]) == []


def test_mask_on_grid():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)       # 1 mm grid, cube of voxels 2..7
    mask[2:8, 2:8, 2:8] = 1
    same = kin.maskOnGrid(mask, np.eye(4), mask.shape, np.eye(4), slicesPerChunk=3)
    assert np.array_equal(same, mask != 0)
    coarse = np.diag([2.0, 2.0, 2.0, 1.0])               # 2 mm grid over the same space
    out = kin.maskOnGrid(mask, np.eye(4), (6, 6, 6), coarse)
    assert out[2, 2, 2] and out[3, 3, 3] and not out[0, 0, 0] and not out[5, 5, 5]
    shifted = np.eye(4)
    shifted[:3, 3] = 100.0                               # entirely outside the mask's grid
    assert not kin.maskOnGrid(mask, np.eye(4), (4, 4, 4), shifted).any()
