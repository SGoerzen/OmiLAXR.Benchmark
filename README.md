# OmiLAXR.Benchmark

Unity package for running reproducible performance benchmarks in the OmiLAXR ecosystem. It provides a benchmark runner, a stress scenario, and CSV/log outputs you can compare across tracking configurations and builds.

**Highlights**
- Warmup + measurement workflow with automatic start/stop.
- Frame time, FPS, memory, and GC allocation sampling.
- CSV aggregation for plotting and comparison.
- Stress scenario to generate deterministic load.
- Optional synthetic event generation with custom metrics.

Just drag and drop the prefab `Resources/Prefabs/Benchmark.prefab` into your scene (or use one of the sample scenes).

## Compatibility

OmiLAXRv2 is modular and works with common XR stacks (MRTK, VRTK, SteamVR, UnityXR, etc.) and a broad range of hardware.

The only hard requirement is the Unity version. This package targets Unity `2020.3` and is tested with `2020.3.15f1` (see `package.json`). If you run a different version, feedback and contributions are welcome.

## Dependencies

- `com.rwth.unity.omilaxr` `2.2.0`

## Install

### Install Using Git URL

1. Go to **Window** -> **Package Manager**.
2. Click the `+` button.
3. Select **Add package from git URL**.
4. Paste `https://github.com/SGoerzen/OmiLAXR.Benchmark.git` and confirm.

### Install via `manifest.json`

Add this to `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.rwth.unity.omilaxr.benchmark": "1.0.0"
  }
}
```

## Quick Start

1. Open a sample scene:
- With OmiLAXR tracking: `Resources/Scenes/BenchmarkScene_WithOmiLAXR.unity`
- Baseline (no OmiLAXR): `Resources/Scenes/BenchmarkScene_WithoutOmiLAXR.unity`
2. Press Play. Warmup runs first, then measurement starts automatically.
3. On completion, the console prints absolute paths to the log and CSV outputs.

## Benchmark Runner

Add the `Benchmark` component (included in `Resources/Prefabs/Benchmark.prefab`) and adjust the main settings:

- `warmupSeconds`: warmup duration before measurement starts.
- `autoStartOnEnable`: start automatically on enable.
- `autoQuitAfterSeconds`: auto end measurement after N seconds (0 = disabled).
- `frameSpikeThresholdMs`: count spikes above this threshold (ms).
- `csvIntervalSeconds`: time-based bin size for CSV output.

Custom metrics:
```csharp
OmiLAXR.Benchmark.Benchmark.StartRecord("MyMetric");
// ... work ...
OmiLAXR.Benchmark.Benchmark.StopRecord("MyMetric");
```

## Stress Scenario

The `StressScenario` component spawns objects and applies deterministic motion to generate load. It also supports optional synthetic events:

- `syntheticEventsPerSecond`
- `allocateBytesPerEvent`
- `onSyntheticEvent` UnityEvent hook

You can use the prefab `Resources/Prefabs/StressScenario.prefab` as a starting point.

## Output Files

Files are written under `Application.persistentDataPath/benchmarks` by default:

- `benchmark_YYYYMMDD_HHMMSS_<Reason>.log`
- `benchmark_YYYYMMDD_HHMMSS_<Reason>.csv`

The absolute path is logged when files are created.

## CSV Format

The default binned CSV header is:
```
t_s,phase,frame_ms,fps,used_mem_mb,frame_ms_p95,frame_ms_max,spike_ratio,frames_in_bin,gc_alloc_bytes_mean,gc_alloc_bytes_max
```

Notes:
- `phase` is `warmup` or `measure`.
- If `csvIntervalSeconds <= 0`, per-frame rows are written instead of binned rows.

## Plotting (Sample Scripts)

Sample plotting helpers live in `Samples~`:

- `Samples~/plot.py`
- `Samples~/do_plot.sh`

Example:
```bash
python plot.py --csv /path/to/benchmark.csv --label Run1 --p95 --max
```

## For Developers

To work with this package, place it outside your Unity project (if it has its own git repo) or at the project root. Then include the package via **Window** -> **Package Manager** -> `+` -> **Add package from disk**, and select this project's `package.json`.

For production use, prefer **Add package from git URL** (above).

## License

AGPL-3.0-or-later. See `LICENSE`.

## Changelog

See `CHANGELOG.md`.
