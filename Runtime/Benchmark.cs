/*
* SPDX-License-Identifier: AGPL-3.0-or-later
* Copyright (C) 2025 Sergej Görzen <sergej.goerzen@gmail.com>
* This file is part of OmiLAXR.
*/
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using OmiLAXR.Endpoints;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Events;

namespace OmiLAXR.Benchmark
{
    /// <summary>
    /// Benchmark runner that performs warmup + measurement, collects frame stats,
    /// and writes optional CSV/log outputs for analysis.
    /// </summary>
    public class Benchmark : MonoBehaviour
    {
        /// <summary>
        /// Singleton instance for easy access from other systems.
        /// Creates a persistent GameObject if no instance exists.
        /// </summary>
        private static Benchmark _instance;
        /// <summary>Global singleton accessor.</summary>
        public static Benchmark Instance
        {
            get
            {
                if (_instance != null) return _instance;

                _instance = FindFirstObjectByType<Benchmark>();
                if (_instance != null) return _instance;

                var go = new GameObject("Benchmark");
                _instance = go.AddComponent<Benchmark>();
                DontDestroyOnLoad(go);
                return _instance;
            }
        }

        [Header("Warmup")]
        [Tooltip("Warmup duration before measurement starts (seconds).")]
        /// <summary>Warmup duration before measurement starts (seconds).</summary>
        public float warmupSeconds = 30f;

        [Header("Session")]
        [Tooltip("Start warmup+measurement automatically when enabled.")]
        /// <summary>Start warmup + measurement automatically on enable.</summary>
        public bool autoStartOnEnable = true;

        [Tooltip("Automatically end MEASUREMENT after X seconds (0 = disabled). Warmup is not counted).")]
        /// <summary>Automatically end measurement after X seconds (0 = disabled).</summary>
        public float autoQuitAfterSeconds = 120f;

        [Tooltip("If true: stop play mode in Editor. In builds: Application.Quit().")]
        /// <summary>Stop play mode in Editor or quit application in builds.</summary>
        public bool stopPlayModeInEditor = true;

        [Tooltip("If true, prints a short line periodically (also during warmup with negative t).")]
        /// <summary>Enable periodic logging (including warmup).</summary>
        public bool periodicLog = false;

        [Tooltip("Periodic log interval (seconds).")]
        /// <summary>Interval for periodic logs in seconds.</summary>
        public float periodicLogIntervalSeconds = 5f;

        [Header("Spike detection")]
        [Tooltip("Count frames whose frame-time exceeds this threshold (ms). Set 0 to disable.")]
        /// <summary>Frame time threshold in ms for spike counting (0 = disabled).</summary>
        public float frameSpikeThresholdMs = 0f;

        [Header("Hotkeys")]
        /// <summary>Enable hotkeys for manual actions.</summary>
        public bool enableHotkeys = true;
        /// <summary>Key to trigger a manual dump.</summary>
        public KeyCode dumpKey = KeyCode.F10;

        [Header("File logging")]
        [Tooltip("Write a summary log file on session end.")]
        /// <summary>Write a summary log file on session end.</summary>
        public bool writeLogFile = true;

        [Tooltip("Optional subfolder under Application.persistentDataPath.")]
        /// <summary>Optional subfolder under Application.persistentDataPath.</summary>
        public string logSubfolder = "benchmarks";

        [Tooltip("Filename prefix for .log files.")]
        /// <summary>Filename prefix for .log files.</summary>
        public string logFilePrefix = "benchmark";

        [Header("CSV logging")]
        [Tooltip("Write a CSV file with sampled (time-based) measurements.")]
        /// <summary>Write a CSV file with sampled measurements.</summary>
        public bool writeCsv = true;

        [Tooltip("Filename prefix for .csv files (saved next to the .log).")]
        /// <summary>Filename prefix for .csv files.</summary>
        public string csvFilePrefix = "benchmark";

        [Tooltip("Write one CSV row every X seconds (recommended; makes CSV size independent of FPS).")]
        /// <summary>Write one CSV row every X seconds.</summary>
        public float csvIntervalSeconds = 1f;

        [Tooltip("Fallback: write one CSV row every N frames (1 = every frame). Used only if csvIntervalSeconds <= 0.")]
        /// <summary>Fallback: write one CSV row every N frames.</summary>
        public int csvEveryNFrames = 1;

        [Header("Hooks")]
        /// <summary>Invoked when measurement starts.</summary>
        public UnityEvent onTrackingStart;
        /// <summary>Invoked when measurement stops.</summary>
        public UnityEvent onTrackingStop;

        /// <summary>
        /// Per-metric timing data for custom measurements.
        /// </summary>
        private sealed class MetricData
        {
            /// <summary>Whether a metric is currently running.</summary>
            public bool running;
            /// <summary>Start ticks for the current sample.</summary>
            public long startTicks;
            /// <summary>Collected sample ticks for this metric.</summary>
            public readonly List<long> samplesTicks = new();
            /// <summary>Minimum sample ticks.</summary>
            public long minTicks = long.MaxValue;
            /// <summary>Maximum sample ticks.</summary>
            public long maxTicks = long.MinValue;
            /// <summary>Sum of sample ticks.</summary>
            public long sumTicks = 0;
        }

        private readonly Dictionary<string, MetricData> _metrics = new(StringComparer.Ordinal);

        
        private bool _capturing;
        private long _frameCount;

        private double _sumFrameMs;
        private double _minFrameMs = double.MaxValue;
        private double _maxFrameMs = double.MinValue;
        private long _frameSpikeCount;

        
        private double _sumFps;
        private double _minFps = double.MaxValue;
        private double _maxFps = double.MinValue;

        private double _sumUsedMemMb;
        private double _minUsedMemMb = double.MaxValue;
        private double _maxUsedMemMb = double.MinValue;

        private float _nextPeriodicLogTime;
        private string _lastReport = "";

        
        
        private double _t0MeasurementStartRealtime;

        
        private ProfilerRecorder _frameTimeNs;
        private ProfilerRecorder _totalUsedMemBytes;
        private ProfilerRecorder _gcAllocInFrameBytes;

        private bool _hasFrameTimeRecorder;
        private bool _hasMemRecorder;
        private bool _hasGcAllocRecorder;

        
        private StreamWriter _csv;
        private string _lastCsvPathAbs = "";

        
        private double _csvWindowStartT = double.NaN;
        private string _csvWindowPhase = "warmup";
        private int _csvWindowFrames = 0;

        private double _csvSumFrameMs = 0;
        private double _csvMaxFrameMs = double.MinValue;

        private double _csvSumFps = 0;
        private double _csvSumMemMb = 0;

        private double _csvSumGcAllocBytes = 0;
        private double _csvMaxGcAllocBytes = double.MinValue;

        private int _csvSpikeCount = 0;

        
        private readonly List<double> _csvFrameMsSamples = new(512);

        
        private int _csvFrameIndex = 0;

        
        /// <summary>Unity Awake hook. Initializes singleton and default settings.</summary>
        private void Awake()
        {
            QualitySettings.vSyncCount = 0;   
            Application.targetFrameRate = 120;
            if (_instance != null && _instance != this)
            {
                Destroy(gameObject);
                return;
            }
            _instance = this;
            DontDestroyOnLoad(gameObject);
        }

        /// <summary>Unity enable hook. Optionally starts warmup + measurement.</summary>
        private void OnEnable()
        {
            if (autoStartOnEnable)
                StartCoroutine(WarmupThenMeasure());
        }

        /// <summary>Unity disable hook. Stops capture and closes CSV.</summary>
        private void OnDisable()
        {
            StopCapture();
            CloseCsvIfOpen();
        }

        /// <summary>Runs warmup, then starts measurement and CSV logging.</summary>
        private IEnumerator WarmupThenMeasure()
        {
            
            var now = Time.realtimeSinceStartupAsDouble;
            _t0MeasurementStartRealtime = now + Math.Max(0.0, warmupSeconds);

            
            OpenCsvIfEnabled("AutoRun");

            
            StopCapture();

            if (warmupSeconds > 0f)
                UnityEngine.Debug.Log($"Benchmark warmup started ({warmupSeconds:F0}s). Measurement starts at t=0.");

            yield return new WaitForSecondsRealtime(Math.Max(0f, warmupSeconds));

            
            onTrackingStart?.Invoke();
            UnityEngine.Debug.Log("Benchmark measurement started (t≈0).");

            StartCapture();

            
            ResetCsvWindow();
        }

        /// <summary>Per-frame update for sampling and periodic logging.</summary>
        private void Update()
        {
            if (enableHotkeys && Input.GetKeyDown(dumpKey))
            {
                DumpAll();
            }

            
            var t = Time.realtimeSinceStartupAsDouble - _t0MeasurementStartRealtime;

            
            var frameMs = TryGetFrameTimeMs(out var ms) ? ms : (Time.unscaledDeltaTime * 1000.0);
            var fps = frameMs > 0.0001 ? (1000.0 / frameMs) : 0.0;
            var usedMb = TryGetUsedMemoryMb(out var memMb) ? memMb : 0.0;
            var gcAllocBytes = TryGetGcAllocInFrameBytes(out var gcBytes) ? gcBytes : 0L;

            
            if (periodicLog && Time.unscaledTime >= _nextPeriodicLogTime)
            {
                _nextPeriodicLogTime = Time.unscaledTime + Mathf.Max(0.5f, periodicLogIntervalSeconds);
                UnityEngine.Debug.Log(GetShortSessionLine(t));
            }

            
            if (_csv != null)
            {
                WriteCsvSample(t, frameMs, fps, usedMb, gcAllocBytes);
            }

            if (!_capturing) return;

            
            if (autoQuitAfterSeconds > 0f && t >= autoQuitAfterSeconds)
            {
                EndSessionAndQuit("AutoQuit");
                return;
            }

            
            _frameCount++;
            _sumFrameMs += frameMs;
            if (frameMs < _minFrameMs) _minFrameMs = frameMs;
            if (frameMs > _maxFrameMs) _maxFrameMs = frameMs;

            if (frameSpikeThresholdMs > 0f && frameMs > frameSpikeThresholdMs)
                _frameSpikeCount++;

            _sumFps += fps;
            if (fps > 0.0 && fps < _minFps) _minFps = fps;
            if (fps > _maxFps) _maxFps = fps;

            if (usedMb > 0.0)
            {
                _sumUsedMemMb += usedMb;
                if (usedMb < _minUsedMemMb) _minUsedMemMb = usedMb;
                if (usedMb > _maxUsedMemMb) _maxUsedMemMb = usedMb;
            }
        }

        /// <summary>Unity application quit hook for final flush.</summary>
        private void OnApplicationQuit()
        {
            if (_capturing)
                EndSession("OnApplicationQuit");
        }

        
        /// <summary>Starts a custom metric timer by name.</summary>
        /// <param name="name">Metric name</param>
        public static void StartRecord(string name) => Instance.StartMetric(name);
        /// <summary>Stops a custom metric timer by name.</summary>
        /// <param name="name">Metric name</param>
        public static void StopRecord(string name) => Instance.StopMetric(name);

        /// <summary>Ends the session and writes reports/logs.</summary>
        public static void DumpAll() => Instance.EndSession("ManualDump");
        /// <summary>Resets all benchmark state and metrics.</summary>
        public static void ResetAll() => Instance.ResetState();

        
        /// <summary>Starts measurement capture and initializes counters.</summary>
        private void StartCapture()
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 120;

            if (_capturing) return;
            _capturing = true;

            UnityEngine.Debug.Log("Started Benchmark capture.");
            ResetSessionStats();

            _hasFrameTimeRecorder = TryStartRecorder(ProfilerCategory.Internal, "Frame Time", out _frameTimeNs);
            _hasMemRecorder = TryStartRecorder(ProfilerCategory.Memory, "Total Used Memory", out _totalUsedMemBytes);
            _hasGcAllocRecorder = TryStartRecorder(ProfilerCategory.Memory, "GC Allocated In Frame", out _gcAllocInFrameBytes);

            _nextPeriodicLogTime = Time.unscaledTime + Mathf.Max(0.5f, periodicLogIntervalSeconds);
        }

        /// <summary>Stops measurement capture and closes recorders.</summary>
        private void StopCapture()
        {
            if (!_capturing) return;
            _capturing = false;

            if (_hasFrameTimeRecorder) _frameTimeNs.Dispose();
            if (_hasMemRecorder) _totalUsedMemBytes.Dispose();
            if (_hasGcAllocRecorder) _gcAllocInFrameBytes.Dispose();

            _hasFrameTimeRecorder = false;
            _hasMemRecorder = false;
            _hasGcAllocRecorder = false;
        }

        /// <summary>Attempts to start a profiler recorder for a given stat.</summary>
        /// <param name="cat">Profiler category</param>
        /// <param name="statName">Stat name</param>
        /// <param name="recorder">Recorder output</param>
        /// <returns>True if recorder started successfully</returns>
        private static bool TryStartRecorder(ProfilerCategory cat, string statName, out ProfilerRecorder recorder)
        {
            try
            {
                recorder = ProfilerRecorder.StartNew(cat, statName);
                return recorder.Valid;
            }
            catch
            {
                recorder = default;
                return false;
            }
        }

        
        /// <summary>Opens the CSV writer if CSV logging is enabled.</summary>
        /// <param name="reasonTag">Tag used in the filename</param>
        private void OpenCsvIfEnabled(string reasonTag)
        {
            if (!writeCsv) return;
            if (_csv != null) return;

            try
            {
                var dir = Path.Combine(Application.persistentDataPath, logSubfolder);
                Directory.CreateDirectory(dir);

                var file = $"{csvFilePrefix}_{DateTime.Now:yyyyMMdd_HHmmss}_{Sanitize(reasonTag)}.csv";
                var path = Path.Combine(dir, file);

                _lastCsvPathAbs = Path.GetFullPath(path);
                _csv = new StreamWriter(path, false, Encoding.UTF8);

                
                _csv.WriteLine("t_s,phase,frame_ms,fps,used_mem_mb,frame_ms_p95,frame_ms_max,spike_ratio,frames_in_bin,gc_alloc_bytes_mean,gc_alloc_bytes_max");
                _csv.Flush();

                UnityEngine.Debug.Log($"Benchmark CSV will be written to (absolute): {_lastCsvPathAbs}");

                ResetCsvWindow();
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning($"Benchmark could not open CSV: {e.Message}");
                _csv = null;
                _lastCsvPathAbs = "";
            }
        }

        /// <summary>Closes the CSV writer if open.</summary>
        private void CloseCsvIfOpen()
        {
            
            try
            {
                if (_csv != null && _csvWindowFrames > 0)
                {
                    FlushCsvWindowRow(force: true);
                }
            }
            catch {  }

            try
            {
                _csv?.Flush();
                _csv?.Dispose();
            }
            catch {  }
            finally
            {
                _csv = null;
            }
        }

        /// <summary>Resets the time-based CSV window accumulators.</summary>
        private void ResetCsvWindow()
        {
            _csvWindowStartT = double.NaN;
            _csvWindowPhase = _capturing ? "measure" : "warmup";
            _csvWindowFrames = 0;

            _csvSumFrameMs = 0;
            _csvMaxFrameMs = double.MinValue;

            _csvSumFps = 0;
            _csvSumMemMb = 0;

            _csvSumGcAllocBytes = 0;
            _csvMaxGcAllocBytes = double.MinValue;

            _csvSpikeCount = 0;

            _csvFrameMsSamples.Clear();
            _csvFrameIndex = 0;
        }

        /// <summary>Writes a single CSV sample row.</summary>
        /// <param name="t">Time in seconds</param>
        /// <param name="frameMs">Frame time in ms</param>
        /// <param name="fps">Frames per second</param>
        /// <param name="usedMb">Used memory in MB</param>
        /// <param name="gcAllocBytes">GC alloc bytes in frame</param>
        private void WriteCsvSample(double t, double frameMs, double fps, double usedMb, long gcAllocBytes)
        {
            
            if (csvIntervalSeconds > 0f)
            {
                
                if (double.IsNaN(_csvWindowStartT))
                {
                    _csvWindowStartT = t;
                    _csvWindowPhase = _capturing ? "measure" : "warmup";
                }

                
                var phaseNow = _capturing ? "measure" : "warmup";
                if (!string.Equals(phaseNow, _csvWindowPhase, StringComparison.Ordinal))
                {
                    FlushCsvWindowRow(force: true);
                    ResetCsvWindow();
                    _csvWindowStartT = t;
                    _csvWindowPhase = phaseNow;
                }

                
                _csvWindowFrames++;
                _csvSumFrameMs += frameMs;
                _csvSumFps += fps;
                if (usedMb > 0) _csvSumMemMb += usedMb;

                _csvSumGcAllocBytes += gcAllocBytes;
                if (gcAllocBytes > _csvMaxGcAllocBytes) _csvMaxGcAllocBytes = gcAllocBytes;

                if (frameMs > _csvMaxFrameMs) _csvMaxFrameMs = frameMs;
                _csvFrameMsSamples.Add(frameMs);

                if (frameSpikeThresholdMs > 0f && frameMs > frameSpikeThresholdMs)
                    _csvSpikeCount++;

                
                if ((t - _csvWindowStartT) >= csvIntervalSeconds)
                {
                    FlushCsvWindowRow(force: false);
                    ResetCsvWindow();
                    
                    _csvWindowStartT = t;
                    _csvWindowPhase = phaseNow;
                }

                return;
            }

            
            if (csvEveryNFrames < 1) csvEveryNFrames = 1;

            if ((_csvFrameIndex % csvEveryNFrames) == 0)
            {
                var phase = _capturing ? "measure" : "warmup";
                _csv.WriteLine($"{t:F3},{phase},{frameMs:F3},{fps:F1},{usedMb:F1},,,,,{gcAllocBytes},");
                _csv.Flush();
            }

            _csvFrameIndex++;
        }

        /// <summary>Flushes the current CSV aggregation window.</summary>
        /// <param name="force">Force a flush even if interval not reached</param>
        private void FlushCsvWindowRow(bool force)
        {
            if (_csv == null) return;
            if (_csvWindowFrames <= 0) return;

            var meanFrame = _csvSumFrameMs / _csvWindowFrames;
            var meanFps = _csvSumFps / _csvWindowFrames;
            var meanMem = (_csvSumMemMb > 0) ? (_csvSumMemMb / _csvWindowFrames) : 0.0;

            var meanGc = _csvSumGcAllocBytes / _csvWindowFrames;
            var maxGc = (_csvMaxGcAllocBytes > double.MinValue) ? _csvMaxGcAllocBytes : 0.0;

            
            double p95 = 0.0;
            if (_csvFrameMsSamples.Count > 0)
            {
                _csvFrameMsSamples.Sort();
                var idx = (int)Math.Ceiling(0.95 * (_csvFrameMsSamples.Count - 1));
                idx = Mathf.Clamp(idx, 0, _csvFrameMsSamples.Count - 1);
                p95 = _csvFrameMsSamples[idx];
            }

            var spikeRatio = 0.0;
            if (frameSpikeThresholdMs > 0f)
                spikeRatio = (double)_csvSpikeCount / Math.Max(1, _csvWindowFrames);

            
            
            var tRow = _csvWindowStartT + Math.Max(0.0, csvIntervalSeconds);

            _csv.WriteLine(
                $"{tRow:F3},{_csvWindowPhase},{meanFrame:F3},{meanFps:F1},{meanMem:F1},{p95:F3},{_csvMaxFrameMs:F3},{spikeRatio:F4},{_csvWindowFrames},{meanGc:F0},{maxGc:F0}"
            );

            
            _csv.Flush();
        }

        
        /// <summary>Ends the session and quits play mode or app.</summary>
        /// <param name="reason">Reason for ending session</param>
        private void EndSessionAndQuit(string reason)
        {
            EndSession(reason);

#if UNITY_EDITOR
            if (stopPlayModeInEditor)
            {
                UnityEditor.EditorApplication.isPlaying = false;
                return;
            }
#endif
            Application.Quit();
        }

        /// <summary>Ends the session and writes reports/logs.</summary>
        /// <param name="reason">Reason for ending session</param>
        private void EndSession(string reason)
        {
            onTrackingStop?.Invoke();
            StopCapture();

            DumpReportToConsoleAndMaybeFile(reason);

            CloseCsvIfOpen();
            if (!string.IsNullOrEmpty(_lastCsvPathAbs))
                UnityEngine.Debug.Log($"Benchmark CSV written to (absolute): {_lastCsvPathAbs}");
        }

        
        /// <summary>Starts timing a custom metric.</summary>
        /// <param name="name">Metric name</param>
        private void StartMetric(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) name = "Unnamed";

            if (!_metrics.TryGetValue(name, out var m))
            {
                m = new MetricData();
                _metrics[name] = m;
            }

            if (m.running)
            {
                UnityEngine.Debug.LogWarning($"Benchmark.StartRecord(\"{name}\") called while already running. Ignored.");
                return;
            }

            m.running = true;
            m.startTicks = Stopwatch.GetTimestamp();
        }

        /// <summary>Stops timing a custom metric.</summary>
        /// <param name="name">Metric name</param>
        private void StopMetric(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) name = "Unnamed";

            if (!_metrics.TryGetValue(name, out var m) || !m.running)
            {
                UnityEngine.Debug.LogWarning($"Benchmark.StopRecord(\"{name}\") called but metric is not running.");
                return;
            }

            var endTicks = Stopwatch.GetTimestamp();
            var deltaTicks = endTicks - m.startTicks;

            m.running = false;
            m.samplesTicks.Add(deltaTicks);
            m.sumTicks += deltaTicks;
            if (deltaTicks < m.minTicks) m.minTicks = deltaTicks;
            if (deltaTicks > m.maxTicks) m.maxTicks = deltaTicks;
        }

        
        /// <summary>Formats a report and writes to console and optional file.</summary>
        /// <param name="reason">Reason tag for the report</param>
        private void DumpReportToConsoleAndMaybeFile(string reason)
        {
            _lastReport = BuildReport(reason);
            UnityEngine.Debug.Log(_lastReport);

            if (writeLogFile)
            {
                try
                {
                    var dir = Path.Combine(Application.persistentDataPath, logSubfolder);
                    Directory.CreateDirectory(dir);

                    var file = $"{logFilePrefix}_{DateTime.Now:yyyyMMdd_HHmmss}_{Sanitize(reason)}.log";
                    var path = Path.Combine(dir, file);

                    File.WriteAllText(path, _lastReport, Encoding.UTF8);

                    var absPath = Path.GetFullPath(path);
                    UnityEngine.Debug.Log($"Benchmark log written to (absolute): {absPath}");
                }
                catch (Exception e)
                {
                    UnityEngine.Debug.LogWarning($"Benchmark could not write log file: {e.Message}");
                }
            }
        }

        private string BuildReport(string reason)
        {
            var sb = new StringBuilder(4096);
            sb.AppendLine("========== Benchmark Report ==========");
            sb.AppendLine($"Reason: {reason}");
            sb.AppendLine($"Time: {DateTime.Now:O}");
            sb.AppendLine();

            sb.AppendLine("Paths:");
            sb.AppendLine($"- persistentDataPath: {Application.persistentDataPath}");
            sb.AppendLine();

            sb.AppendLine("System:");
            sb.AppendLine($"- Unity: {Application.unityVersion}");
            sb.AppendLine($"- Platform: {Application.platform}");
            sb.AppendLine($"- DeviceModel: {SystemInfo.deviceModel}");
            sb.AppendLine($"- OS: {SystemInfo.operatingSystem}");
            sb.AppendLine($"- CPU: {SystemInfo.processorType} ({SystemInfo.processorCount} cores)");
            sb.AppendLine($"- RAM: {SystemInfo.systemMemorySize} MB");
            sb.AppendLine($"- GPU: {SystemInfo.graphicsDeviceName} ({SystemInfo.graphicsMemorySize} MB)");
            var endpoints = FindObjectsByType<Endpoint>(sortMode: FindObjectsSortMode.None, findObjectsInactive: FindObjectsInactive.Include);
            foreach (var endpoint in endpoints)
            {
                sb.AppendLine($"- Endpoint: {endpoint.GetType().Name} recorded {endpoint.RecordedStatements}");
            }
            if (endpoints.Length < 1)
                sb.AppendLine("- No endpoints found. Thus 0 statements in total recorded.");

            sb.AppendLine();

            sb.AppendLine("Session:");
            sb.AppendLine(GetSessionSummaryLine());
            sb.AppendLine();

            if (_metrics.Count == 0)
            {
                sb.AppendLine("Custom metrics: none recorded.");
            }
            else
            {
                sb.AppendLine("Custom metrics (StartRecord/StopRecord durations):");
                foreach (var kvp in _metrics)
                {
                    var name = kvp.Key;
                    var m = kvp.Value;

                    var n = m.samplesTicks.Count;
                    if (n == 0)
                    {
                        sb.AppendLine($"- {name}: 0 samples");
                        continue;
                    }

                    var avgMs = TicksToMs((double)m.sumTicks / n);
                    var minMs = TicksToMs(m.minTicks);
                    var maxMs = TicksToMs(m.maxTicks);

                    sb.AppendLine($"- {name}: n={n}, avg={avgMs:F3}ms, min={minMs:F3}ms, max={maxMs:F3}ms");
                }
            }

            sb.AppendLine("=====================================");
            return sb.ToString();
        }

        /// <summary>Sanitizes a string for filenames.</summary>
        /// <param name="s">Input string</param>
        /// <returns>Sanitized string</returns>
        private static string Sanitize(string s)
        {
            if (string.IsNullOrEmpty(s)) return "NA";
            foreach (var c in Path.GetInvalidFileNameChars())
                s = s.Replace(c, '_');
            return s;
        }

        
        private string GetSessionSummaryLine()
        {
            if (_frameCount == 0)
                return "Session: no measurement frames captured yet.";

            var avgFrame = _sumFrameMs / _frameCount;

            var spikes = frameSpikeThresholdMs > 0f
                ? $" | spikes>{frameSpikeThresholdMs:F1}ms: {_frameSpikeCount}"
                : "";

            var avgFps = _sumFps / _frameCount;
            var fpsPart = $" | FPS avg={avgFps:F1} min={_minFps:F1} max={_maxFps:F1}";

            var memPart = (_sumUsedMemMb > 0.0 && _minUsedMemMb < double.MaxValue)
                ? $" | UsedMem(MB) avg={(_sumUsedMemMb / _frameCount):F1} min={_minUsedMemMb:F1} max={_maxUsedMemMb:F1}"
                : " | UsedMem(MB) n/a";

            
            var t = Time.realtimeSinceStartupAsDouble - _t0MeasurementStartRealtime;
            if (t < 0) t = 0;

            return $"t={t:F1}s | Frames: {_frameCount} | Frame(ms) avg={avgFrame:F3} min={_minFrameMs:F3} max={_maxFrameMs:F3}{spikes}{fpsPart}{memPart}";
        }

        private string GetShortSessionLine(double t)
        {
            
            if (!_capturing)
                return $"[Warmup] t={t:F1}s (0 = measurement start)";

            if (_frameCount == 0) return $"Benchmark: t={t:F1}s waiting for frames...";

            var avgFrame = _sumFrameMs / _frameCount;
            var avgFps = _sumFps / _frameCount;

            return $"Benchmark: t={t:F1}s | Frame avg={avgFrame:F3}ms (min={_minFrameMs:F3} max={_maxFrameMs:F3})"
                 + $" | FPS avg={avgFps:F1} (min={_minFps:F1} max={_maxFps:F1})"
                 + $" | UsedMem avg={(_sumUsedMemMb > 0 ? (_sumUsedMemMb / _frameCount) : 0):F1}MB";
        }

        /// <summary>Resets all internal state and counters.</summary>
        private void ResetState()
        {
            _metrics.Clear();
            ResetSessionStats();
        }

        /// <summary>Resets per-session aggregates.</summary>
        private void ResetSessionStats()
        {
            _frameCount = 0;
            _sumFrameMs = 0;
            _minFrameMs = double.MaxValue;
            _maxFrameMs = double.MinValue;
            _frameSpikeCount = 0;

            _sumFps = 0;
            _minFps = double.MaxValue;
            _maxFps = double.MinValue;

            _sumUsedMemMb = 0;
            _minUsedMemMb = double.MaxValue;
            _maxUsedMemMb = double.MinValue;

            _csvFrameIndex = 0;
            ResetCsvWindow();
        }

        
        private bool TryGetFrameTimeMs(out double frameMs)
        {
            if (_hasFrameTimeRecorder && _frameTimeNs.Valid)
            {
                var ns = _frameTimeNs.LastValue;
                if (ns > 0)
                {
                    frameMs = ns / 1_000_000.0;
                    return true;
                }
            }
            frameMs = 0;
            return false;
        }

        private bool TryGetUsedMemoryMb(out double usedMb)
        {
            if (_hasMemRecorder && _totalUsedMemBytes.Valid)
            {
                var bytes = _totalUsedMemBytes.LastValue;
                if (bytes > 0)
                {
                    usedMb = bytes / (1024.0 * 1024.0);
                    return true;
                }
            }

            try
            {
                var bytes = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong();
                usedMb = bytes > 0 ? bytes / (1024.0 * 1024.0) : 0.0;
                return bytes > 0;
            }
            catch
            {
                usedMb = 0.0;
                return false;
            }
        }

        private bool TryGetGcAllocInFrameBytes(out long bytes)
        {
            if (_hasGcAllocRecorder && _gcAllocInFrameBytes.Valid)
            {
                bytes = _gcAllocInFrameBytes.LastValue;
                return true;
            }

            bytes = 0;
            return false;
        }

        /// <summary>Converts tick counts to milliseconds.</summary>
        /// <param name="ticks">Tick count</param>
        /// <returns>Milliseconds</returns>
        private static double TicksToMs(double ticks)
            => (ticks * 1000.0) / Stopwatch.Frequency;
    }
}
