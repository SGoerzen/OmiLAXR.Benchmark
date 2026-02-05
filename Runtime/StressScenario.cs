using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using Random = System.Random;


namespace OmiLAXR.Benchmark
{
    public class StressScenario : MonoBehaviour
    {
        [Header("Spawn")] [Tooltip("How many objects to spawn.")] [Min(0)]
        public int objectCount = 200;

        [Tooltip("Deterministic seed (same seed => same motion).")]
        public int seed = 12345;

        public GameObject primitivePrefab;

        [Tooltip("If true, create visible primitives. If false, create empty GameObjects (transforms only).")]
        public bool createPrimitives = true;

        [Tooltip("If true, disable Colliders on spawned primitives (recommended).")]
        public bool disableColliders = true;

        [Tooltip("If true, disable MeshRenderer on spawned primitives (keeps transform but removes GPU draw).")]
        public bool disableRenderer = false;

        [Tooltip("Optional tag to assign to spawned objects (empty = keep default).")]
        public string setTag = "";

        [Tooltip("Parent spawned objects under this StressScenario transform.")]
        public bool parentUnderThis = true;

        [Tooltip("Spawn in a centered grid around this object.")]
        public Vector3 gridSpacing = new Vector3(0.25f, 0.25f, 0.25f);

        [Tooltip("Max grid extents (approx). Increase if you spawn many objects and want less overlap.")]
        public Vector3 gridExtents = new Vector3(5f, 2f, 5f);

        [Tooltip("Uniform scale for spawned objects.")] [Min(0.001f)]
        public float uniformScale = 0.08f;

        [Header("Motion")] [Tooltip("Master enable for motion.")]
        public bool enableMotion = true;

        [Tooltip("Amplitude of movement (meters).")] [Min(0f)]
        public float amplitude = 0.15f;

        [Tooltip("Base motion frequency (Hz). Each object gets a deterministic multiplier.")] [Min(0f)]
        public float baseFrequencyHz = 0.6f;

        [Tooltip("If true, also rotate objects.")]
        public bool enableRotation = false;

        [Tooltip("Degrees per second base rotation speed.")] [Min(0f)]
        public float baseRotationDegPerSec = 45f;

        [Tooltip("Use unscaled time (recommended for consistent benchmarking).")]
        public bool useUnscaledTime = true;

        [Header("Synthetic Events (optional)")]
        [Tooltip("Generate synthetic 'events' per second to stress instrumentation/serialization pipeline.")]
        [Min(0f)]
        public float syntheticEventsPerSecond = 0f;

        [Tooltip("Optional GC pressure per event (bytes allocated). 0 = no extra allocation.")] [Min(0)]
        public int allocateBytesPerEvent = 0;

        [Tooltip("Metric name used by Benchmark.StartRecord/StopRecord around synthetic event dispatch.")]
        public string syntheticMetricName = "SyntheticEvent";

        [Tooltip("Invoked for each synthetic event (index, t). Hook your instrumentation here if desired.")]
        public UnityEvent<int, float> onSyntheticEvent;

        [Header("Lifecycle")] [Tooltip("Auto respawn on Start().")]
        public bool spawnOnStart = true;

        [Tooltip("If true, respawn when values change in Editor (OnValidate).")]
        public bool respawnOnValidateInEditor = false;


        private readonly List<Transform> _targets = new();
        private Vector3[] _basePos;
        private Vector3[] _axes;
        private float[] _phase;
        private float[] _freqMul;
        private Vector3[] _rotAxis;
        private float _eventAccumulator;
        private bool _spawned;

        private void Start()
        {
            if (spawnOnStart)
                Respawn();
        }

        private void OnDisable()
        {
        }

#if UNITY_EDITOR
        private void OnValidate()
        {
            if (!respawnOnValidateInEditor) return;
            if (!Application.isPlaying) return;
            Respawn();
        }
#endif

        [ContextMenu("Respawn")]
        public void Respawn()
        {
            Clear();
            Spawn();
        }

        [ContextMenu("Clear")]
        public void Clear()
        {
            for (var i = 0; i < _targets.Count; i++)
            {
                if (_targets[i] != null)
                    Destroy(_targets[i].gameObject);
            }

            _targets.Clear();
            _basePos = null;
            _axes = null;
            _phase = null;
            _freqMul = null;
            _rotAxis = null;
            _eventAccumulator = 0f;
            _spawned = false;
        }

        private void Spawn()
        {
            if (objectCount <= 0)
            {
                _spawned = true;
                return;
            }

            _basePos = new Vector3[objectCount];
            _axes = new Vector3[objectCount];
            _phase = new float[objectCount];
            _freqMul = new float[objectCount];
            _rotAxis = new Vector3[objectCount];

            var rnd = new Random(seed);


            var nx = Mathf.Max(1, Mathf.FloorToInt(gridExtents.x / Mathf.Max(0.001f, gridSpacing.x)) * 2 + 1);
            var ny = Mathf.Max(1, Mathf.FloorToInt(gridExtents.y / Mathf.Max(0.001f, gridSpacing.y)) * 2 + 1);
            var nz = Mathf.Max(1, Mathf.FloorToInt(gridExtents.z / Mathf.Max(0.001f, gridSpacing.z)) * 2 + 1);

            var idx = 0;
            for (var i = 0; i < objectCount; i++)
            {
                GameObject go;
                if (createPrimitives)
                {
                    go = Instantiate(primitivePrefab, transform);
                    if (disableColliders)
                    {
                        var col = go.GetComponent<Collider>();
                        if (col != null) col.enabled = false;
                    }

                    if (disableRenderer)
                    {
                        var mr = go.GetComponent<MeshRenderer>();
                        if (mr != null) mr.enabled = false;
                    }
                }
                else
                {
                    go = new GameObject("StressObj");
                }

                go.name = $"StressObj_{i:00000}";
                if (!string.IsNullOrWhiteSpace(setTag))
                {
                    try
                    {
                        go.tag = setTag;
                    }
                    catch
                    {
                    }
                }

                if (parentUnderThis)
                    go.transform.SetParent(transform, worldPositionStays: false);

                go.transform.localScale = Vector3.one * uniformScale;


                var gx = idx % nx;
                var gy = (idx / nx) % ny;
                var gz = (idx / (nx * ny)) % nz;
                idx++;

                var centered = new Vector3(
                    (gx - (nx - 1) * 0.5f) * gridSpacing.x,
                    (gy - (ny - 1) * 0.5f) * gridSpacing.y,
                    (gz - (nz - 1) * 0.5f) * gridSpacing.z
                );

                go.transform.localPosition = centered;

                _targets.Add(go.transform);
                _basePos[i] = centered;


                _axes[i] = RandomUnitVector(rnd);
                _phase[i] = (float)(rnd.NextDouble() * Math.PI * 2.0);
                _freqMul[i] = 0.5f + (float)rnd.NextDouble() * 2.0f;
                _rotAxis[i] = RandomUnitVector(rnd);
            }

            _spawned = true;
        }

        private void Update()
        {
            if (!_spawned) return;

            var dt = useUnscaledTime ? Time.unscaledDeltaTime : Time.deltaTime;
            var t = useUnscaledTime ? Time.unscaledTime : Time.time;


            if (syntheticEventsPerSecond > 0f)
            {
                _eventAccumulator += syntheticEventsPerSecond * dt;
                var emitCount = Mathf.FloorToInt(_eventAccumulator);
                if (emitCount > 0)
                    _eventAccumulator -= emitCount;

                for (var i = 0; i < emitCount; i++)
                {
                    if (allocateBytesPerEvent > 0)
                    {
                        var bytes = new byte[allocateBytesPerEvent];
                        bytes[0] = 1;
                    }


                    Benchmark.StartRecord(syntheticMetricName);
                    onSyntheticEvent?.Invoke(i, t);
                    Benchmark.StopRecord(syntheticMetricName);
                }
            }


            if (!enableMotion || objectCount <= 0) return;

            var baseOmega = 2f * Mathf.PI * baseFrequencyHz;

            for (var i = 0; i < _targets.Count; i++)
            {
                var tr = _targets[i];
                if (tr == null) continue;

                var omega = baseOmega * _freqMul[i];
                var s = Mathf.Sin(t * omega + _phase[i]);


                tr.localPosition = _basePos[i] + _axes[i] * (amplitude * s);

                if (enableRotation)
                {
                    var rotSpeed = baseRotationDegPerSec * _freqMul[i];
                    tr.localRotation = Quaternion.AngleAxis((t * rotSpeed) % 360f, _rotAxis[i]);
                }
            }
        }

        private static Vector3 RandomUnitVector(Random rnd)
        {
            var x = (float)(rnd.NextDouble() * 2.0 - 1.0);
            var y = (float)(rnd.NextDouble() * 2.0 - 1.0);
            var z = (float)(rnd.NextDouble() * 2.0 - 1.0);
            var v = new Vector3(x, y, z);
            if (v.sqrMagnitude < 1e-6f) v = Vector3.right;
            return v.normalized;
        }
    }
}