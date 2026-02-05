/*
/*
* SPDX-License-Identifier: AGPL-3.0-or-later
* Copyright (C) 2025 Sergej Görzen <sergej.goerzen@gmail.com>
* This file is part of OmiLAXR.
#1#
using UnityEditor;
using UnityEngine;

namespace OmiLAXR.Benchmarking
{
    /// <summary>
    /// Editor window for configuring numeric parameters used by the benchmark plotting workflow.
    /// </summary>
    public sealed class BenchmarkWindow : EditorWindow
    {
        private const string KeyBinSeconds = "OmiLAXR.Benchmark.BinSeconds";
        private const string KeyDownsampleN = "OmiLAXR.Benchmark.DownsampleN";
        private const string KeySmoothWindowN = "OmiLAXR.Benchmark.SmoothWindowN";

        private float _binSeconds = 1.0f;
        private int _downsampleN = 1;
        private int _smoothWindowN = 0;

        /// <summary>
        /// Opens the benchmark window from the OmiLAXR menu.
        /// </summary>
        [MenuItem("OmiLAXR / Open Benchmark Manager", priority = 0)]
        public static void Open()
        {
            var window = GetWindow<BenchmarkWindow>();
            window.titleContent = new GUIContent("Benchmark");
            window.minSize = new Vector2(320f, 180f);
            window.Show();
        }

        private void OnEnable()
        {
            _binSeconds = EditorPrefs.GetFloat(KeyBinSeconds, 1.0f);
            _downsampleN = EditorPrefs.GetInt(KeyDownsampleN, 1);
            _smoothWindowN = EditorPrefs.GetInt(KeySmoothWindowN, 0);
        }

        private void OnDisable()
        {
            SavePrefs();
        }

        private void OnGUI()
        {
            GUILayout.Label("Plot Settings (Python)", EditorStyles.boldLabel);

            using (new EditorGUILayout.VerticalScope("box"))
            {
                _binSeconds = EditorGUILayout.FloatField("Bin Seconds (N)", _binSeconds);
                _downsampleN = EditorGUILayout.IntField("Downsample N", _downsampleN);
                _smoothWindowN = EditorGUILayout.IntField("Smooth Window N", _smoothWindowN);
            }

            _binSeconds = Mathf.Max(0.001f, _binSeconds);
            _downsampleN = Mathf.Max(1, _downsampleN);
            _smoothWindowN = Mathf.Max(0, _smoothWindowN);

            GUILayout.Space(8f);

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Save", GUILayout.Height(28f)))
                {
                    SavePrefs();
                }

                if (GUILayout.Button("Reset", GUILayout.Height(28f)))
                {
                    _binSeconds = 1.0f;
                    _downsampleN = 1;
                    _smoothWindowN = 0;
                    SavePrefs();
                }
            }
        }

        private void SavePrefs()
        {
            EditorPrefs.SetFloat(KeyBinSeconds, _binSeconds);
            EditorPrefs.SetInt(KeyDownsampleN, _downsampleN);
            EditorPrefs.SetInt(KeySmoothWindowN, _smoothWindowN);
        }
    }
}
*/
