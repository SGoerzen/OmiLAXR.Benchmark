for N in 0 1 10 100 300 600 800 850 875 900 1000 3000 5000 10000; do
  python plot.py \
    --csv "benchmark_Baseline_N${N}.csv" --label "Baseline N=${N}" \
    --csv "benchmark_OmiLAXR_N${N}.csv"  --label "OmiLAXR N=${N}" \
    --outdir "plots/N${N}" \
    --phase all \
    --tmin -30 --tmax 120 \
    --p95 \
    --smooth 11 --smooth_method rolling --smooth_center
done