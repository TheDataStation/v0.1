# for size in 1MB 10MB 100MB 1GB 10GB
for size in 1MB 10MB
do
    python -m integration_new.join_scenario_experiment $size
done