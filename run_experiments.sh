# for size in 1MB 10MB 100MB 1GB 10GB
for size in 1MB 10MB
do
    echo "Running join_scenario_experiment for $size"
    python -m integration_new.join_scenario_experiment $size &> join_scenario_experiment_$size.log
done