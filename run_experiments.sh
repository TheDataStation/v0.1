for size in 1MB 10MB 100MB 1GB 10GB
# for size in 10GB
do
    echo "Running join_scenario_experiment for $size"
    python -m integration_new.join_scenario_experiment $size 10 &> join_scenario_experiment_$size.log
done
