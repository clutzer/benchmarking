#!/bin/bash
# vim: set ts=4 sw=4 et:

HF_TOKEN=$(cat /secrets/hf-token.txt)
hf auth login --token $HF_TOKEN

declare -A useCases
 
# Populate the array with use case descriptions and their specified input/output lengths
useCases["translation"]="200/200"
useCases["text_classification"]="200/5"
useCases["text_summary"]="1000/200"
useCases["code_generation"]="200/1000"
 
rm -rf /artifacts/baseline/*

# Function to execute genAI-perf with the input/output lengths as arguments
runBenchmark() {
    local description="$1"
    local lengths="${useCases[$description]}"
    IFS='/' read -r inputLength outputLength <<< "$lengths"
 
    echo "Running genAI-perf for $description with input length $inputLength and output length $outputLength"
    #Runs
    for concurrency in 1 5 10 25 50 100 150 200; do
    #for concurrency in 1 100; do
 
        local INPUT_SEQUENCE_LENGTH=$inputLength
        local INPUT_SEQUENCE_STD=0
        local OUTPUT_SEQUENCE_LENGTH=$outputLength
        local CONCURRENCY=$concurrency
        local MODEL=$(curl -s http://inference-server:8000/v1/models | jq -r '.data[0].id')
        # Recommended measurement intervals by model...
        local MEASUREMENT_INTERVAL_8B=30000
        local MEASUREMENT_INTERVAL_70B=100000
        local MEASUREMENT_INTERVAL=60000
         
        genai-perf profile \
            -m $MODEL \
            --artifact-dir /artifacts/baseline \
            --concurrency $CONCURRENCY \
            --measurement-interval ${MEASUREMENT_INTERVAL} \
            --stability-percentage 10 \
            --endpoint-type chat \
            --streaming \
            -u inference-server:8000 \
            --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
            --synthetic-input-tokens-stddev $INPUT_SEQUENCE_STD \
            --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
            --extra-inputs max_tokens:$OUTPUT_SEQUENCE_LENGTH \
            --extra-inputs min_tokens:$OUTPUT_SEQUENCE_LENGTH \
            --extra-inputs ignore_eos:true \
            --profile-export-file ${INPUT_SEQUENCE_LENGTH}_${OUTPUT_SEQUENCE_LENGTH}.json \
            -- \
            -v \
            --max-threads=256
     
    done
}
 
# Iterate over all defined use cases and run the benchmark script for each
for description in "${!useCases[@]}"; do
    runBenchmark "$description"
done
