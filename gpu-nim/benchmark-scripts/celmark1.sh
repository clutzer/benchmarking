#!/bin/bash

HF_TOKEN=$(cat /secrets/hf-token.txt)
hf auth login --token $HF_TOKEN

declare -A useCases
 
# Populate the array with use case descriptions and their specified input/output lengths
useCases["Translation"]="200/200"
#useCases["Text classification"]="200/5"
#useCases["Text summary"]="1000/200"
#useCases["Code generation"]="200/1000"
#useCases["RAG"]="7000/1000"

 
# Function to execute genAI-perf with the input/output lengths as arguments
runBenchmark() {
    local description="$1"
    local lengths="${useCases[$description]}"
    IFS='/' read -r inputLength outputLength <<< "$lengths"
 
    echo "Running genAI-perf for $description with input length $inputLength and output length $outputLength"

    for concurrency in 1; do
 
        local INPUT_SEQUENCE_LENGTH=$inputLength
        local INPUT_SEQUENCE_STD=0
        local OUTPUT_SEQUENCE_LENGTH=$outputLength
        local CONCURRENCY=$concurrency
	local MODEL=$(curl -s http://inference-server:8000/v1/models | jq -r '.data[0].id')

        genai-perf profile \
            -m $MODEL \
            --endpoint-type chat \
            --streaming \
            -u inference-server:8000 \
            --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
            --synthetic-input-tokens-stddev $INPUT_SEQUENCE_STD \
            --concurrency $CONCURRENCY \
            --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
            --extra-inputs max_tokens:$OUTPUT_SEQUENCE_LENGTH \
            --extra-inputs min_tokens:$OUTPUT_SEQUENCE_LENGTH \
            --extra-inputs ignore_eos:true \
            --profile-export-file ${INPUT_SEQUENCE_LENGTH}_${OUTPUT_SEQUENCE_LENGTH}.json \
            -- \
            -v \
            --max-threads=1000
     
    done
}
 
# Iterate over all defined use cases and run the benchmark script for each
for description in "${!useCases[@]}"; do
    runBenchmark "$description"
done
