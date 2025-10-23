#!/usr/bin/env python

import os
import json
import pandas as pd
from pathlib import Path
import argparse
import subprocess
import re

def load_profile_cache(cache_file="nim_profile_cache.json", verbose=False):
    """Load the Profile ID to human-readable name mapping from cache."""
    cache_file = Path(cache_file)
    if cache_file.exists():
        try:
            with cache_file.open('r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            if verbose:
                print(f"Warning: Failed to load cache file {cache_file}: {e}. Starting with empty cache.")
            return {}
    return {}

def save_profile_cache(profiles, cache_file="nim_profile_cache.json", verbose=False):
    """Save the Profile ID to human-readable name mapping to cache."""
    try:
        with open(cache_file, 'w') as f:
            json.dump(profiles, f, indent=2)
        if verbose:
            print(f"Updated profile cache at {cache_file}")
    except IOError as e:
        if verbose:
            print(f"Warning: Failed to save cache file {cache_file}: {e}")

def fetch_profile_id(verbose=False):
    """Fetch the NIM_MODEL_PROFILE (Profile ID) from the inference-server container."""
    try:
        result = subprocess.run(
            ["docker", "compose", "exec", "inference-server", "bash", "-c", "echo $NIM_MODEL_PROFILE"],
            capture_output=True,
            text=True,
            check=True
        )
        profile_id = result.stdout.strip()
        if profile_id:
            return profile_id
        else:
            if verbose:
                print("Warning: NIM_MODEL_PROFILE is empty in the container.")
            return None
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"Warning: Failed to fetch NIM_MODEL_PROFILE from container: {e}. Using 'Unknown'.")
        return None
    except FileNotFoundError:
        if verbose:
            print("Warning: 'docker' command not found. Using 'Unknown'.")
        return None

def fetch_model_profiles_output(verbose=False):
    """Fetch the raw output of 'list-model-profiles' from the container."""
    try:
        result = subprocess.run(
            ["docker", "compose", "exec", "inference-server", "bash", "-c", "list-model-profiles"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"Warning: Failed to fetch list-model-profiles: {e}. Cannot map Profile ID.")
        return None
    except FileNotFoundError:
        if verbose:
            print("Warning: 'docker' command not found. Cannot map Profile ID.")
        return None

def parse_model_profiles(output):
    """Parse the list-model-profiles output to map Profile IDs to human-readable names."""
    if not output:
        return {}

    # Regex to match: - <64-char hash> (<profile>)
    matches = re.findall(r'-\s+([a-f0-9]{64})\s+\(([^)]+)\)', output)

    profiles = {}
    for profile_id, profile_name in matches:
        # Truncate after first colon
        truncated_name = profile_name.split(':', 1)[0]
        profiles[profile_id] = truncated_name

    if not profiles and verbose:
        print("Warning: No profiles parsed from list-model-profiles output.")

    return profiles

def get_nim_model_profile(use_docker=True, override=None, cache_file="nim_profile_cache.json", verbose=False):
    """Get the human-readable NIM Model Profile, mapping from ID if needed."""
    # Load cache
    profile_cache = load_profile_cache(cache_file, verbose)

    if override:
        # Truncate override if it contains a colon
        return override.split(':', 1)[0]

    if not use_docker:
        return "Unknown"

    profile_id = fetch_profile_id(verbose)
    if not profile_id:
        return "Unknown"

    # If it's already a long descriptive name (not a hash), truncate and use it
    if len(profile_id) > 64 or not re.match(r'^[a-f0-9]{64}$', profile_id):
        if verbose:
            print(f"Using provided NIM_MODEL_PROFILE directly: {profile_id}")
        return profile_id.split(':', 1)[0]

    # Check cache first
    if profile_id in profile_cache:
        if verbose:
            print(f"Using cached profile for ID '{profile_id}': {profile_cache[profile_id]}")
        return profile_cache[profile_id]

    # Fetch and parse list-model-profiles to map ID
    output = fetch_model_profiles_output(verbose)
    profiles_map = parse_model_profiles(output)

    if profile_id in profiles_map:
        human_readable = profiles_map[profile_id]
        # Update cache with new mapping
        profile_cache[profile_id] = human_readable
        save_profile_cache(profile_cache, cache_file, verbose)
        if verbose:
            print(f"Mapped Profile ID '{profile_id}' to: {human_readable}")
        return human_readable
    else:
        if verbose:
            print(f"Warning: Profile ID '{profile_id}' not found in list-model-profiles. Using raw ID.")
        return profile_id

def extract_metrics(json_file, nim_profile, use_case, concurrency):
    """Extract relevant metrics from a genai_perf JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        ttft = data['time_to_first_token']['avg']  # ms
        tps = data['output_token_throughput']['avg']  # tokens/sec
        model = data['input_config']['model_names'][0]

        return {
            'Model': model,
            'NIM Model Profile': nim_profile,
            'Use Case': use_case,
            'Concurrency': concurrency,
            'TTFT (ms)': round(ttft, 2),
            'TPS (tokens/sec)': round(tps, 2)
        }
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error processing {json_file}: {e}")
        return None

def collate_benchmarks(artifact_dir, nim_profile):
    """Collate benchmark data from all genai_perf JSON files."""
    results = []

    # Walk through the artifact directory
    for root, _, files in os.walk(artifact_dir):
        # Skip if no concurrency in dir name (safety check)
        dir_name = os.path.basename(root)
        if 'concurrency' not in dir_name:
            continue

        # Extract concurrency from directory name (e.g., concurrency200 -> 200)
        match = re.search(r'concurrency(\d+)', dir_name)
        concurrency = int(match.group(1)) if match else None

        # Process each genai_perf JSON file
        for file in files:
            if file.endswith('_genai_perf.json'):
                # Extract use case from file name (e.g., 200_5)
                use_case = file.replace('_genai_perf.json', '')
                json_file = os.path.join(root, file)

                # Extract metrics
                metrics = extract_metrics(json_file, nim_profile, use_case, concurrency)
                if metrics:
                    results.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(results)
    # Sort by Model, NIM Model Profile, Concurrency, and Use Case for readability
    if not df.empty:
        df = df.sort_values(['Model', 'NIM Model Profile', 'Concurrency', 'Use Case'])
    return df

def print_markdown_table(df):
    """Print DataFrame as a Markdown table."""
    if df.empty:
        print("No valid data found to display.")
    else:
        print(df.to_markdown(index=False))

def main():
    """Main function to handle command-line arguments and run collation."""
    parser = argparse.ArgumentParser(description="Collate genai-perf benchmark results into a summary table.")
    parser.add_argument(
        "--artifact-dir",
        type=str,
        required=True,
        help="Path to the artifact directory containing benchmark JSON files."
    )
    parser.add_argument(
        "--nim-profile",
        type=str,
        default=None,
        help="Optional: Hardcode the NIM Model Profile (Profile ID or full name; overrides Docker fetch)."
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Enable verbose output for debugging."
    )
    args = parser.parse_args()

    # Validate artifact directory
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        print(f"Error: {artifact_dir} is not a valid directory.")
        return

    # Fetch or use NIM Model Profile (with mapping and cache)
    if args.verbose:
        print("Fetching NIM Model Profile...")
    nim_profile = get_nim_model_profile(use_docker=True, override=args.nim_profile, verbose=args.verbose)
    if args.verbose:
        print(f"Using NIM Model Profile: {nim_profile}")

    # Collate benchmarks and print results
    df = collate_benchmarks(artifact_dir, nim_profile)
    print_markdown_table(df)

if __name__ == "__main__":
    main()
