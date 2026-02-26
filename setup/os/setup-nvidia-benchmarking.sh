#!/bin/bash

# 1. Define variables at the very top (Global Scope)
USER_NAME="nvidia"
GROUP_NAME="nvidia"
TARGET_ID=1000

# [Insert your existing user creation logic here]

# --- Pre-positioning Logic with Safety Checks ---

# 2. Safety Check: Verify the user variable isn't empty
if [ -z "$USER_NAME" ]; then
    echo "Error: USER_NAME variable is empty. Script aborted."
    exit 1
fi

# 3. Safety Check: Verify the user actually exists in /etc/passwd
if ! id "$USER_NAME" >/dev/null 2>&1; then
    echo "Error: User '$USER_NAME' was not successfully created. Cannot clone."
    exit 1
fi

# 4. Identify the current Git remote URL
REPO_URL=$(git remote get-url origin)
TARGET_DIR="/home/$USER_NAME/benchmarking"

echo "Pre-positioning repo for user '$USER_NAME'..."

# 5. Clone and chown the repo...
git clone git@github.com:clutzer/benchmarking.git $TARGET_DIR
chown -R $USER_NAME:$GROUP_NAME $TARGET_DIR

echo "Repo successfully cloned to $TARGET_DIR"

