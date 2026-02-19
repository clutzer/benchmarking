#!/bin/bash

# Configuration
USER_NAME="nvidia"
TARGET_ID=1000

# Ensure the script is run as root
if [ "$EUID" -ne 0 ]; then 
  echo "Please run as root (sudo)."
  exit 1
fi

echo "Checking availability of UID/GID ${TARGET_ID}..."

# 1. Check if UID or GID 1000 is already taken by another user/group
if id -u "$TARGET_ID" >/dev/null 2>&1; then
    EXISTING_USER=$(getent passwd "$TARGET_ID" | cut -d: -f1)
    if [ "$EXISTING_USER" != "$USER_NAME" ]; then
        echo "Error: UID ${TARGET_ID} is already taken by user: ${EXISTING_USER}"
        exit 1
    fi
fi

if getent group "$TARGET_ID" >/dev/null 2>&1; then
    EXISTING_GROUP=$(getent group "$TARGET_ID" | cut -d: -f1)
    if [ "$EXISTING_GROUP" != "$USER_NAME" ]; then
        echo "Error: GID ${TARGET_ID} is already taken by group: ${EXISTING_GROUP}"
        exit 1
    fi
fi

# 2. Create the 'nvidia' group and user if they don't exist
if ! getent group "$USER_NAME" >/dev/null 2>&1; then
    echo "Creating group '${USER_NAME}' with GID ${TARGET_ID}..."
    groupadd -g "$TARGET_ID" "$USER_NAME"
fi

if ! id -u "$USER_NAME" >/dev/null 2>&1; then
    echo "Creating user '${USER_NAME}' with UID ${TARGET_ID}..."
    useradd -m -s /bin/bash -u "$TARGET_ID" -g "$TARGET_ID" "$USER_NAME"
else
    echo "User '${USER_NAME}' already exists."
fi

# 3. Final Verification
RESULT_UID=$(id -u "$USER_NAME")
RESULT_GID=$(id -g "$USER_NAME")

if [ "$RESULT_UID" -eq "$TARGET_ID" ] && [ "$RESULT_GID" -eq "$TARGET_ID" ]; then
    echo "Success: User '${USER_NAME}' verified with UID: ${RESULT_UID} and GID: ${RESULT_GID}."
else
    echo "Verification Failed! Got UID: ${RESULT_UID}, GID: ${RESULT_GID}. Expected ${TARGET_ID}."
    exit 1
fi

