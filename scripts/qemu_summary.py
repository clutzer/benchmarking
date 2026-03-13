#!/usr/bin/env python3
import subprocess
import re

def extract_vm_name(cmd_str):
    """Finds the value following the -name flag."""
    # Matches -name followed by the next word, handling potential quotes
    match = re.search(r'-name\s+([^-\s][\S]*)', cmd_str)
    if match:
        return match.group(1).strip('"\'').split(',')[0] # Split in case of sub-options like -name guest=...
    return "Unknown"

def reformat_qemu_cmd(cmd_str):
    # Split the command string while respecting quoted arguments
    parts = re.findall(r'(?:[^\s"\']|"(?:\\.|[^"])*"|\'(?:\\.|[^\'])*\')+', cmd_str)
    if not parts:
        return ""

    # Define groups for logical organization
    groups = {
        "Base Configuration": ["-name", "-machine", "-accel", "-cpu", "-smp", "-m", "-nodefaults", "-nographic", "-vga"],
        "Boot & Kernel": ["-kernel", "-append", "-initrd", "-bios", "-boot"],
        "Memory & NUMA": ["-object", "-numa"],
        "Storage": ["-drive", "-device", "-fsdev", "-virtfs"],
        "Networking": ["-netdev", "-net"],
        "Management & Misc": ["-chardev", "-mon", "-serial", "-D", "-pidfile", "-daemonize"]
    }

    formatted_lines = [f"{parts[0]} \\"]
    
    def get_group(arg):
        for name, flags in groups.items():
            if arg in flags: return name
        return "Other"

    buckets = {name: [] for name in groups.keys()}
    buckets["Other"] = []

    i = 1
    while i < len(parts):
        arg = parts[i]
        if arg.startswith('-') and i + 1 < len(parts) and not parts[i+1].startswith('-'):
            buckets[get_group(arg)].append(f"  {arg} {parts[i+1]} \\")
            i += 2
        else:
            buckets[get_group(arg)].append(f"  {arg} \\")
            i += 1

    for group_name, lines in buckets.items():
        if lines:
            formatted_lines.append(f"\n  # {group_name}")
            formatted_lines.extend(lines)

    res = "\n".join(formatted_lines).strip()
    return res[:-1].strip() if res.endswith("\\") else res

def main():
    try:
        # Get process list; 'ps -eo args' gives full command lines
        output = subprocess.check_output(["ps", "-eo", "args"], text=True)
        qemu_cmds = [line for line in output.splitlines() if "qemu-system-" in line and "grep" not in line]

        if not qemu_cmds:
            print("No running QEMU VMs found.")
            return

        for idx, cmd in enumerate(qemu_cmds, 1):
            vm_name = extract_vm_name(cmd)
            header = f" VM {idx} ({vm_name}) "
            print(f"{header:=^60}")
            print(reformat_qemu_cmd(cmd))
            print("\n")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

