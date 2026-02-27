#!/usr/bin/env python3

import subprocess
import re
import glob
import os
from collections import defaultdict
import re

def get_pretty_names():
    try:
        output = subprocess.check_output(['lspci', '-Dmm']).decode('utf-8')
    except Exception as e:
        print(f"Warning: could not run lspci -Dmm: {e}", file=sys.stderr)
        return {}

    name_map = {}

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        # Custom tokenizer: extract slot + quoted fields + flags
        tokens = []
        i = 0
        while i < len(line):
            if line[i].isspace():
                i += 1
                continue
            if line[i] == '"':
                # Find matching closing quote
                j = line.find('"', i + 1)
                if j == -1:
                    j = len(line)
                token = line[i + 1 : j].strip()  # remove surrounding quotes, strip extra ws
                tokens.append(token)
                i = j + 1
            else:
                # Non-quoted token (slot or -r01, -p00, etc.)
                j = i
                while j < len(line) and not line[j].isspace():
                    j += 1
                token = line[i:j].strip()
                tokens.append(token)
                i = j

        if len(tokens) < 4:
            continue  # skip malformed lines

        slot = tokens[0]
        # tokens[1] = class (e.g. "Host bridge", "3D controller")
        vendor = tokens[2]
        device = tokens[3]

        # Use the descriptive device name when it looks good
        pretty = device

        # Fallback if device name is generic ("Device xxxx")
        if 'Device' in pretty and '[' not in pretty:
            pretty = f"{vendor} {device}"

        # If subsystem fields exist and look better (often the case for add-in cards)
        if len(tokens) >= 7:
            subsys_vendor = tokens[5]
            subsys_device = tokens[6]
            if '[' in subsys_device or len(subsys_device) > len(device):
                pretty = subsys_device  # e.g. board-level name

        name_map[slot] = pretty

    # Diagnostic print (keep this for now)
    sorted_slots = sorted(name_map.keys())
    if False:
        print("\n=== Diagnostic: pretty_names contents (first 30 + relevant) ===")
        for slot in sorted_slots[:30]:  # limit to avoid flooding terminal
            print(f"  {slot} → {name_map[slot]!r}")

    print("\nRelevant devices (GPU/NIC likely):")
    for slot in sorted_slots:
        name = name_map[slot]
        if any(kw.lower() in name.lower() or kw in slot for kw in ['blackwell', 'rtx', 'nvidia', 'bluefield', 'connectx', 'mt4', 'a2dc', 'c2d5', '101e', '2bb5']):
            print(f"  {slot} → {name!r}")
    print("===================================================\n")

    return name_map

def get_pretty_names_DELETE():
    try:
        output = subprocess.check_output(['lspci', '-Dmm']).decode('utf-8')
    except Exception as e:
        print(f"Warning: could not run lspci -Dmm: {e}", file=sys.stderr)
        return {}

    name_map = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        # Format: domain:slot "class" "vendor" "device" [-rrev] [-pprogif] ["subsys vendor" "subsys device"]
        parts = re.split(r'\s+(?=(?:"[^"]*")|\S)', line.strip())
        if len(parts) < 4:
            continue

        slot = parts[0]
        # class_ = parts[1].strip('"')   # optional, we already have type
        vendor = parts[2].strip('"')
        device = parts[3].strip('"')

        # Use device name if it looks descriptive (contains [ ] or longer than generic)
        pretty = device
        if '[' not in pretty and 'Device' in pretty:
            pretty = f"{vendor} {device}"

        # Optional: if subsystem is present and looks better, use it
        if len(parts) >= 6 and parts[5].startswith('"') and parts[6].startswith('"'):
            subsys_vendor = parts[5].strip('"')
            subsys_device = parts[6].strip('"')
            if len(subsys_device) > len(device) or '[' in subsys_device:
                pretty = subsys_device  # e.g. sometimes subsystem has board name

        name_map[slot] = pretty

    return name_map

def parse_lspci_output(pretty_names):
    try:
        # Use lspci -Dvmmvvv for detailed info including PhySlot if available
        output = subprocess.check_output(['lspci', '-Dvmmvvv']).decode('utf-8')
    except Exception as e:
        raise RuntimeError("Failed to run lspci: ensure you have permissions (may need sudo).")

    devices = []
    current = {}
    for line in output.splitlines():
        if not line.strip() and current:
            devices.append(current)
            current = {}
            continue
        if ':' in line:
            key, value = line.split(':', 1)
            current[key.strip()] = value.strip()
    if current:
        devices.append(current)

    # Filter for relevant devices (GPUs and NICs) and add sysfs info
    relevant_devices = []
    for dev in devices:
        if 'Class' not in dev or 'Slot' not in dev:
            continue
        class_name = dev['Class']
        if class_name.startswith('3D controller') or class_name.startswith('Display controller') or \
           'Ethernet' in class_name or 'Network' in class_name or 'InfiniBand' in class_name:
            slot = dev['Slot']
            sys_path = f"/sys/bus/pci/devices/{slot}"
            if not os.path.exists(sys_path):
                continue  # Skip if sysfs not found

            # Get NUMA node
            try:
                with open(f"{sys_path}/numa_node", 'r') as f:
                    numa = f.read().strip()
                    if numa == '-1':
                        numa = '0'  # Default or handle unbound
            except:
                numa = 'N/A'
            dev['NUMANode'] = numa

            # Get IOMMU group
            iommu_group = 'N/A'
            for group_dir in glob.glob('/sys/kernel/iommu_groups/*'):
                group = os.path.basename(group_dir)
                devices_dir = os.path.join(group_dir, 'devices')
                if os.path.exists(os.path.join(devices_dir, slot)):
                    iommu_group = group
                    break
            dev['IOMMUGroup'] = iommu_group

            # Infer type
            if class_name.startswith('3D') or class_name.startswith('Display'):
                dev['type'] = 'GPU'
            else:
                dev['type'] = 'NIC'

            # Use full vendor/device name from lspci
            vendor = dev.get('Vendor', 'Unknown')
            device = dev.get('Device', 'Unknown')
            if 'SVendor' in dev and 'SDevice' in dev:
                device = f"{dev['SVendor']} {dev['SDevice']}"
            #dev['full_name'] = f"{vendor} {device}"
            dev['full_name'] = pretty_names.get(slot, f"{dev.get('Vendor', 'Unknown')} {dev.get('Device', 'Unknown')}")

            # PhySlot if present
            phy_slot = dev.get('Physical Slot', 'N/A').replace('#', '')
            dev['PhySlot'] = phy_slot

            relevant_devices.append(dev)

    return relevant_devices

def group_by_numa_and_buses(devices):
    numa_groups = defaultdict(list)
    for dev in devices:
        numa = dev['NUMANode']
        if numa != 'N/A':
            numa_groups[numa].append(dev)

    # Sort NUMA keys numerically
    sorted_numa = sorted(numa_groups.keys(), key=int)

    # For each NUMA, sort devices by bus number (hex to int)
    for numa in sorted_numa:
        devs = numa_groups[numa]
        devs.sort(key=lambda d: int(d['Slot'].split(':')[1], 16))

        # Group by consecutive buses
        bus_groups = []
        current_group = []
        prev_bus = -2
        for dev in devs:
            bus = int(dev['Slot'].split(':')[1], 16)
            if bus == prev_bus + 1 or not current_group:
                current_group.append(dev)
                prev_bus = bus
            else:
                bus_groups.append(current_group)
                current_group = [dev]
                prev_bus = bus
        if current_group:
            bus_groups.append(current_group)
        numa_groups[numa] = bus_groups

    return numa_groups, sorted_numa

def print_topology(numa_groups, sorted_numa):
    for idx, numa in enumerate(sorted_numa):
        print(f"NUMA Node {numa} (CPU Socket {numa})")
        groups = numa_groups[numa]
        for g_idx, group in enumerate(groups):
            # Infer bus range
            buses = sorted(set(int(d['Slot'].split(':')[1], 16) for d in group))
            bus_str = '-'.join(hex(b)[2:].zfill(2) for b in [buses[0], buses[-1]]) if len(buses) > 1 else hex(buses[0])[2:].zfill(2)

            print(f"├── Root Complex / Port Group {g_idx + 1}")
            print(f"│   └── PCIe Switch {g_idx} (Inferred from consecutive buses {bus_str})")

            # Group multi-function devices (same bus:dev, different func)
            mf_groups = defaultdict(list)
            for dev in group:
                bus_dev = ':'.join(dev['Slot'].split(':')[:3])  # domain:bus:dev
                mf_groups[bus_dev].append(dev)

            for mf_key, mf_devs in mf_groups.items():
                if len(mf_devs) == 1:
                    dev = mf_devs[0]
                    funcs = ''
                else:
                    funcs = '/' + '/'.join(d['Slot'].split('.')[-1] for d in mf_devs)
                    dev = mf_devs[0]  # Use first for common info
                    iommu = '/'.join(d['IOMMUGroup'] for d in mf_devs)

                slot = mf_key + funcs
                type_ = dev['type']
                full_name = dev['full_name']
                phy = dev['PhySlot']
                iommu = dev['IOMMUGroup'] if len(mf_devs) == 1 else iommu

                #print(f"│       ├── {type_}: {slot} {full_name} (PhySlot: {phy}, IOMMU: {iommu})")
                print(f"│       ├── {type_}: {slot} {full_name} (IOMMU: {iommu})")

        if idx < len(sorted_numa) - 1:
            print("└── Inter-Socket Link (UPI/QPI/Infinity Fabric - spans NUMA for cross-node DMA, potentially slower P2P)")
        else:
            print()

    # Add key insights section
    print("### Key Insights for NVIDIA GPUDirect P2P/RDMA")
    print("- Within Same PCIe Switch (e.g., GPUs and NICs in the same group): Direct P2P via switch (optimal, \"PIX\" in nvidia-smi topo). No root complex involvement, minimal latency.")
    print("- Cross Switches, Same NUMA Node: P2P through root complex (viable but slower, \"PXB/PHB\"). DMA path stays within NUMA.")
    print("- Cross NUMA Nodes: P2P possible but routes through inter-socket link (e.g., UPI/QPI), spans NUMA (\"SYS\"). Reduced performance; may require IOMMU/ACS disabled for best results.")
    print("- GPUDirect RDMA (GPU-NIC): Employed when GPU and NIC share the same upstream root complex or switch (e.g., within a group). Cross-NUMA RDMA spans nodes, potentially suboptimal.")
    print("- Assumptions: Topology inferred from bus groupings (consecutive hex buses suggest shared switch). Unique IOMMU groups per device confirm isolation. Disable PCIe ACS for peak P2P perf (routes traffic to root otherwise).")

if __name__ == "__main__":
    pretty_names = get_pretty_names()
    # Diagnostic print
    #print("\n=== Diagnostic: pretty_names contents ===")
    #import pprint
    #pprint.pprint(pretty_names, width=120, compact=False)
    devices = parse_lspci_output(pretty_names)
    numa_groups, sorted_numa = group_by_numa_and_buses(devices)
    print_topology(numa_groups, sorted_numa)
