#!/usr/bin/env python3

import argparse
import subprocess
import re
import glob
import os
import sys
from collections import defaultdict


def read_command_output(command, mock_file):
    use_mock = os.getenv('PCI_TOPOLOGY_USE_MOCK', '').lower() in ('1', 'true', 'yes', 'on')
    if use_mock:
        with open(mock_file, 'r', encoding='utf-8') as f:
            return f.read()

    return subprocess.check_output(command).decode('utf-8')


def is_relevant_device_text(text):
    t = text.lower()
    return any(keyword in t for keyword in [
        'nvidia', 'rtx', 'blackwell',
        'mellanox', 'connectx', 'bluefield',
        'ethernet', 'network', 'infiniband',
    ])


def infer_device_type(text):
    t = text.lower()
    if any(keyword in t for keyword in ['nvidia', 'rtx', 'blackwell', '3d controller', 'display controller']):
        return 'GPU'
    return 'NIC'


def parse_lspci_tv_output():
    try:
        output = read_command_output(['lspci', '-tv'], 'firefly-lspci-tv.txt')
    except Exception as e:
        raise RuntimeError(f"Failed to get lspci -tv output (or mock file firefly-lspci-tv.txt): {e}")

    root_map = defaultdict(list)
    root_order = []
    current_root = None
    seen = set()
    last_terminal_bus_by_root = {}
    last_branch_key_by_root = {}

    root_re = re.compile(r'[+\\-]-\[([0-9a-f]{4}):([0-9a-f]{2})\]', re.IGNORECASE)
    seg_re = re.compile(r'([0-9a-f]{2}\.[0-7])(?:-\[([0-9a-f]{2})(?:-[0-9a-f]{2})?\])?', re.IGNORECASE)
    range_re = re.compile(r'\[([0-9a-f]{2}(?:-[0-9a-f]{2})?)\]', re.IGNORECASE)

    for raw_line in output.splitlines():
        line = raw_line.rstrip('\n')
        if not line.strip():
            continue

        root_match = root_re.search(line)
        if root_match:
            current_root = f"{root_match.group(1).lower()}:{root_match.group(2).lower()}"
            if current_root not in root_map:
                root_order.append(current_root)

        line_match = re.search(r'^(.*?)([0-9a-f]{2}\.[0-7])\s{2,}(.+)$', line, re.IGNORECASE)
        if not line_match or current_root is None:
            continue

        description = line_match.group(3).strip()
        if not is_relevant_device_text(description):
            continue

        path_text = f"{line_match.group(1)}{line_match.group(2)}"
        segments = list(seg_re.finditer(path_text))
        if not segments:
            continue

        domain, root_bus = current_root.split(':')
        ranges = range_re.findall(path_text.lower())
        if ranges:
            terminal_bus = ranges[-1].split('-')[0]
            last_terminal_bus_by_root[current_root] = terminal_bus
            branch_key = ' -> '.join(ranges)
            last_branch_key_by_root[current_root] = branch_key
            bus_ctx = root_bus
        else:
            bus_ctx = last_terminal_bus_by_root.get(current_root, root_bus)
            branch_key = last_branch_key_by_root.get(current_root, root_bus)

        bdf = None
        for seg in segments:
            devfn = seg.group(1).lower()
            bdf = f"{domain}:{bus_ctx}:{devfn}"
            next_bus = seg.group(2)
            if next_bus:
                bus_ctx = next_bus.lower()

        if bdf is None:
            continue

        sig = (current_root, bdf, description)
        if sig in seen:
            continue
        seen.add(sig)

        root_map[current_root].append({
            'Slot': bdf,
            'full_name': description,
            'type': infer_device_type(description),
            'IOMMUGroup': 'N/A',
            'Branch': branch_key,
        })

    return root_map, root_order


def get_numa_mappings_from_vmm():
    try:
        output = read_command_output(['lspci', '-Dvmmvvv'], 'firefly-lspci-Dvmvvv.txt')
    except Exception:
        return {}, {}

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

    slot_to_numa = {}
    root_to_numa = {}
    for dev in devices:
        slot = dev.get('Slot')
        if not slot:
            continue

        numa = dev.get('NUMANode', 'N/A')
        if numa == '-1':
            numa = '0'
        slot_to_numa[slot] = numa

        class_name = dev.get('Class', '')
        device_name = dev.get('Device', '')
        if class_name.startswith('Host bridge') and 'Root Complex' in device_name:
            root_key = ':'.join(slot.split(':')[:2]).lower()
            root_to_numa[root_key] = numa

    return slot_to_numa, root_to_numa


def print_topology_from_tv(root_map, root_order, slot_to_numa, root_to_numa):
    print("PCIe topology from lspci -tv (simplified)")

    roots_by_numa = defaultdict(list)
    for root in root_order:
        devices = root_map.get(root, [])
        if not devices:
            continue

        numa = root_to_numa.get(root)
        if not numa and devices:
            numa = slot_to_numa.get(devices[0]['Slot'])
        if not numa:
            numa = 'N/A'

        roots_by_numa[numa].append(root)

    def numa_sort_key(numa_key):
        if numa_key == 'N/A':
            return (1, 10**9)
        try:
            return (0, int(numa_key))
        except ValueError:
            return (0, 10**9 - 1)

    for numa in sorted(roots_by_numa.keys(), key=numa_sort_key):
        print(f"NUMA Node {numa} (CPU Socket {numa})")
        for root in roots_by_numa[numa]:
            devices = root_map.get(root, [])
            if not devices:
                continue

            print(f"├── [RC] {root}")
            branch_map = defaultdict(list)
            for dev in devices:
                branch_map[dev['Branch']].append(dev)

            for idx, branch in enumerate(branch_map.keys(), start=1):
                print(f"│   ├── [SW] Branch {idx} [{branch}]")
                for dev in branch_map[branch]:
                    print(f"│   │   ├── {dev['type']}: {dev['Slot']} {dev['full_name']}")
    print()

def get_pretty_names():
    try:
        output = read_command_output(['lspci', '-Dmm'], 'firefly-lspci-Dmm.txt')
    except Exception as e:
        print(f"Warning: could not get lspci -Dmm output (or mock file firefly-lspci-Dmm.txt): {e}", file=sys.stderr)
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

def parse_lspci_output(pretty_names):
    try:
        # Use lspci -Dvmmvvv for detailed info including PhySlot if available
        output = read_command_output(['lspci', '-Dvmmvvv'], 'firefly-lspci-Dvmvvv.txt')
    except Exception as e:
        raise RuntimeError(f"Failed to get lspci -Dvmmvvv output (or mock file firefly-lspci-Dvmvvv.txt): {e}")

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

    # Discover root-complex anchor buses per NUMA node
    root_buses_by_numa = defaultdict(list)
    for dev in devices:
        slot = dev.get('Slot')
        class_name = dev.get('Class', '')
        device_name = dev.get('Device', '')
        numa = dev.get('NUMANode', 'N/A')
        if not slot or numa == 'N/A':
            continue
        if class_name.startswith('Host bridge') and 'Root Complex' in device_name:
            bus = int(slot.split(':')[1], 16)
            root_buses_by_numa[numa].append(bus)

    for numa in list(root_buses_by_numa.keys()):
        root_buses_by_numa[numa] = sorted(set(root_buses_by_numa[numa]))

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

            # Get NUMA node (prefer lspci field in mock/captured data)
            numa = dev.get('NUMANode', 'N/A')
            if numa == '-1':
                numa = '0'
            if numa == 'N/A' and os.path.exists(sys_path):
                try:
                    with open(f"{sys_path}/numa_node", 'r') as f:
                        numa = f.read().strip()
                        if numa == '-1':
                            numa = '0'  # Default or handle unbound
                except:
                    numa = 'N/A'
            dev['NUMANode'] = numa

            # Get IOMMU group (prefer lspci field in mock/captured data)
            iommu_group = dev.get('IOMMUGroup', 'N/A')
            if iommu_group == 'N/A' and os.path.exists('/sys/kernel/iommu_groups'):
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

            # Map endpoint to nearest root-complex bus within the same NUMA node
            bus = int(slot.split(':')[1], 16)
            root_candidates = root_buses_by_numa.get(numa, [])
            if root_candidates:
                lower_or_equal = [r for r in root_candidates if r <= bus]
                chosen_root = max(lower_or_equal) if lower_or_equal else min(root_candidates)
            else:
                chosen_root = bus
            dev['RootBus'] = chosen_root
            dev['RootDomain'] = slot.split(':')[0]

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
    numa_groups = defaultdict(lambda: defaultdict(list))
    for dev in devices:
        numa = dev['NUMANode']
        if numa != 'N/A':
            root_bus = dev.get('RootBus', int(dev['Slot'].split(':')[1], 16))
            numa_groups[numa][root_bus].append(dev)

    # Sort NUMA keys numerically
    sorted_numa = sorted(numa_groups.keys(), key=int)

    # For each NUMA and Root Complex, sort devices by bus number and group by consecutive buses
    for numa in sorted_numa:
        root_map = numa_groups[numa]
        normalized_root_groups = []
        for root_bus in sorted(root_map.keys()):
            devs = root_map[root_bus]
            devs.sort(key=lambda d: int(d['Slot'].split(':')[1], 16))

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

            normalized_root_groups.append({
                'root_bus': root_bus,
                'root_domain': devs[0].get('RootDomain', '0000') if devs else '0000',
                'bus_groups': bus_groups,
            })

        numa_groups[numa] = normalized_root_groups

    return numa_groups, sorted_numa

def print_topology(numa_groups, sorted_numa):
    for idx, numa in enumerate(sorted_numa):
        print(f"NUMA Node {numa} (CPU Socket {numa})")
        root_groups = numa_groups[numa]
        for root_idx, root in enumerate(root_groups):
            root_bus = root['root_bus']
            root_domain = root['root_domain']
            print(f"├── Root Complex [{root_domain}:{hex(root_bus)[2:].zfill(2)}]")

            groups = root['bus_groups']
            for g_idx, group in enumerate(groups):
                # Infer bus range
                buses = sorted(set(int(d['Slot'].split(':')[1], 16) for d in group))
                bus_str = '-'.join(hex(b)[2:].zfill(2) for b in [buses[0], buses[-1]]) if len(buses) > 1 else hex(buses[0])[2:].zfill(2)

                print(f"│   ├── Port Group {g_idx + 1} (buses {bus_str})")

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
                    iommu = dev['IOMMUGroup'] if len(mf_devs) == 1 else iommu

                    print(f"│   │   ├── {type_}: {slot} {full_name} (IOMMU: {iommu})")

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
    parser = argparse.ArgumentParser(description="Print PCI topology for GPUs/NICs")
    parser.add_argument(
        '--use-mock',
        action='store_true',
        help='Read lspci output from mock files instead of running system commands'
    )
    parser.add_argument(
        '--source',
        choices=['tv', 'vmm'],
        default='tv',
        help='Topology source: tv parses lspci -tv directly; vmm uses lspci -Dvmmvvv inference'
    )
    args = parser.parse_args()

    if args.use_mock:
        os.environ['PCI_TOPOLOGY_USE_MOCK'] = '1'

    if args.source == 'tv':
        root_map, root_order = parse_lspci_tv_output()
        slot_to_numa, root_to_numa = get_numa_mappings_from_vmm()
        print_topology_from_tv(root_map, root_order, slot_to_numa, root_to_numa)
    else:
        pretty_names = get_pretty_names()
        devices = parse_lspci_output(pretty_names)
        numa_groups, sorted_numa = group_by_numa_and_buses(devices)
        print_topology(numa_groups, sorted_numa)
