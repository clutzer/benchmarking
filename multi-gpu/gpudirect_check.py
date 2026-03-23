import subprocess
import os
import re
import sys

class GDRDiagnostics:
    def __init__(self):
        self.results = []
        self.failed = False

    def log(self, category, message, success=True):
        status = "[ OK ]" if success else "[FAIL]"
        print(f"{status} {category:15} : {message}")
        if not success: self.failed = True

    def run_cmd(self, cmd):
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
        except subprocess.CalledProcessError as e:
            return e.output.decode()

    def check_kernel_modules(self):
        modules = ["nvidia", "nvidia_uvm", "nvidia_peermem", "mlx5_core", "mlx5_ib", "ib_uverbs"]
        loaded = self.run_cmd("lsmod")
        for mod in modules:
            if mod in loaded:
                self.log("Kernel Module", f"{mod} is loaded")
            else:
                self.log("Kernel Module", f"{mod} is MISSING", False)

    def check_nvidia_gpu(self):
        out = self.run_cmd("nvidia-smi -q -d MEMORY")
        # Check BAR1 Size (Critical for Blackwell)
        bar1_match = re.search(r"BAR1 Memory Usage.*?Total\s+:\s+(\d+)\s+MiB", out, re.S)
        if bar1_match:
            total_bar1 = int(bar1_match.group(1))
            if total_bar1 < 32768: # Blackwell 6000 expects ~64GB, but check for >32GB
                self.log("NVIDIA GPU", f"BAR1 Size is low ({total_bar1} MiB). Ensure 'Above 4G Decoding' is enabled.", False)
            else:
                self.log("NVIDIA GPU", f"BAR1 Size is healthy ({total_bar1} MiB)")
        
        # Check Peer-to-Peer Topology
        topo = self.run_cmd("nvidia-smi topo -m")
        if "PXB" in topo or "PIX" in topo:
            self.log("NVIDIA Topo", "P2P/GPUDirect paths detected (PXB/PIX)")
        else:
            self.log("NVIDIA Topo", "No P2P paths detected (Only SYS/NODE). Check PCIe Switiching.", False)

    def check_rdma_state(self):
        dev_info = self.run_cmd("ibv_devinfo")
        if "PORT_ACTIVE" in dev_info:
            self.log("RDMA Port", "At least one HCA port is ACTIVE")
        else:
            self.log("RDMA Port", "All ports are DOWN or INITIALIZING", False)
            print("      --> Hint: Assign an IP address to the Mellanox Ethernet interface in the guest.")

    def check_network_config(self):
        # Map IB devices to Net devices
        mapping = self.run_cmd("ibdev2netdev")
        print("\n--- Device Mapping ---")
        print(mapping.strip())
        
        # Check for IP addresses on mlx5 interfaces
        ip_addr = self.run_cmd("ip -br addr show")
        for line in mapping.splitlines():
            parts = line.split()
            if len(parts) >= 5:
                net_dev = parts[-2]
                if net_dev in ip_addr and "UP" in ip_addr:
                    match = re.search(rf"{net_dev}\s+UP\s+(\d+\.\d+\.\d+\.\d+)", ip_addr)
                    if match:
                        self.log("Network", f"{net_dev} has IP {match.group(1)}")
                    else:
                        self.log("Network", f"{net_dev} is UP but has NO IP", False)

    def check_pci_topology(self):
        # Verify sibling relationship
        lspci = self.run_cmd("lspci -tv")
        # We look for the common bridge index from your previous output (03.0)
        if "03.0" in lspci:
            self.log("PCIe Topo", "Virtual PCIe Switch (03.0) found")
        else:
            self.log("PCIe Topo", "Could not find unified switch hierarchy", False)

    def run_all(self):
        print("=== GPUDirect RDMA System Health Check ===\n")
        self.check_kernel_modules()
        self.check_pci_topology()
        self.check_nvidia_gpu()
        self.check_rdma_state()
        self.check_network_config()
        print("\n==========================================")
        if self.failed:
            print("RESULT: System is NOT ready for GPUDirect RDMA.")
        else:
            print("RESULT: System appears READY for GPUDirect RDMA.")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Please run as root (sudo).")
        sys.exit(1)
    diag = GDRDiagnostics()
    diag.run_all()
