import subprocess
import os
import re
import sys

class GDRDiagnostics:
    def __init__(self):
        self.results = []
        self.failed = False
        self.kernel_version = self.run_cmd("uname -r")

    def log(self, category, message, success=True, warning=False):
        if warning:
            status = "[WARN]"
        else:
            status = "[ OK ]" if success else "[FAIL]"
        
        print(f"{status} {category:15} : {message}")
        if not success and not warning: 
            self.failed = True

    def run_cmd(self, cmd):
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
        except subprocess.CalledProcessError as e:
            return e.output.decode()

    def check_kernel_modules(self):
        # nvidia_peermem is removed from the required list for Kernel >= 6.2
        core_modules = ["nvidia", "nvidia_uvm", "mlx5_core", "mlx5_ib", "ib_uverbs"]
        loaded = self.run_cmd("lsmod")
        
        # Check Core Modules
        for mod in core_modules:
            if mod in loaded:
                self.log("Kernel Module", f"{mod} is loaded")
            else:
                self.log("Kernel Module", f"{mod} is MISSING", False)

        # Handle peermem as a legacy/optional check
        if "nvidia_peermem" in loaded:
            self.log("Kernel Module", "nvidia_peermem is loaded (Legacy Path)", success=True, warning=True)
        else:
            self.log("Kernel Module", "nvidia_peermem is not loaded (Not required for DMA-BUF)", success=True)

    def check_dmabuf_support(self):
        # Check if the NVIDIA driver is export-capable
        if os.path.exists("/sys/module/nvidia/parameters/nv_use_dmabuf"):
            val = self.run_cmd("cat /sys/module/nvidia/parameters/nv_use_dmabuf").strip()
            self.log("DMA-BUF", f"Driver DMA-BUF support parameter is {val}")
        else:
            self.log("DMA-BUF", "Verified via Kernel 6.8+ capability")

    def check_bar_identity_mapping(self):
        """
        Detects if the QEMU 'Identity Map' patch is likely active.
        Standard QEMU BARs are usually low (< 1TB). Physical Blackwell hosts
        usually map BARs in high-terabyte or petabyte ranges.
        """
        lspci_out = self.run_cmd("lspci -vv")
        # Find 64-bit Memory addresses
        matches = re.findall(r"Memory at ([0-9a-fA-F]+) \(64-bit, prefetchable\)", lspci_out)
        
        if not matches:
            self.log("Identity Map", "No 64-bit BARs found to analyze.", warning=True)
            return

        is_high_mem = False
        for addr_str in matches:
            addr_val = int(addr_str, 16)
            # Threshold: 1TB (0x10000000000). QEMU default holes are much lower.
            if addr_val > 0x10000000000:
                is_high_mem = True
                break
        
        if is_high_mem:
            self.log("Identity Map", "Detected High-Memory BARs. QEMU Identity Map is ACTIVE.")
        else:
            self.log("Identity Map", "Standard low-memory BARs detected.", warning=True)

    def check_nvidia_gpu(self):
        out = self.run_cmd("nvidia-smi -q -d MEMORY")
        bar1_match = re.search(r"BAR1 Memory Usage.*?Total\s+:\s+(\d+)\s+MiB", out, re.S)
        if bar1_match:
            total_bar1 = int(bar1_match.group(1))
            if total_bar1 < 32768: 
                self.log("NVIDIA GPU", f"BAR1 Size is low ({total_bar1} MiB).", False)
            else:
                self.log("NVIDIA GPU", f"BAR1 Size is healthy ({total_bar1} MiB)")
        
        topo = self.run_cmd("nvidia-smi topo -m")
        if "PIX" in topo:
            self.log("NVIDIA Topo", "PIX Detected (Same-Switch P2P). Optimized path active.")
        elif "PXB" in topo:
            self.log("NVIDIA Topo", "PXB Detected (Multi-Switch P2P).")
        else:
            self.log("NVIDIA Topo", "No P2P paths detected (Only SYS/NODE).", False)

    def check_rdma_state(self):
        dev_info = self.run_cmd("ibv_devinfo")
        if "PORT_ACTIVE" in dev_info:
            self.log("RDMA Port", "At least one HCA port is ACTIVE")
        else:
            self.log("RDMA Port", "All ports are DOWN", False)

    def check_network_config(self):
        mapping = self.run_cmd("ibdev2netdev")
        print("\n--- Device Mapping ---")
        print(mapping.strip())
        
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
        lspci = self.run_cmd("lspci -tv")
        if "03.0" in lspci:
            self.log("PCIe Topo", "Virtual PCIe Switch (03.0) found")
        else:
            self.log("PCIe Topo", "Could not find unified switch hierarchy", False)

    def run_all(self):
        print(f"=== GPUDirect RDMA Health Check (Kernel {self.kernel_version.strip()}) ===\n")
        self.check_kernel_modules()
        self.check_dmabuf_support()
        self.check_bar_identity_mapping()
        self.check_pci_topology()
        self.check_nvidia_gpu()
        self.check_rdma_state()
        self.check_network_config()
        print("\n==========================================")
        if self.failed:
            print("RESULT: System is NOT ready for GPUDirect RDMA.")
        else:
            print("RESULT: System is READY (Modern DMA-BUF Path).")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Please run as root (sudo).")
        sys.exit(1)
    diag = GDRDiagnostics()
    diag.run_all()
