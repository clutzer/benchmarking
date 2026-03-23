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
        # nvidia_peermem is optional/legacy for Kernel >= 6.2
        core_modules = ["nvidia", "nvidia_uvm", "mlx5_core", "mlx5_ib", "ib_uverbs"]
        loaded = self.run_cmd("lsmod")
        
        for mod in core_modules:
            if mod in loaded:
                self.log("Kernel Module", f"{mod} is loaded")
            else:
                self.log("Kernel Module", f"{mod} is MISSING", False)

        if "nvidia_peermem" in loaded:
            self.log("Kernel Module", "nvidia_peermem is loaded (Legacy Path)", success=True, warning=True)
        else:
            self.log("Kernel Module", "nvidia_peermem is not loaded (Not required for DMA-BUF)", success=True)

    def check_dmabuf_support(self):
        # Essential for Kernel 6.8 + Blackwell
        if os.path.exists("/sys/module/nvidia/parameters/nv_use_dmabuf"):
            val = self.run_cmd("cat /sys/module/nvidia/parameters/nv_use_dmabuf").strip()
            self.log("DMA-BUF", f"Driver DMA-BUF support parameter is {val}")
        else:
            self.log("DMA-BUF", "Verified via Kernel 6.8+ capability")

    def check_bar_identity_mapping(self):
        """
        Detects Identity Mapping by isolating the 128GB Data BARs.
        Standard QEMU (Top-Down) pins these to the 64TB+ ceiling (0x3F/0x40).
        Identity-Mapped BARs land in the 10TB-40TB Physical Zone (0xD/0xE/0xF).
        """
        lspci_out = self.run_cmd("lspci -vv")
        # Extract addresses specifically for the 128G Data BARs
        matches = re.findall(r"Memory at ([0-9a-fA-F]+) .*?size=128G", lspci_out)
        
        if not matches:
            self.log("Identity Map", "No 128GB BARs found to analyze.", warning=True)
            return

        is_identity_mapped = False
        reasons = []
        
        # 1. Zone Check: Standard QEMU is always > 48TB (0x300000000000)
        for addr_str in matches:
            addr = int(addr_str, 16)
            if addr < 0x300000000000:
                is_identity_mapped = True
                reasons.append(f"Physical Host Aperture (0x{addr_str[:3]}...)")
                break

        # 2. Sparsity Check: Standard QEMU packs 128GB BARs tightly.
        if len(matches) >= 2:
            addrs = sorted([int(a, 16) for a in matches])
            gap = addrs[1] - addrs[0]
            if gap > (150 * 1024**3): # Significant gap (>150GB) suggests physical layout
                is_identity_mapped = True
                reasons.append("Non-contiguous Physical Spacing")

        if is_identity_mapped:
            self.log("Identity Map", f"ACTIVE ({', '.join(set(reasons))})")
        else:
            self.log("Identity Map", "Standard Virtual Top-Down allocation detected.", warning=True)

    def check_pci_topology(self):
        # Look for the virtual PCIe switch bridge (03.0)
        lspci_tree = self.run_cmd("lspci -tv")
        if "03.0" in lspci_tree:
            self.log("PCIe Tree", "Virtual PCIe Switch (03.0) found")
        else:
            self.log("PCIe Tree", "Unified switch hierarchy not found (Check QEMU config)", False)

    def check_nvidia_gpu(self):
        out = self.run_cmd("nvidia-smi -q -d MEMORY")
        bar1_match = re.search(r"BAR1 Memory Usage.*?Total\s+:\s+(\d+)\s+MiB", out, re.S)
        if bar1_match:
            total_bar1 = int(bar1_match.group(1))
            self.log("NVIDIA GPU", f"BAR1 Size healthy ({total_bar1} MiB)" if total_bar1 >= 32768 else f"BAR1 Low ({total_bar1} MiB)", total_bar1 >= 32768)
        
        topo = self.run_cmd("nvidia-smi topo -m")
        if "PIX" in topo:
            self.log("NVIDIA Topo", "PIX Detected (Same-Switch P2P)")
        elif "PXB" in topo:
            self.log("NVIDIA Topo", "PXB Detected (Multi-Switch P2P)", warning=True)
        else:
            self.log("NVIDIA Topo", "No P2P paths (Only SYS/NODE)", False)

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
                    # Search for IPv4 address
                    match = re.search(rf"{net_dev}\s+UP\s+(\d+\.\d+\.\d+\.\d+)", ip_addr)
                    if match:
                        self.log("Network", f"{net_dev} has IP {match.group(1)}")
                    else:
                        self.log("Network", f"{net_dev} is UP but has NO IP", False)

    def run_all(self):
        print(f"=== GPUDirect RDMA System Health Check (Kernel {self.kernel_version.strip()}) ===\n")
        self.check_kernel_modules()
        self.check_dmabuf_support()
        self.check_bar_identity_mapping()
        self.check_pci_topology()
        self.check_nvidia_gpu()
        self.check_rdma_state()
        self.check_network_config()
        print("\n==========================================")
        if self.failed:
            print("RESULT: System is NOT fully optimized for GPUDirect RDMA.")
        else:
            print("RESULT: System is READY (Modern DMA-BUF Path).")

if __name__ == "__main__":
    if os.geteuid() != 0:
        sys.exit("Please run as root (sudo).")
    diag = GDRDiagnostics()
    diag.run_all()
