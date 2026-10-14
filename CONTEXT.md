# Hardware
- OS: Win 11 Pro (no WSL)
- GPU: RTX 5080 Laptop 16GB VRAM (target <14GB dense LLM models/EXL3/INT4 or <30GB MoE LLM models/INT4>)
- RAM: 64GB DDR5-6400
- CPU: Ultra 9 275HX (8P+16E, no HT, AVX2/VNNI, no AVX-512). llama-server.exe: -t 8
- NPU: 17 TOPS (INT8 audio/vision; no complex RNNs)
- Storage: T1 PCIe5 1TB (hot/models), T2 PCIe4 2TB (data), T3 HDD 4TB (cold)
