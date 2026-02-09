# TraMEL

## How To Run the Code

### 1. Download dataset

#### CICAndMal2017
```bash
mkdir -p /scratch/Malware/CICAndMal
wget -c -i data/CICAndMal2017/cic_url.txt -P /scratch/Malware/CICAndMal
```

#### IoT23
```bash
mkdir -p /scratch/Malware/iot23
wget -c -i data/IoT23/iot23_url.txt -P /scratch/Malware/iot23
```

### 2. Data Preprocess

- **CICAndMal2017**: `data/CICAndMal2017/CIC_preprocess.py` (output: `/scratch/Malware/CIC/data/`)
- **IoT23**: `data/IoT23/store_by_capture.py` → `capture_preprocess.py` (output: `/scratch/Malware/iot23/data/`)

### 3. Usage

```bash
python main_all.py --early_stop_enable --exemplar_selection random
```
