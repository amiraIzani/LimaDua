# Tugas Besar 1 IF3270 Pembelajaran Mesin
## Feedforward Neural Network from Scratch

---

## Deskripsi

Repository merupakan implementasi **Feedforward Neural Network (FFNN) from scratch** menggunakan Python dan NumPy

Dataset yang digunakan adalah **Global Student Placement & Salary Dataset** (`datasetml_2026.csv`) untuk klasifikasi `placement_status`.

---

## Anggota Kelompok

**Kelompok LimaDua**

| NIM      | Nama       | Tugas                                                          |
| :------- | :--------- | :------------------------------------------------------------- |
| 13523143 | Amira Izani  | Laporan, Pengembangan ffnn.py, Forward-backward propagation, Update parameter dan regularisasi, Debugging feedforward |
| 13523150 | Benedictus Nelson  |  Laporan, Pengembangan experiment.ipynb, Preprocessing dan evaluasi, Running skenario pengujian,  Visualisasi hasil dan benchmarking sklearn, Debugging training |

---

## Repository Structure

```
.
├── src/
│   ├── ffnn.py                     # Implementasi FFNN
│   └── experiment.ipynb            # Notebook eksperimen dan analisis
├── doc/
│   └── Tubes1_IF3270_G52_K03.pdf   # Laporan tugas besar
├── test/
│   └── model_20260318_223030.pkl   # file model
└── README.md
```

---

## Setup and Installation

### Requirement

- Python 3.8+
- pip

### Install Dependency

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

---

## How to Run

### Run Experiment Notebook

```bash
cd src
jupyter notebook experiment.ipynb
```

