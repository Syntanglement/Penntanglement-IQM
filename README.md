# Finding a recipe to the "Secret Sauce" of Quantum Computing
**An IQuHACK 2026 IQM Challenge Submission by Steven Jaime Salazar, Justin Zou, Cameron Zuschmidt, Yuzhe Pan & Bhargav Yerramsetty**

### How quantum computers do more than classical ones

**What is the goal of this section?**

Our goal here is to explain what superposition is, how it differs from classical randomness, and why it enables quantum advantage.
And then, we will show how to *prove* when superposition is really present.




## Classical bit vs Quantum bit

A **classical bit** can be:
- `0`
- `1`

A **quantum bit (qubit)** can be:
$$ 
|\psi\rangle = \alpha |0\rangle + \beta |1\rangle
 $$
where:
$$
|\alpha|^2 + |\beta|^2 = 1
$$

...and this is also known as a **superposition** of `|0⟩` and `|1⟩`.

(Note that a qubit can exist in *both* states at once until we measure it.)


## Intuition: Not just randomness

A coin flip is random, but a qubit is not just random because it has **phase** and **interference**.

A classical mixture will consist of a:
- 50% chance of 0
- 50% chance of 1

But a quantum superposition will have:
$$
|\psi\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}
$$
This looks like a coin flip when measured…
but behaves *very differently* when combined with other operations.


## ...but Superposition is not enough

At first glance, superposition already seems weirdly powerful:

$$
|\psi\rangle = \alpha |0\rangle + \beta |1\rangle
$$

But there is a catch, because if we prepare two qubits independently, we obtain:

$$
(|0\rangle + |1\rangle) \otimes (|0\rangle + |1\rangle)
$$

Which upon closer inspection, still is only a product of two single-qubit states.

Such states can always be explained as:

• independent quantum systems  
• no non-classical correlations  
• efficiently simulable  

Which, unfortunately, means that superposition alone cannot explain quantum advantage.

So, it's about time we get to the sauce of all of this: Entanglement!

## From a superposition to entanglement

An entangled state is one that **cannot** be written as a product of individual qubit states, so for example:

$$
|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}
$$

But there also exists no single-qubit states $|\psi_1\rangle$, $|\psi_2\rangle$ such that:

$$
|\Phi^+\rangle = |\psi_1\rangle \otimes |\psi_2\rangle
$$

Which means that the qubits do not have independent states, measurement outcomes that are correlated, and no classical randomness can reproduce the statistics, all definitions of what Einstein called *“spooky action at a distance.”*


## The experimental challenge

We shall now face the curmudgeon’s question:

> If I gave you a box of qubits, how do you know they are entangled?

Our task is to construct experiments whose outcomes cannot come from classical randomness & from separable quantum states, and must come ONLY from entangled quantum states

In other words, we must **certify entanglement from measurement data alone**. (or, that's what we aim to do, at least.)


## The first entanglement experiment: Bell states

We shall begin with the simplest nontrivial case of two qubits provided to us by IQM Quantum Computers:

$$
|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}
$$

and then design measurements in accordance to this equation to show that the statistics it produces violate classical explanations and that its correlations cannot arise from separable states in order to scale up to larger-qubit systems.

In this state, each qubit individually appears random and the two qubits are perfectly correlated, but as a consequence this state **cannot** be written as a product:

$$
|\Phi^+\rangle \neq |\psi_1\rangle \otimes |\psi_2\rangle
$$

so our goal will have to be reached experimentally using only measurement data.

Our circuit will consist of a single Hadamard gate on qubit 0 to facilitate superposition and a CNOT gate to cause entanglement:

$$
|00\rangle \xrightarrow{H \otimes I} \frac{|00\rangle + |10\rangle}{\sqrt{2}}
\xrightarrow{\text{CNOT}} \frac{|00\rangle + |11\rangle}{\sqrt{2}}
$$


```python
# An IQM Token is needed to run all code blocks
!pip install -U "iqm-client[qiskit]==33.0.3"
!pip install matplotlib

from iqm.qiskit_iqm import IQMProvider

token = input("Enter your IQM token: ")

provider = IQMProvider(
    "https://resonance.meetiqm.com/",
    quantum_computer="emerald", 
    token=token
)

backend = provider.get_backend()
backend

```




    <iqm.qiskit_iqm.iqm_provider.IQMBackend at 0x24beea3b110>




```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

qc

```




    <qiskit.circuit.quantumcircuit.QuantumCircuit at 0x24bebfa65d0>



## Measurements in the Z basis

We will now use the previous simple circuit measure both qubits in the computational (Z) basis. If the state were separable, we would expect independent outcomes, so instead we will test for correlations.



```python
from qiskit import transpile

qc_z = qc.copy()
qc_z.measure_all()

tqc_z = transpile(qc_z, backend=backend, optimization_level=3)
job_z = backend.run(tqc_z, shots=2000)
result_z = job_z.result()
counts_z = result_z.get_counts()

counts_z
```


    Progress in queue:   0%|          | 0/3 [00:00<?, ?it/s]





    {'11': 929, '00': 984, '01': 22, '10': 65}



The Z-basis results show near-perfect correlation between the two qubits (with a small amount of leakage into 01/10 from noise & readout errors), however, classical correlation alone is not sufficient to prove entanglement, which leads us to test coherence in a second basis.

## Measurements in the X basis & Correlations

We begin with using the same circuit in the X basis



```python
qc_x = qc.copy()
qc_x.h(0)
qc_x.h(1)
qc_x.measure_all()

tqc_x = transpile(qc_x, backend=backend, optimization_level=3)
job_x = backend.run(tqc_x, shots=2000)
result_x = job_x.result()
counts_x = result_x.get_counts()

counts_x
```


    Progress in queue:   0%|          | 0/3 [00:00<?, ?it/s]





    {'11': 911, '00': 985, '01': 30, '10': 74}




We now compute the correlation observables:

$$
\langle Z_0 Z_1 \rangle, \quad \langle X_0 X_1 \rangle
$$

with:


```python
def exp_zz(counts, shots=2000):
    total = 0
    for bitstring, c in counts.items():
        # rightmost bit = qubit 0 in Qiskit
        b0 = int(bitstring[-1])
        b1 = int(bitstring[-2])
        val = 1 if (b0 ^ b1) == 0 else -1
        total += val * c
    return total / shots

zz = exp_zz(counts_z)
xx = exp_zz(counts_x)

zz, xx
```




    (0.913, 0.896)



For the Bell state, quantum theory will predict that

$$
\langle Z_0 Z_1 \rangle = 1, \quad \langle X_0 X_1 \rangle = 1.
$$

Which allows us to combine these into an entanglement *witness* as:

$$
W = \langle Z_0 Z_1 \rangle + \langle X_0 X_1 \rangle.
$$

For across all the separable two-qubit states:

$$
W \le 1.
$$



```python
W = zz + xx
W

```




    1.8090000000000002



Since the measured value violates the separable bounds we've set, we may conclude that the qubits are entangled based only on measurement data, and that no assumption about state preparation was required. 

## The n-qubits entanglement experiment: GHZ Gate with a Stabilizer Witness


The intution starts off from the well-known bell states:
\begin{align*}
   |\phi^+\rangle &= \frac{|00\rangle + |11\rangle}{\sqrt{2}}\\
   |\phi^-\rangle &= \frac{|00\rangle - |11\rangle}{\sqrt{2}}\\
   |\psi^+\rangle &= \frac{|01\rangle + |10\rangle}{\sqrt{2}}\\
   |\psi^-\rangle &= \frac{|01\rangle - |10\rangle}{\sqrt{2}}\\
\end{align*}


These are the well-known double-qubit entangled superpositions. It can be shown that there does not exist single-qubit superpositions $|\psi_1\rangle,|\psi_2\rangle$ such that any of $|\phi^+\rangle, |\phi^-\rangle, |\psi^+\rangle$, and $|\psi^-\rangle$ can be factored into the form $|\psi_1\rangle \otimes |\psi_2\rangle$.


Specifically for $|\phi^+\rangle$ and $|\phi^-\rangle$, it is easy to see that the pattern can indefinitely extend to a higher number $n$ of qubits, as superpositions of the form
\begin{align*}
   \alpha|x^{\otimes n}\rangle + \beta|y^{\otimes n}\rangle
\end{align*}
where $n\in\mathbb{Z}$ such that $n \geq 2$, $|\alpha|^2 + |\beta|^2 = 1$, and $x$ and $y$ are states, can not be factored. That is, no qubit state, not even multi-qubit states, can be factored out.


The Greenberger–Horne–Zeilinger (GHZ) State is then of the form
\begin{align*}
   \frac{\left|0^{\otimes n}\right\rangle + \left|1^{\otimes n}\right\rangle}{\sqrt{2}}
\end{align*}
where $n$ is the number of qubits.

Our goal is to design a entanglement witness for GHZ scaling with system size to show that superpositions of this form are indeed entangled.

Fidelity is a measure of accuracy and reliability for resulting states from a superposition given errors. It is scaled between 0 and 1. The formula for Fidelity $F$ is:
\begin{align*}
F \geq \frac{1}{2}\left( \left\langle X^{\otimes N} \right\rangle + \frac{1}{N - 1}\sum_{i = 1}^{N - 1}\left\langle Z_iZ_{i + 1}\right\rangle\right)
\end{align*}
where $N$ is the total number of qubits used, $\frac{1}{N - 1}\sum_{i = 1}^{N - 1}\left\langle Z_iZ_{i + 1}\right\rangle$ is the average Z-basis measurement and $\left\langle X^{\otimes N} \right\rangle$ is the global coherence (X-basis measurement).

Let $\text{GHZ}_N$ be the GHZ quantum gate for $N$ qubits, where $\text{GHZ}_N\left|0^{\otimes N}\right\rangle = \frac{\left|0^{\otimes n}\right\rangle + \left|1^{\otimes n}\right\rangle}{\sqrt{2}}$.

Let $i\in\mathbb{Z}$ such that $0 \leq i \leq N - 1$. Then, $Z_i = 1$ if the $i$-th resulting qubit state is even (in binary representation). Otherwise, $Z_i = -1$. 

The code used to model our experimental fidelity & to conduct data collection is as shown:


```python
from iqm.qiskit_iqm import IQMProvider

from qiskit import visualization, transpile

def getGhzCircuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc

def getZZExpectation(counts, i, j, shots):
    exp = 0
    for bitstring, freq in counts.items():
        zi = 1 if bitstring[::-1][i] == '0' else -1
        zj = 1 if bitstring[::-1][j] == '0' else -1
        exp += zi * zj * freq
    return exp / shots

def getGlobalXExpectation(counts, shots):
    exp = 0
    for bitstring, freq in counts.items():
        bits = bitstring.replace(" ", "")
        parity = bits.count('1')
        value = 1 if parity % 2 == 0 else -1
        exp += value * freq
    
    return exp / shots

# qubit num
qBits = 54

qcz = getGhzCircuit(qBits)
qcz.measure_all()

qcx = getGhzCircuit(qBits)
for i in range(qBits):
    qcx.h(i)
qcx.measure_all()

qc_transpiled1 = transpile(qcz, backend=backend)
job_z = backend.run(qc_transpiled1, shots=1024)
result_z = job_z.result()
qc_transpiled2 = transpile(qcx, backend=backend)
job_x = backend.run(qc_transpiled2, shots=1024)
result_x = job_x.result()

# Writing data
with open("zz_expectation.txt", 'w') as file1:
    for i in range(qBits - 1):
        s = f"<Z{i}Z{i+1}>:\t {getZZExpectation(result_z.get_counts(), i, i+1, 1024)}\n"
        file1.write(f"{s}\n")

with open("x_expectations.txt", 'w') as file2:
    s = f"<X^prodN>:\t{getGlobalXExpectation(result_x.get_counts(), 1024)}"
    file2.write(f"{s}\n")

zz_sum = 0
with open("fidelity.txt", 'w') as file3:
    file3.write(f"GHZ Witness Fidelity Lower Bound with {qBits} qubits and 1024 shots\n\n\n")
    file3.write(f"zz_sum = {zz_sum}\n")

    # Calculate Fidelity 
    for i in range(qBits - 1):
        zz_sum += getZZExpectation(result_z.get_counts(), i, i + 1, 1024)
    file3.write(f"zz_sum = {zz_sum}\n")

    zz_avg = zz_sum / (qBits - 1)
    file3.write(f"\n\n\nzz_avg = {zz_avg}\n")

    x_val = getGlobalXExpectation(result_x.get_counts(), 1024)
    file3.write(f"\n\n\nx_val = {x_val}\n")

    # Fidelity
    fidelity_lb = 0.5 * (x_val + zz_avg)
    file3.write(f"\n\n\nfidelity lower bound = {fidelity_lb}\n")

    shots = 1024
counts_z = result_z.get_counts()
counts_x = result_x.get_counts()


print("GHZ Witness Results")
print(f"N qubits: {qBits}")
print(f"Shots:    {shots}")

# ZZ nearest-neighbor expectations
print("\nZ-basis nearest-neighbor correlations <Z_i Z_{i+1}>:")
zz_sum_print = 0.0
for i in range(qBits - 1):
    val = getZZExpectation(counts_z, i, i + 1, shots)
    zz_sum_print += val
    print(f"  <Z{i}Z{i+1}> = {val:.6f}")

zz_avg_print = zz_sum_print / (qBits - 1)
print(f"\nzz_sum = {zz_sum_print:.6f}")
print(f"zz_avg = {zz_avg_print:.6f}")

# Global X expectation
x_val_print = getGlobalXExpectation(counts_x, shots)
print("\nGlobal X coherence:")
print(f"  <X^prodN> = {x_val_print:.6f}")

# 3) Fidelity lower bound
fidelity_lb_print = 0.5 * (x_val_print + zz_avg_print)
print("\nFidelity lower bound:")
print(f"  F_lb = 0.5 * (x_val + zz_avg) = {fidelity_lb_print:.6f}")

# 4) top outcomes in Z basis
print("\nTop 10 Z-basis bitstrings by frequency:")
top_z = sorted(counts_z.items(), key=lambda kv: kv[1], reverse=True)[:10]
for bitstring, freq in top_z:
    print(f"  {bitstring}: {freq}")

print("\nFinished")
```


    Progress in queue:   0%|          | 0/9 [00:00<?, ?it/s]



    Progress in queue:   0%|          | 0/3 [00:00<?, ?it/s]


    GHZ Witness Results
    N qubits: 54
    Shots:    1024
    
    Z-basis nearest-neighbor correlations <Z_i Z_{i+1}>:
      <Z0Z1> = 0.554688
      <Z1Z2> = 0.562500
      <Z2Z3> = 0.375000
      <Z3Z4> = 0.505859
      <Z4Z5> = 0.632812
      <Z5Z6> = 0.673828
      <Z6Z7> = 0.617188
      <Z7Z8> = 0.455078
      <Z8Z9> = 0.361328
      <Z9Z10> = 0.511719
      <Z10Z11> = 0.388672
      <Z11Z12> = 0.414062
      <Z12Z13> = -0.214844
      <Z13Z14> = 0.406250
      <Z14Z15> = 0.351562
      <Z15Z16> = 0.328125
      <Z16Z17> = -0.080078
      <Z17Z18> = 0.599609
      <Z18Z19> = 0.527344
      <Z19Z20> = 0.476562
      <Z20Z21> = 0.468750
      <Z21Z22> = 0.427734
      <Z22Z23> = 0.457031
      <Z23Z24> = 0.341797
      <Z24Z25> = 0.488281
      <Z25Z26> = 0.511719
      <Z26Z27> = 0.259766
      <Z27Z28> = 0.167969
      <Z28Z29> = -0.195312
      <Z29Z30> = 0.210938
      <Z30Z31> = 0.169922
      <Z31Z32> = 0.062500
      <Z32Z33> = -0.082031
      <Z33Z34> = 0.015625
      <Z34Z35> = 0.214844
      <Z35Z36> = 0.460938
      <Z36Z37> = -0.009766
      <Z37Z38> = 0.171875
      <Z38Z39> = 0.476562
      <Z39Z40> = 0.427734
      <Z40Z41> = 0.273438
      <Z41Z42> = -0.007812
      <Z42Z43> = -0.392578
      <Z43Z44> = 0.062500
      <Z44Z45> = 0.009766
      <Z45Z46> = 0.078125
      <Z46Z47> = 0.351562
      <Z47Z48> = 0.423828
      <Z48Z49> = -0.236328
      <Z49Z50> = -0.230469
      <Z50Z51> = 0.490234
      <Z51Z52> = -0.269531
      <Z52Z53> = 0.273438
    
    zz_sum = 14.320312
    zz_avg = 0.270195
    
    Global X coherence:
      <X^prodN> = -0.046875
    
    Fidelity lower bound:
      F_lb = 0.5 * (x_val + zz_avg) = 0.111660
    
    Top 10 Z-basis bitstrings by frequency:
      001100100100111111100111000110110000000100010000000000: 1
      011101110001100001111000000011000000000001111100000111: 1
      011001001001111110001011100011101000001101100000000010: 1
      101100111000001001110111101111000000010000000110110111: 1
      101101011111101101100011101101111101111100000011100111: 1
      110101110001011000010011000000000000000000110001000000: 1
      011111101101111110001011111111001111100001000100000001: 1
      101101101001110001001000111000000000000000000000001010: 1
      111100101110100000010110001001101100000001000111110000: 1
      101101001011100001011100011101000001000000110000000000: 1
    
    Finished


## Why are we sure this certifies entanglement?

For the ideal GHZ state
$$
|GHZ_N\rangle = \frac{|0^{\otimes N}\rangle + |1^{\otimes N}\rangle}{\sqrt{2}},
$$
the following stabilizer relations hold:
$$
\begin{align}
Z_i Z_{i+1} |GHZ_N\rangle &= |GHZ_N\rangle \quad \text{for all } i, \\
X^{\otimes N} |GHZ_N\rangle &= |GHZ_N\rangle.
\end{align}
$$

Thus, in the absence of noise,
$$
\begin{align}
\langle Z_i Z_{i+1} \rangle &= 1, \\
\langle X^{\otimes N} \rangle &= 1.
\end{align}
$$

The two components of the fidelity bound look at two different physical properties:

- The operators $Z_i Z_{i+1}$ test whether neighboring qubits share the same classical parity, as expected for strings of the form $00\ldots0$ and $11\ldots1$.

- The operator $X^{\otimes N}$ tests whether the two macroscopically distinct components $|0^{\otimes N}\rangle$ and $|1^{\otimes N}\rangle$ remain in coherent superposition rather than forming a classical mixture.


So, a purely classical correlated state of the form
$$
\begin{align}
\rho_{\text{mix}} = \tfrac12\Big(|0^{\otimes N}\rangle\langle 0^{\otimes N}| 
+ |1^{\otimes N}\rangle\langle 1^{\otimes N}|\Big)
\end{align}
$$
can reproduce large values of
$$
\begin{align}
\langle Z_i Z_{i+1} \rangle,
\end{align}
$$
but necessarily satisfies
$$
\begin{align}
\langle X^{\otimes N} \rangle = 0,
\end{align}
$$
since all phase coherence between the two branches has been destroyed.

Therefore, observing simultaneously:
$$
\begin{align}
\langle Z_i Z_{i+1} \rangle &\approx 1, \\
\langle X^{\otimes N} \rangle &> 0
\end{align}
$$
rules out any classical mixture and requires genuine quantum coherence between the branches.

The fidelity lower bound
$$
\begin{align}
F \geq \frac{1}{2}\left( 
\left\langle X^{\otimes N} \right\rangle 
+ \frac{1}{N - 1}\sum_{i = 1}^{N - 1}\left\langle Z_iZ_{i + 1}\right\rangle
\right)
\end{align}
$$
thus serves as an entanglement witness, and the high values soon observed in our statistics represent both parity correlation and global coherence, which isn't achieved by separable or classically correlated states.

Consequently, when the experimentally measured bound remains significantly above zero as \(N\) increases, it provides direct evidence that the prepared state is genuinely entangled across many qubits.


## Scaling behavior & Dataset

We measure the entanglement witness (or fidelity) as a function of the number of qubits N.

Observed below, the trend has an approximately linear decay that's consistent with an increasing circuit depth with a trend in accumulated gate error and readout noise, but nevertheless, for small and intermediate N we were able to certify multipartite entanglement due to the measured values exceeding the separable-state threshold.

<div style="display: flex; justify-content: center;">
   <img alt="Graph of GHZ Fidelity to the number of Qubits" src="./assets/fidelitygraph.png" width="800"/>
</div>
Here, the fidelity is plotted for increments of every 5 qubits starting from 5 to 50. The minimum number of qubits possible for a GHZ gate (2) and the maximum number of qubits available in the quantum computer (54) are also plotted.

The code used to generate the graph from a trial of the code above is below.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


data = {
    2: 0.8994140625,
    5: 0.8662109375,
    10: 0.7507595486111112,
    15: 0.6273716517857143,
    20: 0.5374177631578947,
    25: 0.4580891927083333,
    30: 0.4538995150862069,
    35: 0.38192210477941174,
    40: 0.3817858573717949,
    45: 0.3424627130681818,
    50: 0.13805404974489796,
    54: 0.07488207547169812
}

threshold = 0.5  # common GME threshold 
df = pd.DataFrame({"N": list(data.keys()), "fidelity_lb": list(data.values())}).sort_values("N")
x = df["N"].to_numpy()
y = df["fidelity_lb"].to_numpy()
if len(df) >= 2:
    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b
else:
    m, b = np.nan, np.nan
    y_pred = None
cert_mask = y > threshold
max_cert_N = int(x[cert_mask].max()) if cert_mask.any() else None

plt.figure()

# data 
plt.plot(x, y, marker="o", linestyle="-", label="Measured data")

# regression line
if y_pred is not None:
    plt.plot(x, y_pred, linestyle="--", label=f"Linear fit: y={m:.3f}N+{b:.3f}")

# threshold line
plt.axhline(threshold, linestyle=":", linewidth=2, label=f"Threshold = {threshold}")

# region shading
if max_cert_N is not None:
    plt.axvspan(x.min(), max_cert_N, alpha=0.15, label=f"Certified up to N={max_cert_N}")
    idx = np.where(x == max_cert_N)[0][0]
    plt.annotate(
        f"max certified N = {max_cert_N}\nF={y[idx]:.3f}",
        xy=(x[idx], y[idx]),
        xytext=(x[idx] + 2, min(0.95, y[idx] + 0.08)),
        arrowprops=dict(arrowstyle="->")
    )

# axes labels/title
plt.xlabel("Number of qubits (N)")
plt.ylabel("Fidelity / witness-derived fidelity")
plt.title("Scaling of entanglement certification vs N")
plt.grid(True)
plt.legend()

plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.01, 0.1))
plt.xlim(0, 55)
plt.xticks(np.arange(0, 55, 5))

plt.tight_layout()

# plot and table export
outdir = Path("results")
(outdir / "figures").mkdir(parents=True, exist_ok=True)
(outdir / "tables").mkdir(parents=True, exist_ok=True)

png_path = outdir / "figures" / "fidelity_vs_n.png"
pdf_path = outdir / "figures" / "fidelity_vs_n.pdf"
csv_path = outdir / "tables" / "fidelity_vs_n.csv"
fit_path = outdir / "tables" / "fit_and_threshold.csv"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

df.to_csv(csv_path, index=False)

pd.DataFrame([{
    "slope": m,
    "intercept": b,
    "threshold": threshold,
    "max_certified_N": max_cert_N,
    "num_points": len(df)
}]).to_csv(fit_path, index=False)

plt.show()

print("Saved:")
print(f" - {png_path}")
print(f" - {pdf_path}")
print(f" - {csv_path}")
print(f" - {fit_path}")

```


    
![png](Penntanglement_files/Penntanglement_24_0.png)
    


    Saved:
     - results\figures\fidelity_vs_n.png
     - results\figures\fidelity_vs_n.pdf
     - results\tables\fidelity_vs_n.csv
     - results\tables\fit_and_threshold.csv


Let $q[i]$ be the $i$-th qubit, where the $0$-th qubit is the first and right-most qubit. Let $\text{CNOT}_{i,j}$ be the CNOT gate being applied with the $i$-th qubit as the control qubit and the $j$-th qubit as the target qubit. Now, the quantum circuit for $\left|\text{GHZ}_{N}\right\rangle$ can be constructed through first applying a Hadamard Gate on $q[0]$, and then applying $\text{CNOT}_{x,x + 1}$ gates for $x\in\mathbb{Z}$ such that $0 \leq i \leq n - 2$. The chaining of CNOT gates is what allows the entanglement to scale with the number of qubits.

Of course, this idea of chaining CNOT gates can be applied to the quantum circuits of other bell states as well. I call these, $\psi^-$ CNOT Extended gates, $\phi^+$ CNOT Extended gates, and $\phi^-$ CNOT Extended gates respectively. Graphing all their fidelity with respect to the number of qubits. A $\psi^-$ CNOT Extended gate applied on $N$ qubits results in the state $\frac{\left|0^{\otimes N} \right\rangle - \left|1^{\otimes N} \right\rangle}{\sqrt{2}}$. a $\phi^+$ CNOT Extended gate applied on $N$ qubits results in the state $\frac{\left|0^{\otimes (N - 1)}1 \right\rangle + \left|1^{\otimes (N - 1)}0 \right\rangle}{\sqrt{2}}$. Finally, a $\phi^-$ CNOT Extended gate applied on $N$ qubits results in the state $\frac{\left|0^{\otimes (N - 1)}1 \right\rangle - \left|1^{\otimes (N - 1)}0 \right\rangle}{\sqrt{2}}$. It can be seen that none of these states can be factored into a tensor product of states with less qubits.

<div style="text-align: center;">
    <img src="./assets/Scaling_Multi_Graph.png" width="800">
</div>

We can see here that the linear relationship extends past just one superposition.

Modifications and user abstractions were made to the code. Here is the updated code for conducting data collection:


```python
import os

from iqm.qiskit_iqm import IQMProvider
provider = IQMProvider("https://resonance.meetiqm.com/", quantum_computer="emerald",
            token="z0NSN7NyK3k1t2PUa+aexwpxoeCEzQ2eS+JVceJWLW0BnBhWQM114JahHDfz8mq0")
backend = provider.get_backend()

from qiskit import QuantumCircuit, transpile, visualization

def getGhzCircuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)

    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc

def getPsiMinusCircuit(n):
    qc = QuantumCircuit(n)
    qc.x(0)
    qc.h(0)

    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc

def getPhiPlusCircuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    qc.x(1)

    for i in range(n - 1):
        qc.cx(i, i + 1)

    return qc

def getPhiMinusCircuit(n):
    qc = QuantumCircuit(n)
    qc.x(0)
    qc.h(0)
    qc.x(1)

    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc

def getZZExpectation(counts, i, j, shots, gate):
    exp = 0
    sign = -1 if i == 0 and (gate == "phiPlus" or gate == "phiMinus") else 1
    for bitstring, freq in counts.items():
        zi = 1 if bitstring[::-1][i] == '0' else -1
        zj = 1 if bitstring[::-1][j] == '0' else -1
        exp += zi * zj * freq * sign
        # multiply by -1 for phi gates since first bit pair is uncorrelated
    return exp / shots

def getGlobalXExpectation(counts, shots):
    exp = 0

    # multiply by (-1) ** (number of X gates applied eve)
    # sign = (-1) ** (qBits - 1) if (gate == "phiPlus" or gate == "phiMinus") else (-1) if gate == "psiMinus" else 1
    for bitstring, freq in counts.items():
        bits = bitstring.replace(" ", "")
        parity = bits.count('1')
        value = 1 if parity % 2 == 0 else -1
        exp += value * freq
    
    return exp / shots

# Gate can be "ghz" (default), "psiMinus", "phiPlus", and "phiMinus"
def getData(gate, rootDir):
    for qBits in range(2, 55):
        if qBits != 2 and qBits != 54 and qBits % 5 != 0:
            continue

        qcz = getGhzCircuit(qBits)
        qcx = getGhzCircuit(qBits)
        
        if gate == "psiMinus":
            qcz = getPsiMinusCircuit(qBits)
            qcx = getPsiMinusCircuit(qBits)
        elif gate == "phiPlus":
            qcz = getPhiPlusCircuit(qBits)
            qcx = getPhiPlusCircuit(qBits)
        elif gate == "phiMinus":
            qcz = getPhiMinusCircuit(qBits)
            qcx = getPhiMinusCircuit(qBits)
        for i in range(qBits):
            qcx.h(i)

        qcz.measure_all()        
        qcx.measure_all()

        # Run the job for measuring with Z-basis
        qc_transpiled1 = transpile(qcz, backend=backend)
        job_z = backend.run(qc_transpiled1, shots=1024)
        result_z = job_z.result()


        # Run the job for measuring with X-basis
        qc_transpiled2 = transpile(qcx, backend=backend)
        job_x = backend.run(qc_transpiled2, shots=1024)
        result_x = job_x.result()

        #Write data to external files
        filePath1 = f"{rootDir}/{rootDir}{qBits}/zz_expectation.txt"
        os.makedirs(os.path.dirname(filePath1), exist_ok=True)
        with open(filePath1, 'w') as file1:
            #Calculate Z-basis measurement
            for i in range(qBits - 1):
                s = f"<Z{i}Z{i+1}>:\t {getZZExpectation(result_z.get_counts(), i, i+1, 1024, gate)}\n"
                file1.write(f"{s}\n")

        filePath2 = f"{rootDir}/{rootDir}{qBits}/x_expectation.txt"
        os.makedirs(os.path.dirname(filePath2), exist_ok=True)
        with open(filePath2, 'w') as file2:
            # Calculate X-basis measurement
            s = f"<X^prodN>:\t{getGlobalXExpectation(result_x.get_counts(), 1024)}"
            file2.write(f"{s}\n")

        zz_sum = 0
        filePath3 = f"{rootDir}/{rootDir}{qBits}/fidelity.txt"
        os.makedirs(os.path.dirname(filePath3), exist_ok=True)
        with open(filePath3, 'w') as file3:
            file3.write(f"{gate} Witness Fidelity Lower Bound with {qBits} qubits and 1024 shots\n\n\n")

            # Calculate Fidelity 
            for i in range(qBits - 1):
                zz_sum += getZZExpectation(result_z.get_counts(), i, i + 1, 1024, gate)
            file3.write(f"zz_sum = {zz_sum}\n")

            zz_avg = zz_sum / (qBits - 1)
            file3.write(f"zz_avg = {zz_avg}\n")

            x_val = getGlobalXExpectation(result_x.get_counts(), 1024)
            file3.write(f"\nx_val = {x_val}\n")

            # sign_x = (-1) ** (number of times X gate is applied ever)
            sign_x = 1
            if gate == "psiMinus" or gate == "phiMinus":
                sign_x = -1
            fidelity_lb = 0.5 * (sign_x * x_val + zz_avg)
            file3.write(f"\nfidelity lower bound = {fidelity_lb}\n")

getData("phiMinus", "phiMinusTest")
```

Here is the updated code for generating the graph:


```python
import matplotlib.pyplot as plt
import numpy as np

#ghz
data1 = {
    2: 0.8994140625,
    5: 0.8662109375,
    10: 0.7507595486111112,
    15: 0.6273716517857143,
    20: 0.5374177631578947,
    25: 0.4580891927083333,
    30: 0.4538995150862069,
    35: 0.38192210477941174,
    40: 0.3817858573717949,
    45: 0.3424627130681818,
    50: 0.13805404974489796,
    54: 0.07488207547169812
}

#psiMinus
data2 = {
    2: 0.9150390625,
    5: 0.87890625,
    10: 0.7483723958333333,
    15: 0.5837751116071428,
    20: 0.45610608552631576,
    25: 0.4770914713541667,
    30: 0.38197063577586204,
    35: 0.3959386488970588,
    40: 0.398036858974359,
    45: 0.33556019176136365,
    50: 0.06772161989795919,
    54: 0.06793558372641509
}

# phiPlus
data3 = {
    2: 0.9169921875,
    5: 0.857421875,
    10: 0.6391059027777778,
    15: 0.5171595982142857,
    20: 0.5138260690789473,
    25: 0.448486328125,
    30: 0.43463766163793105,
    35: 0.39832261029411764,
    40: 0.3388922275641026,
    45: 0.3124112215909091,
    50: 0.0797592474489796,
    54: 0.14954304245283018
}

# phiMinus
data4 = {
    2: 0.9384765625,
    5: 0.86962890625,
    10: 0.6312934027777778,
    15: 0.6170479910714286,
    20: 0.5193256578947368,
    25: 0.4418131510416667,
    30: 0.37722252155172414,
    35: 0.3810604319852941,
    40: 0.38221153846153844,
    45: 0.2859108664772727,
    50: 0.053750797193877556,
    54: 0.13797169811320756
}

data_all = [
    (data1, 'GHZ', 'blue'),
    (data2, 'Ψ⁻ CNOT Extended', 'red'),
    (data3, 'Φ⁺ CNOT Extended', 'green'),
    (data4, 'Φ⁻ CNOT Extended', 'orange')
]

plot_all = True

if plot_all:
    for data, label, color in data_all:
        x = np.array(list(data.keys()))
        y = np.array(list(data.values()))

        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b

        plt.scatter(x, y, color=color, label=label)
        plt.plot(x, y_pred, color=color, linestyle='--', alpha=0.5, label=f'{label} fit: y={m:.4f}x+{b:.2f}')
else:
    x = np.array([i for i in data1.keys()])
    y = np.array([i for i in data1.values()])

    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b

    plt.scatter(x, y, color='blue', label='Original Data')
    plt.plot(x, y_pred, color='red', label=f'Regression Line: y={m:.2f}x+{b:.2f}')


plt.xlabel('Number of qubits (N)')
plt.ylabel('Fidelity')
plt.title('Scaling of Entanglement with Four Superpositions')    #change this accordingly
plt.legend()
plt.grid(True)


plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.01, 0.1))

plt.xlim(0, 55)
plt.xticks(np.arange(0, 55, 5))

plt.show()
```

The main differences are in how the lower bound for fidelity is calculated changes. $\psi^-$ CNOT Extended and $\phi^-$ CNOT Extended have a phase shift of $\pi$ in the superposition, requiring the measurement in X basis to change signs in those cases. $\phi^+$ CNOT Extended and $\phi^-$ CNOT Extended are identical to $GHZ$ and $\psi^-$ CNOT Extended respectively except that their first qubit is flipped, which requires the first $\langle Z_0, Z_1\rangle$ correlation pair to change parity.

## Some Theoretical Ideas

There are two theroetical states possible with such constructions derived from the bell states. The observation can be made that we can further construct more quantum gates that result in entangled states from adding X gates to any already existing quantum circuit board that results in an entangled state. Each $X$ gate results in flipping an even number qubits with the same number of qubits being flipped for both states.

This fact indicates a lower bound on the number of <b>constructable</b> entangled quantum states based on the number of qubits $n$.

With the construction of adding $X$ gates, all tensor factors inbetween are such that the state pairs are different in every qubit position. Take entangled state $\frac{|01100110\rangle+|10011001\rangle}{\sqrt{2}}$ for example. This means that such entangled states are also maximally entangled quantum states. From this, it can be shown that the total number of entangled states constructable from adding $X$ gates to the $\text{GHZ}_n$ gate is $2^n$.

Now consider the total number of possible entangled quantum states for $n$ qubits. Observe that tensor product state pairs are such that the first tensor factor of both as well as the last tensor factor of both are always different from one another. Take $\left| \phi^+ \right\rangle = \frac{|01\rangle+|10\rangle}{\sqrt{2}}$. The first of the left is $|0\rangle$ while the first of the right is $|1\rangle$. Likewise, the last of the left is $|1\rangle$ while the last of the right is $|0\rangle$. There are only 4 such combinations of this (why there are four bell states). Then, the $n-2$ tensor factors inbetween for each member of the pair can be either one. Thus, the total number of entangled states with $n$ qubits is $4 \cdot 2^{n-2}\cdot2^{n-2} = 2^{2n-2}$.

Therefore, we have now proven that the total number of constructable entangled quantum states lies between $2^n$ and $2^{2n - 2}$. For $n = 2$, this lies between 4 and 4, which is true (the four bell states). For $n = 3$, the number is now between 8 and 16.

To summarize:
1. The total number of entangled states constructable from adding $X$ gates to the $\text{GHZ}_{n}$ gate is $2^n$
2. The total number of entangled states with $n$ qubits is $2^{2n-2}$
3. The total number of constructable entangled quantum states lies between $2^n$ and $2^{2n - 2}$

Though tedious, note that it would be theoretically possible to show the scaling between the lower bound of fidelity and the number of qubits for each of the constructable entangled states for arbitrary $n$ qubits given that we have to tweak the fidelity lower bound calculation every time.

## Final experiment: Hardware-optimized GHZ entanglement on IQM Emerald

Before, we briefly mentioned Emerald, but in our attempt to entangle more qubits, the best place to look would be at the core of the problem...

For context, IQM Emerald is based on superconducting transmon qubits arranged in a square lattice with **CZ as the native two-qubit gate** and parameterized X/Y rotations (PRX) as native single-qubit gates.

In previous sections, GHZ states were constructed using CNOT (CX) gates. However, since CX is not native to Emerald, each CX must be decomposed into multiple single-qubit rotations and a CZ gate which unfortunately increases circuit depth and accelerates decoherence, especially for the global coherence observable ⟨X^{⊗N}⟩.

Here, we try to better match the hardware specifications of Emerald, so we constructed the GHZ state directly using CZ gates via the identity:
$$
\text{CNOT}_{i,j} = (I \otimes H)\,\text{CZ}_{i,j}\,(I \otimes H),
$$
so that the entangling operation is implemented natively on the device.

We then further instruct the transpiler to:

• optimize gate count and depth  
• respect the square-lattice topology  
• minimize SWAP operations  

by using SABRE layout and routing with the highest optimization level, and then we then compare the witness values, circuit depth, and CZ gate count to verify that the circuit has been optimized for IQM Emerald.



```python
from qiskit import QuantumCircuit, transpile
import pandas as pd
from pathlib import Path

# CZ-GHZ construction
def getGhzCircuit_cz(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.h(i+1)
        qc.cz(i, i+1)
        qc.h(i+1)
    return qc


# circuitry
qBits = 54 # prev value
shots = 1024 # prev value

qcz_opt = getGhzCircuit_cz(qBits)
qcz_opt.measure_all()

qcx_opt = getGhzCircuit_cz(qBits)
for i in range(qBits):
    qcx_opt.h(i)
qcx_opt.measure_all()

# emerald transpilation
qc_transpiled_z_opt = transpile(
    qcz_opt,
    backend=backend,
    optimization_level=3,
    layout_method="sabre",
    routing_method="sabre",
    seed_transpiler=7
)

qc_transpiled_x_opt = transpile(
    qcx_opt,
    backend=backend,
    optimization_level=3,
    layout_method="sabre",
    routing_method="sabre",
    seed_transpiler=7
)

job_z_opt = backend.run(qc_transpiled_z_opt, shots=shots)
result_z_opt = job_z_opt.result()

job_x_opt = backend.run(qc_transpiled_x_opt, shots=shots)
result_x_opt = job_x_opt.result()

# witness computation
zz_sum_opt = 0
for i in range(qBits - 1):
    zz_sum_opt += getZZExpectation(result_z_opt.get_counts(), i, i + 1, shots)

zz_avg_opt = zz_sum_opt / (qBits - 1)
x_val_opt = getGlobalXExpectation(result_x_opt.get_counts(), shots)
fidelity_lb_opt = 0.5 * (x_val_opt + zz_avg_opt)

print("\nHardware-optimized GHZ results:")
print(f"zz_avg = {zz_avg_opt}")
print(f"x_val  = {x_val_opt}")
print(f"fidelity lower bound = {fidelity_lb_opt}")

# Circuit metrics 
ops_z = qc_transpiled_z_opt.count_ops()
ops_x = qc_transpiled_x_opt.count_ops()

depth_z = qc_transpiled_z_opt.depth()
depth_x = qc_transpiled_x_opt.depth()

print("\nTranspiled circuit metrics:")
print("Z circuit depth:", depth_z)
print("X circuit depth:", depth_x)
print("Z ops:", ops_z)
print("X ops:", ops_x)

# other metrics
outdir = Path("results/tables")
outdir.mkdir(parents=True, exist_ok=True)

row = {
    "N": qBits,
    "shots": shots,
    "zz_avg": zz_avg_opt,
    "x_val": x_val_opt,
    "fidelity_lb": fidelity_lb_opt,
    "depth_Z": depth_z,
    "depth_X": depth_x,
    "ops_Z": str(dict(ops_z)),
    "ops_X": str(dict(ops_x)),
    "note": "cz_native_optimized"
}

metrics_path = outdir / "hardware_optimized_runs.csv"

df_row = pd.DataFrame([row])
if metrics_path.exists():
    df_prev = pd.read_csv(metrics_path)
    pd.concat([df_prev, df_row], ignore_index=True).to_csv(metrics_path, index=False)
else:
    df_row.to_csv(metrics_path, index=False)

print(f"\nSaved hardware-optimized metrics to {metrics_path}")

```


    Progress in queue:   0%|          | 0/6 [00:00<?, ?it/s]



    Progress in queue:   0%|          | 0/5 [00:00<?, ?it/s]


    
    Hardware-optimized GHZ results:
    zz_avg = 0.36147553066037735
    x_val  = 0.029296875
    fidelity lower bound = 0.19538620283018868
    
    Transpiled circuit metrics:
    Z circuit depth: 216
    X circuit depth: 216
    Z ops: OrderedDict([('r', 236), ('cz', 129), ('measure', 54), ('barrier', 1)])
    X ops: OrderedDict([('r', 274), ('cz', 129), ('measure', 54), ('barrier', 1)])
    
    Saved hardware-optimized metrics to results\tables\hardware_optimized_runs.csv


## Interpretation of the hardware-optimized results based on IQM Emerald

For the hardware-optimized run at $N = 54$ qubits, we obtained:
$$
\begin{align}
\text{zz\_avg} &\approx 0.361 \\
\langle X^{\otimes N} \rangle &\approx 0.029 \\
F_{LB} &\approx 0.195
\end{align}
$$

The transpiled circuits required approximately:
$$
\begin{align}
\text{CZ count} &\approx 129 \\
\text{Circuit depth} &\approx 216
\end{align}
$$

From these values, we noticed 2 important things:

- The nearest-neighbor parity correlations $\text{zz\_avg}$ remained finite, so many qubits were still aligned in patterns close to $00\ldots0$ or $11\ldots1$.

- The global coherence term $\langle X^{\otimes N} \rangle$ was strongly supressed, and it meant that phase coherence between the two GHZ branches has largely decayed.

As a result, the prepared state is well approximated by the classical correlated mixture
$$
\begin{align}
\rho \approx \tfrac12 \left( 
|0^{\otimes N}\rangle\langle 0^{\otimes N}| 
+ 
|1^{\otimes N}\rangle\langle 1^{\otimes N}| 
\right).
\end{align}
$$

Although our newer implementation uses the native CZ gate of IQM Emerald and reduces logical gate overhead, the circuit still requires a large number of entangling operations and has substantial depth, so at this scale, accumulated decoherence dominates the dynamics and suppresses the global coherence observable.

Compared to the non-optimized (CX-based) implementation, the hardware-optimized circuit demonstrates a measureable improvement from device aware compilation by yielding a higher fidelity lower bound at $N = 54$, but this improvement is insufficient to preserve large-scale GHZ coherence.


So, maybe the "Secret Sauce" of quantum computing was the friends we made along the way... or with our statistics of two similar but parametrically different approaches, but with the results of our second experiment showing reliable Fidelity values at the 20-qubit range and the help of IQM Emerald, maybe we've unsecreted a bit more about this secret today. Take that, Curmudgeon.
