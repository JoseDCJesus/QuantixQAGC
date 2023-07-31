Git Repository for team Quantux:
Karim Elgammal

Phung Cheng Fei

José Diogo da Costa Jesus

Marc Maussner

Tuomas Sorakivi

Jesmer Wong

The file answer.py in folder problem contains our answer. In the same folder, the presentation notebook shows how we got to our solution. Our method is based on VQE (variational quantum eigensolver) with ZNE (zero noise extrapolation).

Project Description:
The goal of this project is to calculate the eigenstate energy for some Hamiltonians. To do this, we first crated an appropriate ansatz, havily inspired on the hardware efficient anstaz. Then we implemented the VQE algorithm using qiskit and quri-parts. We use multiple shots to average out natural variance and noise. 
To cut through the noise we implemented a linear ZNE extrapolation. The final answer is an average of the local minima found during the VQE process.
