# -*- coding: utf-8 -*-
"""
# Projet 2 Analyse

Projet réalisé par :

  Papa Ba GAYE
  Ndeye Mareme NGOM
  Cherif Assane Fall MBENGUE
  Mamadou NGOM
  Ibrahima CAMARA
  Abdou Karim SOW
  Shamsidine DIATTA
  Adama CISSE
  

**Importation des modules**
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert

"""1 - Fonction **cramer**"""

def cramer(matrice, vecteur):
  if matrice.shape[0] == vecteur.shape[1]:
    n = matrice.shape[0]
    Mat = []
    sol = []
    for i in range(n):
      Mat.append(matrice.copy().T)
    
    for i in Mat:
      for j in range(n):
        Mat[j][j] = vecteur
    detP = np.linalg.det(matrice)

    for i in range(len(Mat)):
      sol.append(np.linalg.det(Mat[i]) / detP)
    sol = np.array([sol])
    return sol
  else:
    print('Le nombre de ligne de la matrice A doit etre égale au nombre de colonne deu vecteur b')

# Creation de la matrice de hilbert
matrice_hil = hilbert(5)

vect = np.array([[1]*5])

print('Solution cramer')
print(cramer(matrice_hil, vect))

"""2 - Fonction **decente**"""

def decente_lu(A, b):
  A = np.tril(A, k=0)
  A[np.diag_indices_from(A)] = 1
  return cramer(A, b)

A = np.array([[3, 2, 1],
              [5, 1, -1],
              [1, -3, 5]])

B = np.array([[1, 6, -4]])

print("Solution decente_lu")
print(decente_lu(A, B))

"""3 - Fonction **remontee**"""

def remontee(A, b):
  A = np.triu(A, k=0)
  return cramer(A, b)

print("Solution remontee")
print(remontee(A, B))

"""4 - Fonction **facto_LU**"""

def facto_LU(A):
  n = A.shape[0]
  U = np.copy(A)
  L = np.eye(n)
  for i in range(n) :
    p = U[i,i]
    for j in range(i+1, n):
      L[j,i] = U[j,i]/p
      U[j] = U[j] - L[j,i] * U[i]
  return L,U

X = np.array([[2., 4., 5., 6.],
              [-1., 2., 8.5, 1.],
              [3., 8., 3., -3.],
              [5., 2., 1.5, 6.4]])

print("Factorisation LU")
Q = facto_LU(X)
print(f'L : \n{Q[0]}')
print('')
print(f'U : \n{Q[1]}')

# Vérification par le produit matriciel
print("Resultat Verification")
print(Q[0] @ Q[1])

"""5 - **Fonction resol_LU** <br>
Avec la factorisation sous forme LU l'équation Ax = <=> LUx = b.
On pose Ux = y => Ly = b


"""

def resol_LU(A, b):
  LU = facto_LU(A)
  desc = decente_lu(LU[0], b)
  solution = remontee(LU[1], desc)
  return solution

A = np.array([[0.3, 0.52, 1],
              [0.5, 1, 1.9],
              [0.1, 0.3, 0.5]])

B = np.array([[-0.01, 0.67, -0.44]])

print("Solution resol_LU")
print(resol_LU(A, B))

"""## **Question6** """

tab_error1 = []
for n in range(1,26):
  # print(f"value of n {n}")
  mat = hilbert(n)
  vector = np.array([[1]*n])
  sol1 = cramer(mat,vector)
  error1 = np.linalg.norm(sol1-vector)
  tab_error1.append(error1)
  # print(f"Error with cramer {error1}")

plt.figure(figsize=(15, 8))
plt.plot(tab_error1)
plt.title('Erreurs en norme avec la formule de Cramer')
plt.show()

tab_error2 = []
for n in range(1,26):
  # print(f"value of n {n}")
  mat = hilbert(n)
  vector = np.array([[1]*n])
  sol1 = resol_LU(mat,vector)
  error1 = np.linalg.norm(sol1-vector)
  tab_error2.append(error1)
  # print(f"Error with cramer {error2}")

plt.figure(figsize=(15, 8))
plt.plot(tab_error2)
plt.title('Erreurs en norme avec la factorisation LU')
plt.show()
