import numpy as np

####################################################################################################
# Exercise 1: DFT

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    # TODO: initialize matrix with proper size
    F = np.zeros((n, n), dtype='complex128')

    # TODO: create principal term for DFT matrix
    o = np.exp((-2j * np.pi)/n)
    # TODO: fill matrix with values
    for x in range(0, n):
        for y in range(0, n):
            F[x, y] = o**(x * y)
    temp = 1/np.sqrt(n)
    F = temp * F.transpose()
    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    if not np.allclose(np.eye(matrix.shape[0]), np.dot(matrix, np.transpose(np.conjugate(matrix)))):
        return False

    return True


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []

    # TODO: create signals and extract harmonics out of DFT matrix
    for x in range(0, n):
        e = np.eye(n)[:, x]
        sigs.append(e)
    delt = dft_matrix(n)
    for x in range(0, n):
        fsigs.append(delt.dot(sigs[x]))
    return sigs, fsigs


####################################################################################################
# Exercise 2: FFT

def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """

    # TODO: implement shuffling by reversing index bits
    n = len(np.binary_repr(int(data.shape[0] - 1)))

    for x in range(0, data.shape[0]):
        neu_p = int(np.binary_repr(x, width=n)[::-1], 2)
        if x < neu_p:
            data[x], data[neu_p] = data[neu_p], data[x] #swap

    return data


def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """

    set_output = True

    # Setze Eingangsdaten für Ausgabe um mit Anhang zu Vergleichen
    if set_output:
        # data    = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        data = [1, 0, 0, 0, 0, 0, 0, 0]

    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size

    # check if input length is power of two
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    ### first step of FFT: shuffle data
    # Verwende Funktion aus Aufgabe 2.1
    fdata = shuffle_bit_reversed_order(fdata)

    ## Ausgabe von fdata nach shuffel
    if set_output:
        print("\nNach shuffel: " + str(fdata))

    ### second step, recursively merge transforms
    # Durchlaufe alle Ebenen des Baums; Beginnend mit 0 bis log2( n ); log2( n ) weil paarweise auf Blattebene bis zur Wurzel kombiniert wird
    for m in range(0, int(np.log2(n))):
        # Durchlaufe alle Elemente der aktuellen Ebene im Intervall [0 ... 2^m[
        for k in range(0, 2 ** m):
            # Berechne den omega Faktor für das aktuelle k; in Python wird für die imaginäre Zahl das in der Elektrotechnik gebräuchliche j verwendet.
            omega = np.exp((-2j * np.pi * k) / (2 ** (m + 1)))

            # Für alle Werte i, j mit i = [k, k + x * 2^(m+1), k + x * 2^(m+1), ..., n[ mit x = [1...n] je Imkrementierung von i
            for i in range(k, n, 2 ** (m + 1)):
                ## Führe elementare Transformation durch
                # Berechne j anhand des Abstands 2^m von i zu j
                j = i + 2 ** m

                # Berechne p
                p = omega * fdata[j]

                # Berechne f[j]
                fdata[j] = fdata[i] - p

                # Berechne f[i]
                fdata[i] = fdata[i] + p

                ## Ausgaben zum Vergleich mit Anhang
                if set_output:
                    # Ausgabe aktuelle Parameter
                    print("m = " + str(m) + ", k = " + str(k) + ", i = " + str(i) + ", j = " + str(j))

                    # Ausgabe des reelen Anteils der Berechnungsergebnisse
                    print(fdata)

    ### normalize fft signal
    # Normalisierung gemäß Vorlesung
    fdata /= np.sqrt(n)

    return fdata


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0
    dis = x_max - x_min

    data = np.zeros(int(num_samples * dis))
    # TODO: Generate sine wave with proper frequency
    for i in range(0, data.shape[0]):
        data[i] = np.sin((2.0 * np.pi * f * i) / (num_samples - 1))
    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """
    
    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit*adata.size/sampling_rate)

    # TODO: compute Fourier transform of input data
    fdata = fft(adata)

    t = (fdata.shape[0]-bandlimit_index)
    # TODO: set high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.
    for i in range(0, fdata.shape[0]):
        if bandlimit_index < i < t:
            fdata[i] = 0
    # TODO: compute inverse transform and extract real component
    conj = np.conjugate(fft(np.conjugate(fdata)))
    adata_filtered = np.zeros(adata.shape[0])
    adata_filtered = np.real(conj)
    return adata_filtered


if __name__ == '__main__':
    print(fft(True))
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
