""" Open an optimization analysis history .csv file and write its headers. """

from datetime import datetime

def opt_open_history(L, M, N):
    """
    Return an open history file with timestamp and column headers.

    history_file, filename = opt_open_history(L, M, N)

    Input
    =====
    L : the number of random variables, r
    M : the number of design constraints, g
    N : the number of design variables, f

    Output
    ======
    history_file : the open file object 
    filename : the file name as analysis_record-YYYYMMDD-HHMMSS.csv

    Append numerical rows ...

        vrfg = np.concatenate((c._v, c._r, [f], g))
        np.savetxt(c._history_file, vrfg.reshape(1, -1), delimiter=",", fmt="%.17g")
        c._history_file.flush() 

    ... and close the returned file ...

        if c._history_file is not None:
            c._history_file.close()
            c._history_file = None

    ... read the completed file  ...
        data = np.loadtxt(filename, delimiter=",", ndmin=2)
    """
    now = datetime.now()
    filename = now.strftime("analysis_record_%Y%m%d-%H%M%S.csv")
    history_file = open(filename, "w", encoding="utf-8")

    try:
        history_file.write(
            now.strftime("# history of analysis data %Y-%m-%d %H:%M:%S\n")
        )
        labels = (
            [f"v{i+1}" for i in range(N)]
            + [f"r{i+1}" for i in range(L)]
            + ["f"]
            + [f"g{i+1}" for i in range(M)]
        )
        history_file.write("# " + ", ".join(labels) + "\n")
        history_file.flush()
    except BaseException:
        history_file.close()
        raise

    return history_file, filename
