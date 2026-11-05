"""Open an optimization analysis history file and write its headers."""

# After each analysis, write one row of numerical values.
#    row = np.concatenate((c._v, c._r, [f], g))
#    np.savetxt(history_file, row.reshape(1, -1), delimiter=",", fmt="%.17g")

from datetime import datetime

def opt_open_history(L, M, N):
    """Return an open history file with timestamp and column headers.

    now is a datetime object; L, M, and n are the numbers of random
    variables, design constraints, and design variables.
    The caller appends numerical rows and closes the returned file.
    The filename contains a colon and is intended for Linux or macOS.
    Read the completed file with:
        np.loadtxt(filename, delimiter=",", ndmin=2)
    """
    now = datetime.now()
    filename = now.strftime("analysis_file_%Y%m%d-%H%M%S.csv")
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

    return history_file
