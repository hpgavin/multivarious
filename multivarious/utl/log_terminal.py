""" Write terminal output to a log file. """

from datetime import datetime

def log_open(file_name_prefix): 
    """
    Return an open log file with timestamp.

    log_file, log_filename = open_log(file_name_prefix)

    Input
    =====
    file_name_prefix

    Output
    ======
    log_file     : the open log file object 
    log_filename : the file name as file_name_prefix-log-YYYYMMDD-HHMMSS.txt
    """

    now = datetime.now()
    calendar_clock = now.strftime("log_%Y%m%d-%H%M%S.txt")
    log_filename = f'{file_name_prefix}-{calendar_clock}'
    log_file = open(log_filename, "w", encoding="utf-8")

    log_file.write(f'# {log_filename}\n')
    log_file.flush()

    return log_file, log_filename


def ltw(file, text): 
    """
    Write text to the termiminal and to file
    """

    print(text)
    file.write(text)
    file.write('\n')
    file.flush()

