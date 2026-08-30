def disable_logfile_if_none(optimizer, logfile) -> None:
    if logfile is not None:
        return
    current = getattr(optimizer, 'logfile', None)
    close = getattr(current, 'close', None)
    if close is not None:
        close()
    optimizer.logfile = None


def flush_logfile(logfile) -> None:
    flush = getattr(logfile, 'flush', None)
    if flush is None:
        return
    cls = logfile.__class__
    if cls.__module__ == 'ase.optimize.optimize' and cls.__name__ == 'Log':
        return
    try:
        flush()
    except TypeError:
        pass
