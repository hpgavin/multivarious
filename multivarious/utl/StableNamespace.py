class StableNamespace:
    """
    A multivarious namespace object with a fixed set of mutible attributes.

    Attributes can be accessed and updated using dot notation.
    Once created, the set of attributes is fixed:
     adding new attributes is not allowed,
     but existing attribute values can be changed freely.

    This provides a lightweight, mutable container with a consistent attribute
    structure, useful for configurations, constants, or grouped data where the
    attribute keys, once created, do not change. 

    Example:
        >>> ns = StableNamespace(a=1, b=2, c=[3, 4])
        >>> print(ns.a)
        1
        >>> ns.b = 5
        >>> ns.c.append(6)
        >>> print(ns.c)
        [3, 4, 6]
        >>> print(ns)
        StableNamespace(a=1, b=5, c=[3, 4, 6])
        >>> ns.d = 7
        Traceback (most recent call last):
            ...
        AttributeError: Cannot add new attribute 'd' to StableNamespace
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._locked = True

    def __setattr__(self, key, value):
        if key == '_locked':
            super().__setattr__(key, value)
            return
        if getattr(self, '_locked', False) and key not in self.__dict__:
            raise AttributeError(f"Cannot add new attribute '{key}' to StableNamespace")
        super().__setattr__(key, value)

    def __repr__(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items() if k != '_locked')
        return f"StableNamespace({attrs})"


