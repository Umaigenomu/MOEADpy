__all__ = ["constraints", "decomposition", "neighboorhood",
           "scalarization", "termination", "update", "variation"]

class Component:
    def __call__(self):
        raise NotImplementedError("This 'Component' class hasn't implemented the method '__call__' yet.")
        pass
