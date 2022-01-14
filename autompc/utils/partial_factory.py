

def apply_partial(factory, instance_class, num_args, args, kwargs):
        if len(args) < num_args:
            return PartialFactory(factory, instance_class, num_args, args, kwargs)
        else:
            return factory.create(*args, **kwargs)

class PartialFactory:
    def __init__(self, factory, instance_class, num_args, args, kwargs):
        self.factory = factory
        self.instance_class = instance_class
        self.num_args = num_args
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        args = self.args + args
        kwargs.update(self.kwargs)
        return apply_partial(self.factory, self.instance_class, self.num_args, args, kwargs)
