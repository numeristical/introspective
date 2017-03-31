def _gca():
    import matplotlib.pyplot as plt
    return plt.gca()


def is_classifier(estimator):
    """Returns True if the given estimator is (probably) a classifier."""
    return getattr(estimator, "_estimator_type", None) == "classifier"


def is_regressor(estimator):
    """Returns True if the given estimator is (probably) a regressor."""
    return getattr(estimator, "_estimator_type", None) == "regressor"

#def is_classifier2(model):
#    """Returns True if the given estimator is (probably) a classifier."""
#    return (len(getattr(model, "classes_", [])) >0)
