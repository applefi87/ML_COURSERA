import numpy as np
import fractions

def float_to_fraction(float_value, denominator=9999999):
    """Convert a floating-point number to a fraction with a specified denominator."""
    numerator = round(float_value * denominator)
    return fractions.Fraction(numerator, denominator)

def linspace_fraction(start, stop, num, denominator=9999999):
    """Generate a linearly spaced array of fractions between start and stop."""
    float_linspace = np.linspace(start, stop, num)
    fraction_linspace = [float_to_fraction(f, denominator) for f in float_linspace]
    return np.array(fraction_linspace, dtype=object)
