import matplotlib.pyplot as plt
from fraction_utils import float_to_fraction, linspace_fraction
import numpy as np

# Usage example:
start = -10
stop = 10
num = 50

x = linspace_fraction(start, stop, num)
x_float = [float(f) for f in x]  # Convert fractions to float for computation

y = ((2)**np.array(x_float))

y_real = [f.real for f in y]
y_imag = [f.imag for f in y]
y_magnitude = [abs(f) for f in y]
def handle_nan(value):
    return value if not np.isnan(value) else 0.0  # Replace NaN with 0.0
# Handle NaN values and round to 10 decimal places
y_rounded = [round(handle_nan(f.real), 10) + round(handle_nan(f.imag), 10)*1j for f in y]

print(y_rounded)

plt.figure()
plt.scatter(x_float, y_rounded, label='Real part of y = ((-2)^x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = ((-2)^x)')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
