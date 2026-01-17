import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Your code here
	def forward(f, x):
        if f == "square":
            return x ** 2
        if f == "sin":
            return np.sin(x)
        if f == "exp":
            return np.exp(x)
        return np.log(x)
    def backward(f, x):
        if f == "square":
            return 2 * x
        if f =="sin":
            return np.cos(x)
        if f == "exp":
            return np.exp(x)
        return 1 / x
    interm = []
    for f in functions[::-1]:
        interm.append(x)
        x = forward(f, x)
    ans = 1
    f