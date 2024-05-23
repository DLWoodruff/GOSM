"""
This is an incomplete example script which is untested and currently serves
the purpose of demonstrating how the classes in mesh.py can be used.
"""


if __name__ == '__main__':
    # Suppose f is a function of 2 variables that is predefined
    curr_mesh = CubeMesh([0, 0], capacity, n)
    while True:
        outer_shell = mesh.outer_shell()
        outer_sum = outer_shell.integrate_with(f)
        if outer_sum < epsilon:
            break
        curr_mesh = curr_mesh.add_shell()


