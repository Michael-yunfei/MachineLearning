# Gradient Descent: mathematical example
# @ Michael


# define function
def qudraticfun(x):
    y = 1/2 * x**2 - 3 * x + 3
    return(y)


# define the derivative
def qudraticder(x):
    yprime = x - 3
    return(yprime)


# find the minium numerically

update_x = 0  # most time we start it from zero
alpha = 0.01  # learning rate
tolerate_rule = 1  # initialize the tolerate rule for stopping iterating
max_iters = 10000  # set the maximum iterative number

i = 0  # interation counting index
while tolerate_rule >= 0.00001 and i <= max_iters:
    start_x = update_x  # set the starting value
    update_x = start_x - alpha * qudraticder(start_x)
    tolerate_rule = abs(update_x - start_x)

print("The minimum value is", qudraticfun(update_x),
      "when x is equal to", update_x)

# The minimum value is -1.4999995136117308 when x is equal to 2.999013705653870
