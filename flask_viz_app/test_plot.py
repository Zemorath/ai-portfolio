import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3])
plt.savefig("test.png")
plt.close()