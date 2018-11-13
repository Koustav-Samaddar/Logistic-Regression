
def time_to_str(t):
	if t > 3600:
		return "{0:d}h {1:d}m {2:.3f}s".format(t // 3600, (t % 3600) // 60, t % 60)
	else:
		return "{0:d}m {1:.3f}s".format(t // 60, t % 60)
