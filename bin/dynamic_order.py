Lmax = 4 # maxiaml abstraction level
K = 2  # abstraction ratio
	
def generate_level(l: int, curr: str, t: int): 
    if l <= Lmax:
        curr += str(l)
        if t % K == 0: 
            return generate_level(l+1, curr, t // K)
    return curr
		
total_str = ""
for t in range(1, 2**4+1):
	t_str = generate_level(0, "", t)
	total_str += t_str
	print(f"Step {t} string: {total_str}")  