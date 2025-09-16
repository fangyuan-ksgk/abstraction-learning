# NIL algorithm
# Within each generation, we inherit prev generation's abstraction, and continue training 
# Reset -> Weak-supervision -> Train -> Record 
# ------------------------------------------------------------------------------------------------

def record_abstraction(gat: GAT, dataset): 
    """Inferece on dataset & label each trajectory with abstraction"""
    raise NotImplementedError("Not implemented")

# Weak-supervision involves direct supervision on abstraction-labeled trajectory dataset
# controlling iteration number to avoid overfitting & underfitting


