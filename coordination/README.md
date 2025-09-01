## Coordination
This code is adapted and cleaned from https://github.com/ValeriaPante/coordinatedActivity/

A number of issues were addressed in this new code:
- Some variables were referenced in functions but were missing in the function parameters
- Code was inconsistent when refering to names of data variables (e.g., user IDs, tweet IDs)
- Code assumed a treatment and control group, which is not needed when we typically have only data with unknown labels
- Code was sometimes slow when we used FAISS with low cosine similarity values
- Code often created self-loops, which we now remove from analysis
