"""Main script for the thesis project."""

import sys
import pprint


## Local imports
from thesis2024.datamodules.crag import Crag













if __name__ == "__main__":


    
    # Build the graph
    crag_class = Crag()
    app = crag_class.build_rag_graph()
    print(app)
    # Run the graph
        # Run
    inputs = {"keys": {"question": "Who is the teacher of the machine learning course, and how come the highest mountains are located in asia?"}}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    # Final generation
    pprint.pprint(value["keys"]["generation"])

    print("Graph run complete.")
    # sys.exit(0)

