import os
import pinecone
import numpy as np
import pandas as pd
import plotly.express as px
import pacmap_plot
from pacmap import PaCMAP

from dotenv import load_dotenv
load_dotenv()

# 1. Initialize Pinecone
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],  # or paste your key as a string
    environment=os.environ["PINECONE_ENVIRONMENT"]  # e.g. "us-west4-gcp"
)

# 2. Connect to your index (replace with your actual index name)
index = pinecone.Index("pdf-vectorized")

# 3. List of vector IDs you want to visualize
#    Typically, you'd have saved these IDs when uploading your document chunks.
all_ids = [
    "7d3ba9e8-1f05-4282-9cea-127db8eeeaf1",
    "dd0489974-ed50-45bf-be23-86004101547a",
    "dcd24ba4b-127c-438e-99f3-132884df25a8",
    '555e88e5-9c03-4520-8ea6-29168c64ac29',
    '6c8520fb-877a-4ae2-b4cd-01d6df4e3984',
    'f7974cc7-14cc-4da3-96fa-d9d44f94f451',
    '609c2b0d-64b0-431b-9bc4-2086a5b64ca0',
    '460ec36e-5ead-404f-8355-5c7563b4b1e2',
    '95672185-8b91-4be3-ac0e-2cd38e9a39f2',
    '3871165a-597d-4a4d-ba14-8311a32dfc19'
    # ... etc ...
]

fetch_res = index.fetch(ids=all_ids)

# 5. Extract the vectors and metadata into lists
vectors = []
metadata_list = []
for vector_id, record in fetch_res["vectors"].items():
    vectors.append(record["values"])         # The embedding array
    metadata_list.append(record["metadata"]) # Dict with fields like {"source": "DocA", ...}

# Convert vectors to a NumPy array
embeddings = np.array(vectors)  # shape (N, embedding_dim)

# 6. Reduce dimensions with PaCMAP (to 2D)
pacmap_model = PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
reduced_embeddings = pacmap_model.fit_transform(embeddings, init="pca")
# reduced_embeddings.shape == (N, 2)

# 7. Create a DataFrame for plotting
df = pd.DataFrame({
    "x": reduced_embeddings[:, 0],
    "y": reduced_embeddings[:, 1],
    # Suppose each chunk has a "source" field in metadata to identify the document
    "source": [m.get("source", "unknown") for m in metadata_list],
    # You can store the chunk text for hover info if you have it
    "chunk_text": [m.get("text", "") for m in metadata_list]
})

# 8. Plot with Plotly
#    - color: groups points by their source doc
#    - hover_data: shows chunk_text on hover, if available
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="source",
    hover_data=["chunk_text"],
    title="2D Projection of Chunk Embeddings via PaCMAP"
)
fig.update_layout(xaxis_title="X", yaxis_title="Y")
fig.savefig('plot_pac.png')
